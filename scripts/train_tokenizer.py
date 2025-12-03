import argparse
from einops import rearrange
import os
from safetensors.torch import load_file
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import wandb

from vito.common.checkpoint import get_last_ckpt, resume_from_ckpt
from vito.common.common_utils import set_random_seed
from vito.common.config import VitoConfig
from vito.common.logger import vito_logger
from vito.evaluation.psnr import get_psnr
from vito.evaluation.lpips import get_lpips
from vito.evaluation.ssim import get_ssim_and_msssim
from vito.data.vanilla_video_dataset import VideoData
from vito.infra.distributed.dist_utils import dist_init, reduce_losses
from vito.loss.disc_loss import adopt_weight, get_disc_loss, lecam_reg_zero
from vito.loss.lpips import LPIPS
from vito.model.discriminator import ImageDiscriminator
from vito.model.vae.vae_model import ViTVAE


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run video tokenizer")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file")
    parser.add_argument("--default_root_dir", type=str, required=True, help="Default root path to save checkpoints")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint to resume from")
    parser = VideoData.add_data_specific_args(parser)
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = VitoConfig.from_json(args.config_file)
    set_random_seed(42)
    dist_init(config)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    vito_logger.info(f"{config=}")

    if rank == 0:
        os.makedirs(args.default_root_dir, exist_ok=True)
        checkpoint_dir = os.path.join(args.default_root_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        if args.use_wandb:
            wandb.init(
            entity="zephyrus",
            project="vito",
            name=os.path.basename(os.path.normpath(args.default_root_dir)),
            dir=args.default_root_dir,
            config=args,
            mode="offline" if args.debug else "online"
        )

    # init data
    data = VideoData(args)
    dataloader = data.train_dataloader()
    if not hasattr(dataloader.sampler, 'set_epoch'):
        vito_logger.warning("Dataloader sampler does not support set_epoch, skipping epoch update")
    dataloader_iter = iter(dataloader)
    val_dataloader = data.val_dataloader()
    val_dataloader_iter = iter(val_dataloader)
    data_epoch = 0

    # init model
    d_vae = ViTVAE(
        config.vae_config.encoder_config,
        config.vae_config.decoder_config,
        pretrained=config.vae_config.pretrained,
        model_type=config.vae_config.model_type,
    ).to(device)
    image_disc = ImageDiscriminator(config.disc_config).to(device)    

    opt_vae = torch.optim.AdamW(d_vae.parameters(), lr=config.optim_config.lr)
    opt_disc = torch.optim.AdamW(image_disc.parameters(), lr=config.optim_config.lr * config.disc_config.disc_lr_multiplier)

    model_optims = {
        "d_vae": d_vae,
        "image_disc": image_disc,
        "opt_vae": opt_vae,
        "opt_disc": opt_disc,
    }

    if args.evaluate:
        assert args.checkpoint is not None, "Checkpoint must be provided for evaluation mode"
        if args.checkpoint.endswith("safetensors"):
            state_dict = load_file(args.checkpoint)
            d_vae.load_state_dict(state_dict)
        else:
            state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            d_vae.load_state_dict(state_dict["d_vae"])
        d_vae = DistributedDataParallel(d_vae, device_ids=[device], output_device=device)
        evaluate(d_vae, val_dataloader_iter)
        return
    # resume from default_root_dir
    ckpt_path = None
    assert not args.default_root_dir is None 
    ckpt_path = get_last_ckpt(args.default_root_dir)
    init_step = 0
    if ckpt_path:
        vito_logger.info(f"Resume from checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model_optims, init_step = resume_from_ckpt(state_dict, model_optims, load_optimizer=True)

    d_vae = DistributedDataParallel(d_vae, device_ids=[device], output_device=device)
    image_disc = DistributedDataParallel(image_disc, device_ids=[device], output_device=device)
    perceptual_model = LPIPS().to(device)
    disc_loss = get_disc_loss(config.disc_config.disc_loss_type)

    start_time = time.time()
    for global_step in range(init_step, config.optim_config.max_steps):

        if global_step == config.disc_config.discriminator_iter_start - config.disc_config.disc_pretrain_iter:
            vito_logger.info("Starting discriminator pretraining")
        if global_step == config.disc_config.discriminator_iter_start:
            log_str = "add GAN loss into training"
            if config.disc_config.disc_pretrain_iter > 0:
                log_str += f", discriminator ends pretraining"
            vito_logger.info(log_str)

        try:
            batch = next(dataloader_iter)
        except StopIteration:
            data_epoch += 1
            vito_logger.info(f"Starting epoch {data_epoch}")
            dataloader.sampler.set_epoch(data_epoch)
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        videos = batch["video"].to(device)  # (B, T, C, H, W)
        videos = videos.permute(0, 2, 1, 3, 4)

        # VAE forward with mixed precision
        vae_loss_dict = {}
        opt_vae.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            recon_videos, posteriors = d_vae(videos)
            recon_loss = F.l1_loss(recon_videos, videos)
            recon_videos_in_frames = rearrange(recon_videos, "b c t h w -> (b t) c h w")
            videos_in_frames = rearrange(videos, "b c t h w -> (b t) c h w")
            lpips_loss = perceptual_model(recon_videos_in_frames, videos_in_frames).mean()
            kl_loss = posteriors.kl().mean()
            vae_loss_dict["train/recon_loss"] = recon_loss.detach()
            vae_loss_dict["train/lpips_loss"] = lpips_loss.detach()
            vae_loss_dict["train/kl_loss"] = kl_loss.detach()
            disc_factor = adopt_weight(global_step, threshold=config.disc_config.discriminator_iter_start, warmup=config.disc_config.disc_warmup_steps)
            if config.disc_config.disc_gan_weight > 0:
                logits_image_fake = image_disc(recon_videos_in_frames)
                g_loss = -torch.mean(logits_image_fake) * disc_factor * config.disc_config.disc_gan_weight
            else:
                g_loss = torch.tensor(0.0).to(device)
            vae_loss_dict["train/g_image_loss"] = g_loss.detach()
            vae_loss = recon_loss
            vae_loss += lpips_loss * config.optim_config.lpips_weight
            vae_loss += kl_loss * config.optim_config.kl_weight

        # VAE backward
        vae_loss.backward()

        if config.optim_config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(d_vae.parameters(), config.optim_config.max_grad_norm)
        opt_vae.step()
        opt_vae.zero_grad()

        disc_loss_dict = {}
        disc_factor = adopt_weight(global_step, threshold=config.disc_config.discriminator_iter_start - config.disc_config.disc_pretrain_iter)
        for disc_step in range(config.disc_config.disc_optim_steps):
            if config.disc_config.disc_disc_weight > 0:
                opt_disc.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits_real = image_disc(videos_in_frames, pool_name="real")
                    logits_fake = image_disc(recon_videos_in_frames.detach(), pool_name="fake")
                    d_loss = disc_loss(logits_real, logits_fake)
                    disc_loss_dict["train/logits_image_real"] = logits_real.mean().detach()
                    disc_loss_dict["train/logits_image_fake"] = logits_fake.mean().detach()
                    disc_loss_dict["train/d_image_loss"] = d_loss.detach()
                    disc_loss_val = d_loss * config.optim_config.disc_weight
                    if global_step >= config.disc_config.discriminator_iter_start and config.disc_config.use_lecam_reg_zero:
                        lecam_zero_loss = lecam_reg_zero(logits_real.mean(), logits_fake.mean())
                        disc_loss_dict["train/lecam_zero_loss"] = lecam_zero_loss.detach()
                        disc_loss_val += lecam_zero_loss * config.disc_config.lecam_weight
                    disc_loss_val = disc_loss_val * disc_factor

                disc_loss_val.backward()
                if config.optim_config.max_grad_norm_disc > 0:
                    torch.nn.utils.clip_grad_norm_(image_disc.parameters(), config.optim_config.max_grad_norm_disc)
                opt_disc.step()
                opt_disc.zero_grad()
            
        loss_dict = {**vae_loss_dict, **disc_loss_dict}
        if (global_step + 1) % args.log_every == 0:
            reduced_loss_dict = reduce_losses(loss_dict)
        else:
            reduced_loss_dict = {}

        if (global_step + 1) % args.log_every == 0:
            torch.cuda.synchronize()
            end_time = time.time()
            iter_speed = (end_time - start_time) / args.log_every
            if rank == 0:
                for k, v in reduced_loss_dict.items():
                    wandb.log({k: v}, step=global_step+1) if args.use_wandb else None
                vito_logger.info(f"Step {global_step+1} | Iteration Speed: {iter_speed:.4f}s/step")
            start_time = time.time()

        if (global_step + 1) % args.save_every == 0 and global_step != init_step:
            if rank == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'model_step_{global_step}.ckpt')
                save_dict = {}
                for k in model_optims:
                    save_dict[k] = None if model_optims[k] is None \
                        else model_optims[k].module.state_dict() if hasattr(model_optims[k], "module") \
                        else model_optims[k].state_dict()
                torch.save({
                    'step': global_step,
                    **save_dict,
                }, checkpoint_path)
                vito_logger.info(f"Saved checkpoint to: {checkpoint_path}")

def evaluate(d_vae, dataloader_iter):
    # Run evaluation: iterate through the provided dataloader iterator and perform inference.
    # d_vae can be a DistributedDataParallel wrapper; get device from model parameters.
    device = next(d_vae.parameters()).device if hasattr(d_vae, "parameters") else torch.device("cuda")
    psnr_list, lpips_list, ssim_list, msssim_list = [], [], [], []
    d_vae.eval()
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for batch in dataloader_iter:
                videos = batch["video"].to(device)  # (B, T, C, H, W)
                videos = videos.to(torch.bfloat16)
                videos = videos.permute(0, 2, 1, 3, 4)
                recon_videos, _ = d_vae(videos)
                psnr = get_psnr(videos, recon_videos, zero_mean=True, is_video=True)
                lpips = get_lpips(videos, recon_videos, zero_mean=True, network_type='alex', is_video=True)
                ssim, msssim = get_ssim_and_msssim(videos, recon_videos, zero_mean=True, is_video=True)
                gathered_psnr = [torch.zeros_like(psnr) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_psnr, psnr)
                psnr_list.extend(gathered_psnr)
                gathered_lpips = [torch.zeros_like(lpips) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_lpips, lpips)
                lpips_list.extend(gathered_lpips)
                gathered_ssim = [torch.zeros_like(ssim) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_ssim, ssim)
                ssim_list.extend(gathered_ssim)
                gathered_msssim = [torch.zeros_like(msssim) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_msssim, msssim)
                msssim_list.extend(gathered_msssim)
    psnr_list = torch.cat(psnr_list)
    lpips_list = torch.cat(lpips_list)
    ssim_list = torch.cat(ssim_list)
    msssim_list = torch.cat(msssim_list)
    if dist.get_rank() == 0:
        print(f"PSNR={psnr_list.mean().item()}")
        print(f"LPIPS (AlexNet) = {lpips_list.mean().item()}")
        print(f"SSIM = {ssim_list.mean().item()}")
        print(f"MSSSIM = {msssim_list.mean().item()}")
        vito_logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
