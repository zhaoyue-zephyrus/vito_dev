import argparse
import math
from einops import rearrange
import glob
import imageio
import os.path as osp
import time
import torch
from torchcodec.decoders import set_cuda_backend, VideoDecoder

from vito.common.common_utils import set_random_seed
from vito.common.config import VitoConfig
from vito.evaluation.lpips import get_lpips
from vito.evaluation.psnr import get_psnr
from vito.evaluation.ssim import get_ssim_and_msssim
from vito.infra.distributed.dist_utils import dist_init
from vito.pipeline.video_process import VaeHelper

filename = "/simurgh/group/TokenBench/tokenbench/panda70m_test_0000000_00000.mp4"


def parse_arguments():
    parsers = argparse.ArgumentParser(description="Run video tokenizer")
    parsers.add_argument("--config_file", type=str, required=True, help="Path to the config file")
    parsers.add_argument("--video_path", type=str, required=False, default=filename, help="Path to the video file")
    parsers.add_argument("--input_folder", type=str, required=False, default=None, help="Path to load the original videos")
    parsers.add_argument("--output_folder", type=str, required=False, default=None, help="Path to save the reconstructed videos")
    parsers.add_argument("--padded_length", type=int, default=12)
    return parsers.parse_args()


def main():
    args = parse_arguments()
    config = VitoConfig.from_json(args.config_file)
    set_random_seed(42)
    dist_init(config)

    vae = VaeHelper.get_vae("downloads/vae")

    if args.input_folder is not None:
        video_list = glob.glob(args.input_folder + "/*.mp4")
        if args.output_folder is not None:
            done_list = glob.glob(args.output_folder + "/*.mp4")
        else:
            done_list = []
        print(len(video_list))
        # video_list = video_list[:2]
    else:
        video_list = [args.video_path, ]
        done_list = []

    for filename in video_list:
        basename = filename.split('/')[-1]
        if any([basename in done_video for done_video in done_list]):
            print(f"{basename} done already; skipping")
            continue
        with set_cuda_backend("beta"):
            dec = VideoDecoder(filename, device="cuda")

        batch = dec.get_frames_at(list(range(0, dec._num_frames, 1)))
        fps = dec._num_frames / (dec._end_stream_seconds - dec._begin_stream_seconds)
        print(f"{batch.data.shape=}")

        video = batch.data / 127.5 - 1.0  # to [-1, 1]
        video = video.permute(1, 0, 2, 3) # TCHW to CTHW
        video = video.unsqueeze(0)  # add batch dim
        video_pm1 = video
        video = video.bfloat16()
        if video.shape[2] % args.padded_length != 0:
            num_chunks = math.ceil(video.shape[2] / args.padded_length)
            last_frame = video[:, :, -2:-1, :, :]
            video = torch.cat([
                video, last_frame.repeat(1, 1, num_chunks * args.padded_length - video.shape[2], 1, 1)
            ], dim=2)
            print(video.shape)
        tic = time.time()
        chunks = vae.tiled_encode_3d(
            video,
            tile_sample_min_length=12,
            tile_sample_min_height=256,
            tile_sample_min_width=256,
            spatial_tile_overlap_factor=0.25,
            temporal_tile_overlap_factor=0,
            allow_spatial_tiling=True,
            parallel_group=None,
        )
        print(f"{basename}, {chunks.shape=}")
        print(f"tiled_encode_3d | {time.time() - tic}")
        tic = time.time()
        recon_pm1 = []
        for t in range(0, chunks.shape[2], args.padded_length):
            chunk = chunks[:, :, t : t + args.padded_length, :, :]
            chunk = vae.tiled_decode_3d(
                chunk,
                tile_sample_min_length=12,
                tile_sample_min_height=256,
                tile_sample_min_width=256,
                spatial_tile_overlap_factor=0.25,
                temporal_tile_overlap_factor=0,
                allow_spatial_tiling=True,
                parallel_group=None,
            )
            recon_pm1.append(chunk.cpu().float())
        recon_pm1 = torch.cat(recon_pm1, dim=2)
        recon_pm1 = recon_pm1[:, :, :dec._num_frames, :, :]
        chunk = rearrange(recon_pm1, "b c t h w -> (b t) c h w")
        chunk = (chunk + 1) * 127.5
        chunk = chunk.clamp(0, 255)
        chunk = chunk.type(torch.uint8)

        if args.output_folder is not None:
            writer = imageio.get_writer(osp.join(args.output_folder, basename), fps=fps,  macro_block_size=None)
            for frame in chunk:
                writer.append_data(frame.permute(1, 2, 0).cpu().numpy())
            writer.close()

        # psnr = get_psnr(video_pm1, recon_pm1, zero_mean=True, is_video=True)
        # ssim, msssim = get_ssim_and_msssim(video_pm1, recon_pm1, zero_mean=True, is_video=True)
        # lpips = get_lpips(video_pm1, recon_pm1, zero_mean=True, is_video=True)
        # print(f"PSNR={psnr.item():.4f}dB | SSIM={ssim.item():.4f} | MS-SSIM={msssim.item():.4f} | LPIPS (AlexNet)={lpips.item():.4f}")
        print(f"tiled_decode_3d | {time.time() - tic}")


if __name__ == "__main__":
    main()