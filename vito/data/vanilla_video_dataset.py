import argparse
import os
import numpy as np
import random
import torch
import torch.distributed as dist
from torchvision.transforms import v2, InterpolationMode
from torchcodec.decoders import set_cuda_backend, VideoDecoder

from vito.common.logger import vito_logger


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_folder,
        data_list,
        train=True,
        resolution=256,
        use_resize=True,
        use_crop=True,
        use_flip=False,
        sample_clip=True,
        num_clips=1,
        frames_per_clip=24,
        stride=1,
        use_gpu_decode=False,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.data_list = data_list
        self.train = train
        self.resolution = resolution

        self.sample_clip = sample_clip
        self.num_clips = num_clips
        self.frames_per_clip = frames_per_clip
        self.stride = stride

        self.use_gpu_decode = use_gpu_decode

        with open(self.data_list) as f:
            self.annotations = f.readlines()

        # Build transforms list conditionally to avoid unpicklable lambdas
        transforms = []

        if use_resize:
            transforms.append(v2.Resize(int(resolution * 1.125), interpolation=InterpolationMode.BILINEAR))

        if train:
            transforms.append(v2.RandomCrop(resolution))
        elif use_crop:
            transforms.append(v2.CenterCrop(resolution))

        if train and use_flip:
            transforms.append(v2.RandomHorizontalFlip())

        transforms.extend([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.transforms = v2.Compose(transforms)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_path = self.annotations[idx].strip()
        if self.data_folder is not None:
            video_path = os.path.join(self.data_folder, video_path)

        try:
            if self.use_gpu_decode:
                with set_cuda_backend("beta"):
                    dec = VideoDecoder(video_path, device="cuda")
            else:
                dec = VideoDecoder(video_path)
        except Exception as e:
            vito_logger.error(f"Error decoding video {video_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        total_frames = dec._num_frames
        if self.train:
            assert self.sample_clip, "In training, we must sample clips."
            max_start_frame = total_frames - self.num_clips * self.frames_per_clip * self.stride
            if max_start_frame < 0:
                vito_logger.warning(f"Video {video_path} is too short ({total_frames} frames). Using start_frame=0.")
                start_frame = 0
            else:
                start_frame = random.randint(0, max_start_frame)
            frame_indices = [
                start_frame + clip_id * self.frames_per_clip * self.stride + frame_id * self.stride
                for frame_id in range(self.frames_per_clip) for clip_id in range(self.num_clips)
            ]
            if max(frame_indices) >= total_frames:
                vito_logger.warning(f"Frame index out of range for video {video_path}. Clipping indices.")
                frame_indices = [min(fi, total_frames - 1) for fi in frame_indices]
            batch = dec.get_frames_at(frame_indices)
        else:
            if self.sample_clip:
                max_start_frame = total_frames - self.num_clips * self.frames_per_clip * self.stride
                if max_start_frame < 0:
                    vito_logger.warning(f"Video {video_path} is too short ({total_frames} frames). Using start_frame=0.")
                    start_frame = 0
                else:
                    start_frame = max_start_frame // 2
                frame_indices = [
                    start_frame + clip_id * self.frames_per_clip * self.stride + frame_id * self.stride
                    for frame_id in range(self.frames_per_clip) for clip_id in range(self.num_clips)
                ]
                if max(frame_indices) >= total_frames:
                    vito_logger.warning(f"Frame index out of range for video {video_path}. Clipping indices.")
                    frame_indices = [min(fi, total_frames - 1) for fi in frame_indices]
                batch = dec.get_frames_at(frame_indices)
            else:
                frame_indices = list(range(0, total_frames, self.stride))
                batch = dec.get_frames_at(frame_indices)
        video = batch.data  # TCHW, uint8
        video = self.transforms(video)  # TCHW, float32, [-1, 1]

        return {"video": video}


class VideoData():
    def __init__(self, args, shuffle=True):
        self.args = args
        self.shuffle = shuffle

    def _dataset(self, train):
        dataset = VideoDataset(
            data_folder=self.args.data_folder,
            data_list=self.args.train_list if train else self.args.val_list,
            train=train,
            resolution=self.args.resolution,
            use_resize=self.args.use_resize,
            use_crop=self.args.use_crop,
            use_flip=self.args.use_flip,
            sample_clip=self.args.sample_clip,
            num_clips=self.args.num_clips,
            frames_per_clip=self.args.frames_per_clip,
            stride=self.args.stride,
            use_gpu_decode=self.args.use_gpu_decode,
        )

        return dataset

    def _dataloader(self, train):
        dataset = self._dataset(train)
        if dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
            )
            global_rank = dist.get_rank()
        else:
            sampler = None
            global_rank = None

        def seed_worker(worker_id):
            if global_rank:
                seed = self.args.num_workers * global_rank + worker_id
            else:
                seed = worker_id
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=self.shuffle if train else False,
            num_workers=self.args.num_workers,
            pin_memory=True if not self.args.use_gpu_decode else False,
            drop_last=train,
            multiprocessing_context='spawn' if self.args.num_workers > 0 else None,
            # persistent_workers=True if self.args.num_workers > 0 else False,
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(train=True)

    def val_dataloader(self):
        return self._dataloader(train=False)


    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_folder", type=str, required=True, help="Path to the folder containing videos")
        parser.add_argument("--train_list", type=str, required=True, help="Path to the training data list file")
        parser.add_argument("--val_list", type=str, required=True, help="Path to the validation data list file")
        parser.add_argument("--resolution", type=int, default=256, help="Spatial resolution of the video frames")
        parser.add_argument("--use_resize", action="store_true", help="Whether to resize the frames before cropping")
        parser.add_argument("--use_crop", action="store_true", help="Whether to crop the frames")
        parser.add_argument("--use_flip", action="store_true", help="Whether to apply random horizontal flip")
        parser.add_argument("--sample_clip", action="store_true", help="Whether to sample clips from the videos")
        parser.add_argument("--num_clips", type=int, default=1, help="Number of clips to sample from each video")
        parser.add_argument("--frames_per_clip", type=int, default=24, help="Number of frames per clip")
        parser.add_argument("--stride", type=int, default=1, help="Temporal stride between frames")
        parser.add_argument("--batch_size", type=int, default=4, help="Batch size for dataloader")
        parser.add_argument("--use_gpu_decode", action="store_true", help="Use GPU for video decoding")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")

        return parser