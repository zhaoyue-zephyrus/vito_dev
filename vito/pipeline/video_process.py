# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
import tempfile

import torch
from einops import rearrange

from vito.model.vae.vae_model import AutoModel, VideoTokenizerABC
from vito.model.vae.vae_module import DiagonalGaussianDistribution


############################################
# VaeHelper
###########################################
class SingletonMeta(type):
    """
    Singleton metaclass
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class VaeHelper(metaclass=SingletonMeta):
    def __init__(self):
        # Initialize cache dict
        if not hasattr(self, "vae_cache_dict"):
            self.vae_cache_dict = {}

    @staticmethod
    def get_vae(vae_ckpt: str) -> VideoTokenizerABC:
        """
        Load a pretrained VAE model.

        Args:
            vae_ckpt (str): Path to the pretrained VAE checkpoint.

        Returns:
            VideoTokenizerABC: Pretrained VAE model.
        """
        vae_helper = VaeHelper()

        if vae_ckpt not in vae_helper.vae_cache_dict:
            vae = AutoModel.from_pretrained(vae_ckpt)
            vae.encode = vae_helper.patch_vae_encode.__get__(vae)
            vae.cuda()
            vae.eval()
            vae.bfloat16()
            if os.environ.get("OFFLOAD_VAE_CACHE") == "true":
                return vae
            vae_helper.vae_cache_dict[vae_ckpt] = vae
        return vae_helper.vae_cache_dict[vae_ckpt]

    @staticmethod
    @torch.no_grad()
    def patch_vae_encode(vae: VideoTokenizerABC, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input video.

        Args:
            x (torch.Tensor): Input video tensor with shape (N, C, T, H, W).
            sample_posterior (bool): Whether to sample from the posterior.

        Returns:
            torch.Tensor: Encoded tensor with additional information.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input x to be torch.Tensor, but got {type(x)}.")
        if len(x.shape) != 5:
            raise ValueError(f"Expected input tensor x to have shape (N, C, T, H, W), but got {x.shape}.")

        if not hasattr(vae, "encoder") or not callable(vae.encoder):
            raise AttributeError("Encoder is not defined or callable. Please initialize 'self.encoder'.")

        # for setting vae encoding to deterministic
        N, C, T, H, W = x.shape
        if T == 1:
            x = x.expand(-1, -1, 4, -1, -1)
            x = vae.encoder(x)
            posterior = DiagonalGaussianDistribution(x)
            z = posterior.mode()

            return z[:, :, :1, :, :].type(x.dtype)
        else:
            x = vae.encoder(x)
            posterior = DiagonalGaussianDistribution(x)
            z = posterior.mode()

            return z.type(x.dtype)