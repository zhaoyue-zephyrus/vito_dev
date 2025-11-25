import dataclasses
import json
from typing import Literal

import torch


@dataclasses.dataclass
class OptimConfig:
    lr: float = 1e-4
    max_steps: int = 500
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0
    max_grad_norm_disc: float = 1.0

    lpips_weight: float = 0.1
    kl_weight: float = 0.1
    disc_weight: float = 0.1

@dataclasses.dataclass
class VAEConfig:
    ddconfig: dict
    model_type: Literal["vit", "vit_ncthw"]


@dataclasses.dataclass
class DiscriminatorConfig:
    disc_layers: int = 3
    base_ch_disc: int = 64
    disc_pool: bool = False
    disc_pool_size: int = 1000   # Size of the image buffer if using image pool
    disc_warmup_steps: int = 0
    disc_lr_multiplier: float = 1.0
    discriminator_iter_start: int = 0
    disc_pretrain_iter: int = 0
    disc_optim_steps: int = 1
    disc_loss_type: Literal["hinge", "vanilla"] = "hinge"
    disc_gan_weight: float = 0.3
    disc_disc_weight: float = 1.0

@dataclasses.dataclass
class EngineConfig:
    # Parallism strategy
    distributed_backend: str = "nccl"  # Choices: ["nccl", "gloo"]
    distributed_timeout_minutes: int = 10  # Timeout minutes for torch.distributed.
    dp_size: int = 1  # Degree of data parallelism.
    pp_size: int = 1  # Degree of pipeline model parallelism.
    cp_size: int = 1  # Degree of context parallelism.
    cp_strategy: str = "none"  # Choices: ["none", "cp_ulysses", "cp_shuffle_overlap"]
    ulysses_overlap_degree: int = 1  # Overlap degree for Ulysses


@dataclasses.dataclass
class VitoConfig:
    optim_config: OptimConfig
    vae_config: VAEConfig
    disc_config: DiscriminatorConfig
    engine_config: EngineConfig

    @classmethod
    def _check_missing_fields(cls, config_dict: dict, required_fields: list):
        actual_fields = set(config_dict.keys())
        missing_fields = set(required_fields) - actual_fields
        if missing_fields:
            raise ValueError(f"Missing fields in the configuration file: {', '.join(missing_fields)}")

    @classmethod
    def _create_nested_config(cls, config_dict: dict, config_name: str, config_cls):
        nested_config_dict = config_dict.get(config_name, {})
        cls._check_missing_fields(nested_config_dict, config_cls.__dataclass_fields__.keys())
        return config_cls(**nested_config_dict)

    @classmethod
    def _create_config_from_dict(cls, config_dict: dict):
        cls._check_missing_fields(config_dict, cls.__dataclass_fields__.keys())

        # Create nested configs
        optim_config = cls._create_nested_config(config_dict, "optim_config", OptimConfig)
        vae_config = cls._create_nested_config(config_dict, "vae_config", VAEConfig)
        disc_config = cls._create_nested_config(config_dict, "disc_config", DiscriminatorConfig)
        engine_config = cls._create_nested_config(config_dict, "engine_config", EngineConfig)

        return cls(
            optim_config=optim_config,
            vae_config=vae_config,
            disc_config=disc_config,
            engine_config=engine_config
        )

    @classmethod
    def from_json(cls, json_path: str):
        def simple_json_decoder(dct):
            dtype_map = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16, "torch.float32": torch.float32}
            if 'params_dtype' in dct:
                dct['params_dtype'] = dtype_map[dct['params_dtype']]
            return dct

        with open(json_path, "r") as f:
            config_dict = json.load(f, object_hook=simple_json_decoder)
        vito_config = cls._create_config_from_dict(config_dict)

        return vito_config