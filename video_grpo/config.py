import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import yaml


@dataclass
class FSDPConfig:
    auto_wrap_policy: str = "transformer_based_wrap"
    backward_prefetch: str = "backward_pre"
    forward_prefetch: bool = True
    cpu_ram_efficient_loading: bool = False
    cpu_offload: bool = False
    sharding_strategy: str = "full_shard"
    state_dict_type: str = "sharded_state_dict"
    sync_module_states: bool = False
    use_orig_params: bool = True
    activation_checkpointing: bool = True


@dataclass
class AccelerateConfig:
    distributed_type: str = "FSDP"
    mixed_precision: str = "bf16"
    num_processes: int = 1
    num_machines: int = 1
    machine_rank: int = 0
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)


@dataclass
class TrainConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int | None = (
        None  # if None, derive from sample settings
    )
    num_inner_epochs: int = 1
    timestep_fraction: float = 0.99
    beta: float = 0.0
    learning_rate: float = 1e-4
    clip_range: float = 1e-3
    adv_clip_max: float = 5.0
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    use_8bit_adam: bool = False
    ema: bool = True
    ema_decay: float = 0.9
    ema_update_interval: int = 8
    cfg: bool = True
    full_finetune: bool = False
    lora_path: Optional[str] = None
    loss_reweighting: Optional[str] = None
    weight_advantages: bool = False  # If True, weight advantages after computing them; if False, weight rewards before computing advantages
    # LoRA hyperparams
    lora_r: int = 32
    lora_alpha: int = 64
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "add_k_proj",
            "add_q_proj",
            "add_v_proj",
            "to_add_out",
            "to_k",
            "to_out.0",
            "to_q",
            "to_v",
        ]
    )


@dataclass
class SampleConfig:
    batch_size: int = 8
    eval_batch_size: int = 2
    num_batches_per_epoch: int = 2
    num_steps: int = 20
    eval_num_steps: int = 50
    guidance_scale: float = 4.5
    eval_guidance_scale: Optional[float] = None
    num_video_per_prompt: int = 4
    kl_reward: float = 0.0
    global_std: bool = False
    max_group_std: bool = False
    same_latent: bool = False
    noise_level: Optional[float] = 0.7
    sde_window_size: Optional[int] = None
    sde_window_range: Optional[Any] = None
    sde_type: Optional[str] = "flow_sde"  # 'flow_sde' or 'flow_cps'
    diffusion_clip: bool = False
    diffusion_clip_value: float = 0.45


@dataclass
class ProjectPaths:
    save_dir: str = "logs/checkpoints"
    dataset: str = "dataset/ocr"
    pretrained_model: str = ""
    resume_from: Optional[str] = None
    accelerate_config: Optional[str] = None


@dataclass
class Config:
    run_name: str = "run"
    project_name: str = "VideoGRPO"
    seed: int = 42
    num_epochs: int = 1
    eval_freq: int = 30
    initial_eval: bool = False
    save_freq: int = 60
    num_checkpoint_limit: int = 1
    height: int = 240
    width: int = 416
    frames: int = 33
    reward_fn: Dict[str, float] = field(default_factory=lambda: {"video_ocr": 1.0})
    eval_reward_fn: Optional[Dict[str, float]] = None
    reward_module: Optional[str] = None
    eval_reward_module: Optional[str] = None
    prompt_fn: str = "general_ocr"
    trainer: str = "wan"
    use_lora: bool = True
    allow_tf32: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    per_prompt_stat_tracking: bool = True
    resume_from: Optional[str] = None
    sample: SampleConfig = field(default_factory=SampleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    accelerate: AccelerateConfig = field(default_factory=AccelerateConfig)


def load_config(path: str) -> Config:
    """Load a YAML/JSON config file into the strongly typed Config dataclass.

    Args:
        path: Path to a `.yaml/.yml` or `.json` config file.

    Returns:
        Parsed `Config` object with expanded user paths.
    """
    with open(path, "r") as f:
        if path.endswith((".yaml", ".yml")):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    def build_dataclass(cls, src: Dict[str, Any]):
        """Recursively construct nested dataclasses from plain dicts."""
        kwargs = {}
        for field_name, field_def in cls.__dataclass_fields__.items():  # type: ignore
            if field_name in src:
                val = src[field_name]
                # dispatch based on nested dataclass types
                if (
                    isinstance(field_def.default, FSDPConfig)
                    or field_def.type == FSDPConfig
                ):
                    kwargs[field_name] = build_dataclass(FSDPConfig, val)
                elif (
                    isinstance(field_def.default, AccelerateConfig)
                    or field_def.type == AccelerateConfig
                ):
                    kwargs[field_name] = build_dataclass(AccelerateConfig, val)
                elif (
                    isinstance(field_def.default, TrainConfig)
                    or field_def.type == TrainConfig
                ):
                    kwargs[field_name] = build_dataclass(TrainConfig, val)
                elif (
                    isinstance(field_def.default, SampleConfig)
                    or field_def.type == SampleConfig
                ):
                    kwargs[field_name] = build_dataclass(SampleConfig, val)
                elif (
                    isinstance(field_def.default, ProjectPaths)
                    or field_def.type == ProjectPaths
                ):
                    kwargs[field_name] = build_dataclass(ProjectPaths, val)
                else:
                    kwargs[field_name] = val
        return cls(**kwargs)

    cfg = build_dataclass(Config, data)

    # expand paths
    cfg.paths.save_dir = os.path.expanduser(cfg.paths.save_dir)
    cfg.paths.dataset = os.path.expanduser(cfg.paths.dataset)
    if cfg.paths.pretrained_model:
        cfg.paths.pretrained_model = os.path.expanduser(cfg.paths.pretrained_model)
    if cfg.paths.accelerate_config:
        cfg.paths.accelerate_config = os.path.expanduser(cfg.paths.accelerate_config)
    return cfg
