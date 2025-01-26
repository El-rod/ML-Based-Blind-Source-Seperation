from dataclasses import MISSING, asdict, dataclass, field
from datetime import datetime
from typing import Optional

from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver(
    "datetime", lambda s: f'{s}_{datetime.now().strftime("%H_%M_%S")}')


@dataclass
class ModelConfig:
    """
    WaveNet pytorch model parameters configuration
    """
    input_channels: int = 2
    residual_layers: int = 30
    residual_channels: int = 64
    dilation_cycle_length: int = 10


@dataclass
class DataConfig:
    """
    WaveNet pytorch dataset parameters configuration
    """
    root_dir: str = MISSING
    batch_size: int = 16
    num_workers: int = 4
    train_fraction: float = 0.8


@dataclass
class DistributedConfig:
    """
    WaveNet pytorch process distributed parameters configuration
    """
    distributed: bool = False
    # Number of processes participating in distributed training
    world_size: int = 2


@dataclass
class TrainerConfig:
    """
    WaveNet pytorch training parameters configuration
    """
    learning_rate: float = 2e-4
    max_steps: int = 1000
    max_grad_norm: Optional[float] = None
    fp16: bool = False

    log_every: int = 50
    save_every: int = 2000
    validate_every: int = 100


@dataclass
class Config:
    """
    dataclass that stores all types of configuration dataclasses
    (nested structured configs)
    """
    model_dir: str = MISSING

    # model: ModelConfig = ModelConfig()
    # data: DataConfig = DataConfig(root_dir="")
    # distributed: DistributedConfig = DistributedConfig()
    # trainer: TrainerConfig = TrainerConfig()

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=lambda: DataConfig(root_dir=""))
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


def parse_configs(cfg: DictConfig, cli_cfg: Optional[DictConfig] = None) -> DictConfig:
    """
    cfg: Config Dictionary
    cli_cfg: Config from CLI (command line interface)

    returns ConfigDict of inserted ConfigDict merged with
    content of sys.arg (command line arguments)
    """
    # Structured configs are used to create OmegaConf configuration object with runtime type safety.
    base_cfg = OmegaConf.structured(Config)
    # Merging configurations enables the creation of reusable configuration files
    # for each logical component instead of a single config file for each variation of your task.
    merged_cfg = OmegaConf.merge(base_cfg, cfg)
    if cli_cfg is not None:
        merged_cfg = OmegaConf.merge(merged_cfg, cli_cfg)
    return merged_cfg


if __name__ == "__main__":
    base_config = OmegaConf.structured(Config)
    config = OmegaConf.load("configs/short_ofdm.yaml")
    config = OmegaConf.merge(base_config, OmegaConf.from_cli(), config)
    config = Config(**config)

    print(asdict(config))
