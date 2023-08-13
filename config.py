from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field


@dataclass
class Report:
    with_tracking: bool = False
    name: str = 'wandb'
    wandb_id: str = 'your_wandb_id'

@dataclass
class Path:
    dataset_path: str = 'your_dataset_path'
    output_dir: str = './outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'

@dataclass
class Model:
    name: str = 'skt/kogpt2-base-v2'
    hidden_size: int = 768

@dataclass
class Data:
    max_seq_len: int = 128

@dataclass
class Param:
    seed: int = 42
    epochs: int = 1
    lr: float = 5e-5
    batch_size: int = 8

@dataclass
class KoGPT2Config:
    report: Report = field(default_factory=Report)
    path: Path = field(default_factory=Path)
    model: Model = field(default_factory=Model)
    data: Data = field(default_factory=Data)
    param: Param = field(default_factory=Param)


cs = ConfigStore.instance()
cs.store(name='config', node=KoGPT2Config)

