from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Tuple, Dict, Any, Optional


@dataclass
class DataConfig:
    dataset: str = "npy"  # "npy" for datasets/X.npy,y.npy,L.npy
    data_dir: str = "data"
    test_size: float = 0.2
    stratify: bool = True
    graph_mode: str = "fixed"  # "fixed" or "pearson_train"
    pearson_k: int = 10
    pearson_abs_corr: bool = True
    pearson_normalized_laplacian: bool = False


@dataclass
class ModelConfig:
    hidden_dims: Tuple[int, ...] = (20,)
    task: str = "binary"  # "binary" only for now
    threshold: float = 0.5


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 40
    batch_size: int = 32
    lr_mlp: float = 1e-3
    lr_theta: Optional[float] = None  # if None, use lr_mlp
    lambda_l1: float = 0.1
    gamma: float = 0.01
    prox_interval: int = 1  # apply prox every N epochs
    device: str = "cpu"
    theta_nnz_eps: float = 1e-8
    standardize_x: bool = False
    standardize_eps: float = 1e-8


@dataclass
class RunConfig:
    #  use default_factory to avoid mutable default dataclass instances
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    results_dir: str = "results"
    run_name: str = "graph_lassonet"
    save_theta: bool = True
    save_selected: bool = True
    save_history: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

"""
#做外部实验时用下面这版
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Tuple, Dict, Any, Optional


@dataclass
class DataConfig:
    dataset: str = "npy"  # "npy" for datasets/X.npy,y.npy,L.npy
    data_dir: str = "data"
    test_size: float = 0.2
    stratify: bool = True


@dataclass
class ModelConfig:
    hidden_dims: Tuple[int, ...] = (20,)
    task: str = "binary"  # "binary" only for now
    threshold: float = 0.5


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 40
    batch_size: int = 32
    lr_mlp: float = 1e-3
    lr_theta: Optional[float] = None  # if None, use lr_mlp
    lambda_l1: float = 0.1
    gamma: float = 0.01
    prox_interval: int = 1  # apply prox every N epochs
    device: str = "cpu"
    theta_nnz_eps: float = 1e-8
    standardize_x: bool = False # change to True when doing external experiments
    standardize_eps: float = 1e-8


@dataclass
class RunConfig:
    #  use default_factory to avoid mutable default dataclass instances
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    results_dir: str = "results"
    run_name: str = "graph_lassonet"
    save_theta: bool = True
    save_selected: bool = True
    save_history: bool = True
    save_model: bool = True
    save_scaler: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        """
