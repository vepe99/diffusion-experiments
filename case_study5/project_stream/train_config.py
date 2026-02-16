from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class GlobalModelConfig:
    dropout: float
    summary_dim: int
    embed_dims: int
    num_heads: int

    # inference_mlp_depth: Optional[int] = None
    # inference_mlp_width: Optional[int] = None
    # time_embedding_dim: Optional[int] = None



@dataclass
class LocalModelConfig:
    dropout: float
    summary_dim: int
    # embed_dims: Optional[int] = None
    # num_heads: Optional[int] = None

    # inference_mlp_depth: Optional[int] = None
    # inference_mlp_width: Optional[int] = None
    # time_embedding_dim: Optional[int] = None



@dataclass
class TrainConfig:
    #paths  
    base_dir: str
    data_dir: str
    N_simulations: int
    results_dir: str
    test: bool
    

    #training hyperparameters
    n_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    lr_scheduler_step_size: int
    lr_scheduler_gamma: float
    verbose: int

    #carthesian or observed space
    sim_data: str
    #inference condition
    inference_conditions: list

    #nmes of parameters
    parameters_global: list
    parameters_local: list
    paramater_global_pretty: list
    parameter_local_pretty: list
    inference_conditions: list

    #Stream augmentation
    target_streams: dict
    observational_window: dict  
    observed_n_stars: dict
    augmentations: list
    gaia_id: dict

    #models hyperparameters
    global_model: GlobalModelConfig
    local_model: LocalModelConfig

    #use causal_mask
    # use_causal_mask: bool

    hydra: Optional[Any] = field(default=None)