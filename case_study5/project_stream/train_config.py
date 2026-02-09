from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class GlobalModelConfig:
    dropout: float
    summary_dim: int
    # embed_dims: Optional[int] = None
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
    results_dir: str

    #training hyperparameters
    n_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    lr_scheduler_step_size: int
    lr_scheduler_gamma: float
    verbose: int

    #nmes of parameters
    parameters_global: list
    parameters_local: list
    paramater_global_pretty: list
    parameter_local_pretty: list
    inference_conditions: list

    #carthesian or observed space
    sim_data: str

    #add observational uncertatinty
    # loading_function: str

    #priors and scores
    # priors_global: dict  # str for the type and a list of values for the parameters of the prior
    # priors_local: dict   # str for the type and a list of values for the parameters of the prior

    #models hyperparameters
    global_model: GlobalModelConfig
    local_model: LocalModelConfig

    #use causal_mask
    # use_causal_mask: bool

    hydra: Optional[Any] = field(default=None)