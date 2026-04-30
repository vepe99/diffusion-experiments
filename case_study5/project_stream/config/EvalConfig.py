from typing import Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class GlobalModelConfig:
    dropout: float
    summary_dim: int
    embed_dims: Optional[int] 
    mlp_depths: Optional[int] 
    mlp_widths: Optional[int] 
    inference_time_embedding_dim: Optional[int]
    num_heads: int
    inference_mlp_depth: Optional[int] 
    inference_mlp_width: Optional[int] 
    time_embedding_dim: Optional[int] 



@dataclass
class LocalModelConfig:
    dropout: float
    summary_dim: int
    embed_dims: Optional[int] 
    num_heads: int
    inference_mlp_depth: Optional[int] 
    inference_mlp_width: Optional[int] 
    time_embedding_dim: Optional[int] 



@dataclass
class EvalConfig:
    #paths  
    base_dir: str
    data_dir: str
    model_dir: str
    results_dir: str
    multistream_n_simulation: int
    use_streamax_simulator: bool #this will generate q from dirx, diry, dirz

    #evalutation hyperparameters
    n_samples: int
    method: str
    steps: str
    max_steps: int
    mini_batch_size: Optional[int]
    batch_size: int
    inverse_compositional_bridge_d1: Optional[Union[int, float]]
    noise_schedule: Optional[dict]
    
    verbose: int

    #nmes of parameters
    parameters_global: list
    parameters_local: list
    paramater_global_pretty: list
    parameter_local_pretty: list
    inference_conditions: list

    #carthesian or observed space
    sim_data: str
    
    #Stream augmentation
    target_streams: dict
    observational_window: dict  
    observed_n_stars: dict
    min_star_with_vlos: Optional[dict]
    error_keys: Optional[list]
    masked_value_vlos: Optional[dict]
    augmentations: list
    gaia_id: dict



    #models hyperparameters
    global_model: GlobalModelConfig
    local_model: LocalModelConfig

    #use causal_mask
    # use_causal_mask: bool

    hydra: Optional[Any] = field(default=None)