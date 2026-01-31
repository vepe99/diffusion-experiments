from typing import Any, Optional
from dataclasses import dataclass, field

@dataclass
class OdisseoConfig:
    N_particles: int
    return_snapshots: bool
    num_snapshots: int
    num_timesteps: int
    external_accelerations: tuple
    acceleration_scheme: str
    softening: float
    integrator: str
    fixed_timestep: bool
    diffrax_solver: str
    glorder: int

@dataclass
class SimulatorConfig:

    n_simulations: int
    batch_size: int

    base_dir: str
    data_dir: str

    parameters_global: list
    parameters_local: list

    pretty_parameters_global: list
    pretty_parameters_local: list

    priors_global: dict #str for the type and a list of values for the parameters of the prior
    priors_local: dict  #str for the type and a list of values for the parameters of the prior

    target_streams: dict

    n_simulations: int

    batch_size: int

    odisseo_config: OdisseoConfig 

    hydra: Optional[Any] = field(default=None)

    
