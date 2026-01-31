from autocvd import autocvd
autocvd(num_gpus = 1)

import os

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore
from simulate_config import SimulatorConfig

from odisseo.option_classes import SimulationConfig

from utils_simulate import sample_parameters

cs = ConfigStore.instance()
cs.store(name="simulator_config", node=SimulatorConfig)

@hydra.main(version_base=None, config_path="config", config_name="simulate_config",)
def main(cfg: SimulatorConfig):
    # os.makedirs(os.path.join(cfg.base_dir, cfg.data_dir), exist_ok=True)
    print(cfg.odisseo_config)

    # config = 

    # for i in range(n, n+cfg.batch_size):
    #     sim_output, params_vector = vmpa(run_simulation)(parameters_dict)
        
    #     for j in range(len()):
    #         np.savez('simulation_{}')
    samples = sample_parameters(prior_global_dict=cfg.priors_global, prior_local_dict=cfg.priors_local, n_samples=cfg.n_simulations, target_streams=cfg.target_streams)
    print(samples)


    
    
if __name__ == "__main__":
    main()