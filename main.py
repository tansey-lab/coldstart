import hydra
from omegaconf import DictConfig, OmegaConf
from pipeline.main import run

@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg: DictConfig):    
    print("\nFinal merged config:\n")
    print(OmegaConf.to_yaml(cfg))
    run(cfg)

if __name__ == "__main__":
    main()
