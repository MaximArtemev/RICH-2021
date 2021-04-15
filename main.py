from os.path import join
import os
from core.train import train
import hydra
import logging
from omegaconf.dictconfig import DictConfig


log = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"


def prepare_config(config: DictConfig) -> None:

    assert config.experiment.particle, "Specify particle str"

    if not os.path.isabs(config.data.data_path):
        config.data.data_path = os.path.join(hydra.utils.get_original_cwd(),
                                             config.data.data_path)
        log.debug(f"config.data.data_path modified to {config.data.data_path}")

    # modifying path to checkpoints
    if not os.path.isabs(config.experiment.checkpoint_path):
        config.experiment.checkpoint_path = os.path.join(hydra.utils.get_original_cwd(),
                                                         config.experiment.checkpoint_path)
        log.debug(f"config.experiment.checkpoint_path modified to {config.experiment.checkpoint_path}")


def prepare_dirs() -> None:
    # since hydra will run script in a new subfolder it is necessary to create dirs
    os.makedirs(join(os.getcwd(), 'checkpoint'), exist_ok=True)
    log.debug("Created checkpoint dir")


# decorator allows hydra to load and parse config
# additionally, hydra will setup a global logger
@hydra.main(config_path=CONFIG_PATH)
def main(config: DictConfig) -> None:
    log.info(config.pretty())
    log.info("Current working directory  : {}".format(os.getcwd()))

    prepare_dirs()
    prepare_config(config)

    train(config)


if __name__ == '__main__':
    main()
