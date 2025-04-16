import yaml
from typing import Any

def load_config(config_path: str='config.yaml') -> Any:
    """
    This function loads the parameters from the yaml config file
    :param config_path: The path to the .yaml config file
    :return: The parameters as a dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)