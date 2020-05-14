import os
from collections import OrderedDict
from pathlib import Path

from craft_text_detector import file_utils


def copyStateDict(state_dict):
    """
    Copies network(model) deserialized weights and biases.
    :param state_dict: Deserialized weights and biases
    :return: New deserialized weights and biases
    """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    """
    Converts string to bool.
    If input includes any of ("yes", "y", "true", "t", "1") them, returns True
    :param v: Input string
    :type v: str
    :return: bool
    """
    return v.lower() in ("yes", "y", "true", "t", "1")


def get_weight_path(model_path, model_url, net_name: str):
    """
    Downloads weights and biases if model_path is empty.
        Default download path:
            Linux: $HOME/.craft_text_detector/weights
            Windows: $HOME/.craft_text_detector/weights
    :param model_path: Serialized network(model) file
    :param model_url: network(model) url
    :param net_name: network(model) file name
    :type net_name: str
    :return: weight path
        if model_path is None:
            weight_path = "$HOME/.craft_text_detector/weights"
    """
    home_path = str(Path.home())
    weight_path = os.path.join(
        home_path, ".craft_text_detector", "weights", net_name
    )
    # check if weights are already downloaded, if not download
    if os.path.isfile(weight_path) is not True:
        print("Craft text detector weight will be downloaded to {}".format(weight_path))
        if model_path is None:
            url = model_url
            file_utils.download(url=url, save_path=weight_path)
        else:
            # TODO! give path to load craft_model
            weight_path = model_path
    return weight_path