import os
import time
from pathlib import Path

import cv2
import numpy as np

import torch
from torch.backends import cudnn
from torch.autograd import Variable

import craft_text_detector.craft_utils as craft_utils
import craft_text_detector.imgproc as imgproc
import craft_text_detector.file_utils as file_utils
from craft_text_detector.models.craftnet import CRAFT
from craft_text_detector.models.refinenet import RefineNet

from collections import OrderedDict

# CRAFT_GDRIVE_URL = "https://drive.google.com/uc?id=1bupFXqT-VU6Jjeul13XP7yx2Sg5IHr4J"
# REFINENET_GDRIVE_URL = (
#     "https://drive.google.com/uc?id=1xcE9qpJXp4ofINwXWVhhQIh9S8Z7cuGj"
# )

# my google drive
CRAFT_GDRIVE_URL = "https://drive.google.com/open?id=1CV3Ao4zuDikOkuHHNCw0_YRVWq7YsPBQ"
REFINENET_GDRIVE_URL = (
    "https://drive.google.com/open?id=1ZDe0WRwlxLRkwofQt8C18ZTF-oA3vbfs"
)


def copyStateDict(state_dict):
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
    return v.lower() in ("yes", "y", "true", "t", "1")


def get_weight_path(craft_model_path, net_name: str):
    home_path = str(Path.home())
    weight_path = os.path.join(
        home_path, ".craft_text_detector", "weights", net_name
    )
    # check if weights are already downloaded, if not download
    if os.path.isfile(weight_path) is not True:
        print("Craft text detector weight will be downloaded to {}".format(weight_path))
        if craft_model_path is None:
            url = CRAFT_GDRIVE_URL
            file_utils.download(url=url, save_path=weight_path)
        else:
            # TODO! give path to load craft_model
            weight_path = craft_model_path
    return weight_path


def net_to_cuda(net, weight_path, cuda: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device, "!!!CUDA is not available!!!"
    if device and cuda:  # Double check
        net.load_state_dict(copyStateDict(torch.load(weight_path)))

        net = net.cuda()  # TODO! replace .cuda with .device beginning of the module.
        net = torch.nn.DataParallel(net)
        cudnn.enabled = True
        cudnn.benchmark = False  # TODO! add benchmark mode for faster operations
    else:
        net.load_state_dict(
            copyStateDict(torch.load(weight_path, map_location="cpu"))
        )
    net.eval()
    return net


def load_craftnet_model(cuda: bool = False, craft_model_path=None):
    # load craft net
    craft_net = CRAFT()  # initialize

    # get craft net path
    weight_path = get_weight_path(craft_model_path, "craft_mlt_25k.pth")

    # arange device
    return net_to_cuda(craft_net, weight_path, cuda)


def load_refinenet_model(cuda: bool = False, refinenet_model_path=None):
    # load refine net
    refine_net = RefineNet()  # initialize

    # get refine net path
    weight_path = get_weight_path(refinenet_model_path, "craft_refiner_CTW1500.pth")

    # arange device
    return net_to_cuda(refine_net, weight_path, cuda)


def get_prediction(image, craft_net, refine_net=None, text_threshold: float = 0.7, link_threshold: float = 0.4,
                   low_text: float = 0.4, long_size: int = 1280, cuda: bool = False, poly: bool = True,
                   show_time: bool = False):
    """
    Arguments:
        image: image to be processed
        output_dir: path to the results to be exported
        craft_net: craft net model
        refine_net: refine net model
        text_threshold: text confidence threshold
        link_threshold: link confidence threshold
        low_text: text low-bound score
        cuda: Use cuda for inference
        canvas_size: image size for inference
        long_size: desired longest image size for inference
        poly: enable polygon type
        show_time: show processing time
    Output:
        {"masks": lists of predicted masks 2d as bool array,
         "boxes": list of coords of points of predicted boxes,
         "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
         "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
         "heatmaps": visualizations of the detected characters/links,
         "times": elapsed times of the sub modules, in seconds}
    """
    t0 = time.time()

    # resize
    # TODO! interpolation=cv2.INTER_CUBIC
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, long_size, interpolation=cv2.INTER_CUBIC
    )
    ratio_h = ratio_w = 1 / target_ratio
    resize_time = time.time() - t0
    t0 = time.time()

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()
        if not craft_net is None:
            craft_net = craft_net.cuda()
        if not refine_net is None:
            refine_net = refine_net.cuda()
    preprocessing_time = time.time() - t0
    t0 = time.time()

    # forward pass
    with torch.no_grad():
        y, feature = craft_net(x)
    craftnet_time = time.time() - t0
    t0 = time.time()

    # make score and link map
    # TODO! test .detach()
    score_text = y[0, :, :, 0].detach().cpu().data.numpy()
    score_link = y[0, :, :, 1].detach().cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].detach().cpu().data.numpy()
    refinenet_time = time.time() - t0
    t0 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    # get image size
    img_height = image.shape[0]
    img_width = image.shape[1]

    # calculate box coords as ratios to image size
    boxes_as_ratio = []
    for box in boxes:
        boxes_as_ratio.append(box / [img_width, img_height])
    boxes_as_ratio = np.array(boxes_as_ratio)

    # calculate poly coords as ratios to image size
    polys_as_ratio = []
    for poly in polys:
        polys_as_ratio.append(poly / [img_width, img_height])
    polys_as_ratio = np.array(polys_as_ratio)

    text_score_heatmap = imgproc.cvt2HeatmapImg(score_text)
    link_score_heatmap = imgproc.cvt2HeatmapImg(score_link)

    postprocess_time = time.time() - t0

    times = {
        "resize_time": resize_time,
        "preprocessing_time": preprocessing_time,
        "craftnet_time": craftnet_time,
        "refinenet_time": refinenet_time,
        "postprocess_time": postprocess_time,
    }

    if show_time:
        print(
            "\ninfer/postproc time : {:.3f}/{:.3f}".format(
                refinenet_time + refinenet_time, postprocess_time
            )
        )

    return {
        "boxes": boxes,
        "boxes_as_ratios": boxes_as_ratio,
        "polys": polys,
        "polys_as_ratios": polys_as_ratio,
        "heatmaps": {
            "text_score_heatmap": text_score_heatmap,
            "link_score_heatmap": link_score_heatmap,
        },
        "times": times,
    }


class predict:
    def __init__(self):
        CRAFT_GDRIVE_URL = "https://drive.google.com/uc?id=1bupFXqT-VU6Jjeul13XP7yx2Sg5IHr4J"
        REFINENET_GDRIVE_URL = (
            "https://drive.google.com/uc?id=1xcE9qpJXp4ofINwXWVhhQIh9S8Z7cuGj"
        )

    def __call__(self, *args, **kwargs):
        pass

    def get_weight_path(self, craft_model_path, net_name: str):
        home_path = str(Path.home())
        weight_path = os.path.join(
            home_path, ".craft_text_detector", "weights", net_name
        )
        # check if weights are already downloaded, if not download
        if os.path.isfile(weight_path) is not True:
            print("Craft text detector weight will be downloaded to {}".format(weight_path))
            if craft_model_path is None:
                url = CRAFT_GDRIVE_URL
                file_utils.download(url=url, save_path=weight_path)
            else:
                # TODO! give path to load craft_model
                weight_path = craft_model_path
        return weight_path

    def to_cuda(self, net, weight_path, cuda: bool = False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert device, "!!!CUDA is not available!!!"
        if device and cuda:  # Double check
            net.load_state_dict(copyStateDict(torch.load(weight_path)))

            net = net.cuda()  # TODO! replace .cuda with .device beginning of the module.
            net = torch.nn.DataParallel(net)
            cudnn.enabled = True
            cudnn.benchmark = False  # TODO! add benchmark mode for faster operations
        else:
            net.load_state_dict(
                copyStateDict(torch.load(weight_path, map_location="cpu"))
            )
        net.eval()
        return net

    def load_craftnet_model(self, cuda: bool = False, craft_model_path=None):
        # load craft net
        craft_net = CRAFT()  # initialize

        # get craft net path
        weight_path = get_weight_path(craft_model_path, "craft_mlt_25k.pth")

        # arange device
        return net_to_cuda(craft_net, weight_path, cuda)

    def load_refinenet_model(self, cuda: bool = False, refinenet_model_path=None):
        # load refine net
        refine_net = RefineNet()  # initialize

        # get refine net path
        weight_path = get_weight_path(refinenet_model_path, "craft_refiner_CTW1500.pth")

        # arange device
        return net_to_cuda(refine_net, weight_path, cuda)


if __name__ == "__main__":
    import craft_text_detector as craft

    # set image path and export folder directory
    image_name = 'a8.png'
    image_path = '../figures/IAM8/'+image_name
    output_dir = 'outputs/'

    # read image
    image = craft.read_image(image_path)

    # load models
    craft_net = craft.load_craftnet_model()
    refine_net = craft.load_refinenet_model()

    # perform prediction
    # TODO!  cuda=True -->> error
    prediction_result = craft.get_prediction(image=image, craft_net=craft_net, refine_net=refine_net,
                                             text_threshold=0.7, link_threshold=0.4, low_text=0.4, long_size=1280,
                                             cuda=True, show_time=True)

    # export detected text regions
    exported_file_paths = craft.export_detected_regions(
        image_path=image_path,
        image=image,
        regions=prediction_result["boxes"],
        output_dir=output_dir,
        rectify=True
    )

    # export heatmap, detection points, box visualization
    craft.export_extra_results(
        image_path=image_path,
        image=image,
        regions=prediction_result["boxes"],
        heatmaps=prediction_result["heatmaps"],
        output_dir=output_dir
    )
