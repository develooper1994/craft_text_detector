import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn

from craft_text_detector import (
    craft_utils,
    read_image,
    export_detected_regions,
    export_extra_results,
)
from craft_text_detector import imgproc
from craft_text_detector.models.craftnet import CRAFT
from craft_text_detector.models.refinenet import RefineNet
# my google drive
from craft_text_detector.craft_detector_util import copyStateDict, get_weight_path

# Original
# CRAFT_GDRIVE_URL = "https://drive.google.com/uc?id=1bupFXqT-VU6Jjeul13XP7yx2Sg5IHr4J"
# REFINENET_GDRIVE_URL = (
#     "https://drive.google.com/uc?id=1xcE9qpJXp4ofINwXWVhhQIh9S8Z7cuGj"
# )

CRAFT_GDRIVE_URL = "https://drive.google.com/open?id=1CV3Ao4zuDikOkuHHNCw0_YRVWq7YsPBQ"
REFINENET_GDRIVE_URL = (
    "https://drive.google.com/open?id=1ZDe0WRwlxLRkwofQt8C18ZTF-oA3vbfs"
)


# !!! Legacy functional Way. DON'T TOUCH IT!!!
# TODO! Provide better legacy-functional interface
def net_to_cuda(net, weight_path, cuda: bool = False):
    is_device = torch.cuda.is_available()
    # device = torch.device('cuda' if is_device else 'cpu')
    assert is_device, "!!!CUDA is not available!!!"
    if is_device and cuda:  # Double check
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
    weight_path = get_weight_path(craft_model_path,
                                  CRAFT_GDRIVE_URL,
                                  "craft_mlt_25k.pth")

    # arange device
    return net_to_cuda(craft_net, weight_path, cuda)
    # prediction_object.cuda = cuda
    # return prediction_object.load_craftnet_model(craft_model_path)


def load_refinenet_model(cuda: bool = False, refinenet_model_path=None):
    # load refine net
    refine_net = RefineNet()  # initialize

    # get refine net path
    weight_path = get_weight_path(refinenet_model_path,
                                  REFINENET_GDRIVE_URL,
                                  "craft_refiner_CTW1500.pth")

    # arange device
    return net_to_cuda(refine_net, weight_path, cuda)
    # prediction_object.cuda = cuda
    # return prediction_object.load_refinenet_model(refinenet_model_path)


def get_prediction(image, craft_net,
                   refine_net=None,
                   text_threshold: float = 0.7,
                   link_threshold: float = 0.4,
                   low_text: float = 0.4,
                   target_size: int = 1280,
                   cuda: bool = False,
                   poly: bool = True,
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
        target_size: desired longest image size for inference
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
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, target_size, interpolation=cv2.INTER_CUBIC  # old: cv2.INTER_LINEAR
    )
    ratio_h = ratio_w = 1 / target_ratio
    resize_time = time.time() - t0
    t0 = time.time()

    # preprocessing
    x = imgproc.normalize_mean_variance(img_resized)
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
    boxes, polys = craft_utils.get_detection_boxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # coordinate adjustment
    boxes = craft_utils.adjust_result_coordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjust_result_coordinates(polys, ratio_w, ratio_h)
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
    # prediction_object.craft_net = craft_net
    # prediction_object.refine_net = refine_net
    # prediction_object.cuda = cuda
    # return prediction_object.get_prediction(image=image,
    #                                         text_threshold=text_threshold,
    #                                         link_threshold=link_threshold,
    #                                         low_text=low_text,
    #                                         target_size=target_size,
    #                                         poly=poly,
    #                                         show_time=show_time)


# !!! New oops way.
class craft_detector:
    """
    Craft(Character Region Awareness for Text Detection) implementation
    """
    def __init__(self, image=None,
                 refiner=True,
                 craft_model_path=None,
                 refinenet_model_path=None,
                 cuda: bool = False,
                 benchmark: bool = False):
        """
        Configures class initializer.
        :param image: Input image
        :param refiner: refiner switch
        :param craft_model_path: craft network(model) path with name
        :param refinenet_model_path: refiner network(model) path with name
        :param cuda: cuda switch
        :param benchmark: cudnn benchmark mode switch
        :return: None
        """
        self.reload(image=image,
                    refiner=refiner,
                    craft_model_path=craft_model_path,
                    refinenet_model_path=refinenet_model_path,
                    cuda=cuda,
                    benchmark=benchmark)

    def __call__(self, *args, **kwargs):
        return self.get_prediction(**kwargs)

    def reload(self, image=None,
               refiner=True,
               craft_model_path=None,
               refinenet_model_path=None,
               cuda: bool = False,
               benchmark: bool = False):
        """
        Configures class initializer. Sometimes the class needs to be reloaded (configured) with new parameters
        :param image: Input image
        :param refiner: refiner switch
        :param craft_model_path: craft network(model) path with name
        :param refinenet_model_path: refiner network(model) path with name
        :param cuda: cuda switch
        :param benchmark: cudnn benchmark mode switch
        :return: None
        """
        # load craft input image
        self.__set_image(image)
        # load craft net
        self.craft_net = CRAFT()  # initialize
        # load refine net
        self.refine_net = RefineNet()  # initialize
        # Double check device
        self.__set_device(cuda, benchmark)
        # Original
        # CRAFT_GDRIVE_URL = "https://drive.google.com/uc?id=1bupFXqT-VU6Jjeul13XP7yx2Sg5IHr4J"
        # REFINENET_GDRIVE_URL = (
        #     "https://drive.google.com/uc?id=1xcE9qpJXp4ofINwXWVhhQIh9S8Z7cuGj"
        # )
        # my google drive
        self.CRAFT_GDRIVE_URL = "https://drive.google.com/uc?id=1bupFXqT-VU6Jjeul13XP7yx2Sg5IHr4J"
        self.craft_model_name = "craft_mlt_25k.pth"
        self.craft_model_path = craft_model_path
        self.craft_net = self.__load_craftnet_model(self.craft_model_path)
        self.REFINENET_GDRIVE_URL = (
            "https://drive.google.com/uc?id=1xcE9qpJXp4ofINwXWVhhQIh9S8Z7cuGj"
        )
        self.refinenet_model_name = "craft_refiner_CTW1500.pth"
        self.refinenet_model_path = refinenet_model_path
        self.__set_refiner(refiner)

    def __set_image(self, image):
        """
        Configure input image
        :param image: input image
        :return: None
        """
        self.image = image
        if isinstance(image, str):
            # consider image is image
            self.image = read_image(image)

    def __set_device(self, cuda: bool = False, benchmark: bool = False):
        """
        Detects device(CPU/GPU) and configures device that network(model) running on.
        :param cuda: cuda configuration swtich.
            Default: False
        :param benchmark: cudnn benchmark mode switch.
            Default: False
        :return: None
        """
        self.cuda = cuda
        self.benchmark = benchmark
        self.is_device = torch.cuda.is_available()
        # self.device = torch.device('cuda' if self.is_device and self.cuda else 'cpu')
        if self.cuda and self.is_device:
            assert self.is_device, "!!!CUDA is not available!!!"  # Double check ;)
            self.device = torch.device('cuda')
            cudnn.enabled = True
            cudnn.benchmark = self.benchmark
        else:
            self.device = torch.device('cpu')

    def __set_refiner(self, refiner: bool = True):
        """
        Configure refiner mode.
        True -> load refine_net
        False -> don't load refine_net
        :param refiner: mode swtich
        :type refiner: bool
        :return: None
        """
        if refiner:
            self.refine_net = self.__load_refinenet_model(self.refinenet_model_path)
        else:
            self.refine_net = None

    def get_prediction(self, image=None,
                       text_threshold: float = 0.7,
                       link_threshold: float = 0.4,
                       low_text: float = 0.4,
                       target_size: int = 1280,
                       poly: bool = True,
                       show_time: bool = False):
        """
        Predicts bounding boxes where the text. The main function that gives bounding boxes.
        :param image: image to be processed
        :param text_threshold: text confidence threshold
        :param link_threshold: link confidence threshold
        :param low_text: text low-bound score
        :param target_size: desired longest image size for inference
        :param poly: enable polygon type
        :param show_time: show processing time
        :return:
            {
            "masks": lists of predicted masks 2d as bool array,
            "boxes": list of coords of points of predicted boxes,
            "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
            "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
            "heatmaps": visualizations of the detected characters/links,
            "times": elapsed times of the sub modules, in seconds
            }
        :returns:
            {
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
        """
        if image is None:
            image = self.image
        t0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
            image, target_size, interpolation=cv2.INTER_CUBIC  # old: cv2.INTER_LINEAR
        )
        ratio_h = ratio_w = 1 / target_ratio
        resize_time = time.time() - t0
        t0 = time.time()

        # preprocessing
        x = imgproc.normalize_mean_variance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = x.unsqueeze(0)  # [c, h, w] to [b, c, h, w]
        x = x.to(self.device)
        preprocessing_time = time.time() - t0
        t0 = time.time()

        # forward pass
        with torch.no_grad():
            y, feature = self.craft_net(x)
        craftnet_time = time.time() - t0
        t0 = time.time()

        # make score and link map
        score_text = y[0, :, :, 0].detach().cpu().data.numpy()
        score_link = y[0, :, :, 1].detach().cpu().data.numpy()

        # refine link
        if self.refine_net is not None:
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].detach().cpu().data.numpy()
        refinenet_time = time.time() - t0
        t0 = time.time()

        # Post-processing
        boxes, polys = craft_utils.get_detection_boxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly
        )

        # coordinate adjustment
        boxes = craft_utils.adjust_result_coordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjust_result_coordinates(polys, ratio_w, ratio_h)
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

    # detect texts
    def detect_text(self, image=None,
                    output_dir=None,
                    rectify=True,
                    export_extra=True,
                    text_threshold=0.7,
                    link_threshold=0.4,
                    low_text=0.4,
                    long_size=1280,
                    show_time=False,
                    crop_type="poly"):
        """
        Detects text but has some extra functionalities.
        :param image: path to the image to be processed
        :param output_dir: path to the results to be exported
        :param rectify: rectify detected polygon by affine transform
        :param export_extra: export heatmap, detection points, box visualization
        :param text_threshold: text confidence threshold
        :param link_threshold: link confidence threshold
        :param low_text: text low-bound score
        :param long_size: desired longest image size for inference
        :param show_time: show processing time
        :param crop_type: crop regions by detected boxes or polys ("poly" or "box")
        :return:
            {
            "masks": lists of predicted masks 2d as bool array,
            "boxes": list of coords of points of predicted boxes,
            "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
            "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
            "heatmaps": visualization of the detected characters/links,
            "text_crop_paths": list of paths of the exported text boxes/polys,
            "times": elapsed times of the sub modules, in seconds
            }
        :returns:
            prediction_result
        """

        # load image
        if image is None:
            image = self.image
        elif isinstance(image, str):
            # consider image is image
            image = read_image(image)

        # perform prediction
        prediction_result = self.get_prediction(image=image,
                                                text_threshold=text_threshold,
                                                link_threshold=link_threshold,
                                                low_text=low_text,
                                                target_size=long_size,
                                                show_time=show_time)

        # arange regions
        if crop_type == "box":
            regions = prediction_result["boxes"]
        elif crop_type == "poly":
            regions = prediction_result["polys"]
        else:
            raise TypeError("crop_type can be only 'polys' or 'boxes'")

        # export if output_dir is given
        prediction_result["text_crop_paths"] = []
        if output_dir is not None:
            # export detected text regions
            exported_file_paths = export_detected_regions(
                image_path=image_path,
                image=image,
                regions=regions,
                output_dir=output_dir,
                rectify=rectify,
            )
            prediction_result["text_crop_paths"] = exported_file_paths

            # export heatmap, detection points, box visualization
            if export_extra:
                export_extra_results(
                    image_path=image_path,
                    image=image,
                    regions=regions,
                    heatmaps=prediction_result["heatmaps"],
                    output_dir=output_dir,
                )

        # return prediction results
        return prediction_result

    def __load_state_dict(self, net, weight_path):
        """
        1) Loads weights and biases.
        2) Deserialize them.
        3) Transport to cuda
        4) Make it pytorch "dataparallel"
        5) Turn it into evaluation mode.
        6) Return it.
        :param net: Artificial Neural network(model) that makes main job
        :param weight_path: Serialized pth file path with name
        :return: loaded network
        """
        net.load_state_dict(copyStateDict(torch.load(weight_path)))

        net = net.to(self.device)
        net = torch.nn.DataParallel(net)
        net.eval()
        return net

    def __load_craftnet_model(self, craft_model_path=None):
        """
        Loads craftnet network(model)
        :param craft_model_path: Serialized craftnet network(model) file path with name
        :return: loaded network
        """
        # get craft net path
        weight_path = get_weight_path(craft_model_path,
                                      self.CRAFT_GDRIVE_URL,
                                      self.craft_model_name)

        # arange device
        craft_net = self.__load_state_dict(self.craft_net, weight_path)
        return craft_net

    def __load_refinenet_model(self, refinenet_model_path=None):
        """
        Loads refinenet network(model)
        :param refinenet_model_path: Serialized refinenet network(model) file path with name
        Refiner network eliminates low probability detections.
            Default: None
        :return: loaded network
        """
        # get refine net path
        weight_path = get_weight_path(refinenet_model_path,
                                      self.REFINENET_GDRIVE_URL,
                                      "craft_refiner_CTW1500.pth")

        # arange device
        refine_net = self.__load_state_dict(self.refine_net, weight_path)
        return refine_net


if __name__ == "__main__":
    # set image path and export folder directory
    image_name = 'idcard.png'
    image_path = '../figures/' + image_name
    output_dir = 'outputs/'


    def test_oops(image_path, output_dir):
        show_time = False
        # read image
        image = read_image(image_path)
        # create craft_detector class
        pred = craft_detector(image=image, refiner=False, cuda=True)
        prediction_result = pred.detect_text(image=image_path,
                                             output_dir=output_dir,
                                             rectify=True,
                                             export_extra=False,
                                             text_threshold=0.7,
                                             link_threshold=0.4,
                                             low_text=0.4,
                                             long_size=720,
                                             show_time=show_time,
                                             crop_type="poly")
        print(len(prediction_result["boxes"]))  # 51
        print(len(prediction_result["boxes"][0]))  # 4
        print(len(prediction_result["boxes"][0][0]))  # 2
        print(int(prediction_result["boxes"][0][0][0]))  # 115
        # perform prediction
        prediction_result = pred(image=image,
                                 text_threshold=0.7,
                                 link_threshold=0.4,
                                 low_text=0.4,
                                 target_size=1280,
                                 show_time=True)
        # export detected text regions
        exported_file_paths = export_detected_regions(
            image_path=image_path,
            image=image,
            regions=prediction_result["boxes"],
            output_dir=output_dir,
            rectify=True
        )
        # export heatmap, detection points, box visualization
        export_extra_results(
            image_path=image_path,
            image=image,
            regions=prediction_result["boxes"],
            heatmaps=prediction_result["heatmaps"],
            output_dir=output_dir
        )


    test_oops(image_path, output_dir)  # Best time: 0.252/0.171
