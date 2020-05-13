from __future__ import absolute_import

__version__ = "0.2.2"
__author__ = "Mustafa Selçuk Çağlar"

from craft_text_detector.imgproc import read_image

from craft_text_detector.file_utils import export_detected_regions, export_extra_results

from craft_text_detector.predict import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    predict,
)


def detect_text(image_path, output_dir=None, rectify=True, export_extra=True, text_threshold=0.7, link_threshold=0.4,
                low_text=0.4, long_size=1280, cuda=False, show_time=False, refiner=True, crop_type="poly"):
    """
    Arguments:
        image_path: path to the image to be processed
        output_dir: path to the results to be exported
        rectify: rectify detected polygon by affine transform
        export_extra: export heatmap, detection points, box visualization
        text_threshold: text confidence threshold
        link_threshold: link confidence threshold
        low_text: text low-bound score
        cuda: Use cuda for inference
        long_size: desired longest image size for inference
        show_time: show processing time
        refiner: enable link refiner
        crop_type: crop regions by detected boxes or polys ("poly" or "box")
    Output:
        {"masks": lists of predicted masks 2d as bool array,
         "boxes": list of coords of points of predicted boxes,
         "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
         "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
         "heatmaps": visualization of the detected characters/links,
         "text_crop_paths": list of paths of the exported text boxes/polys,
         "times": elapsed times of the sub modules, in seconds}
    """
    # load craft model
    craft_net = predict(image=image_path, refiner=refiner, cuda=cuda)

    prediction_result = craft_net.detect_text(image=image_path, output_dir=output_dir, rectify=rectify,
                                              export_extra=export_extra,
                                              text_threshold=text_threshold, link_threshold=link_threshold,
                                              low_text=low_text, long_size=long_size, show_time=show_time,
                                              crop_type=crop_type)

    # return prediction results
    return prediction_result
