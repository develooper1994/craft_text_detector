https://pypi.org/project/craft-text-detector/
# CRAFT: Character-Region Awareness For Text detection

[![CI](https://github.com/fcakyon/craft-text-detector/workflows/CI/badge.svg)](https://github.com/fcakyon/craft-text-detector/actions?query=event%3Apush+branch%3Amaster+is%3Acompleted+workflow%3ACI)

Packaged, Pytorch-based, easy to use, cross-platform version of the CRAFT text detector | 
[Paper](https://arxiv.org/abs/1904.01941) |

## Overview

PyTorch implementation for CRAFT text detector that effectively detect text area by exploring 
each character region and affinity between characters. The bounding box of texts are obtained 
by simply finding minimum bounding rectangles on binary map after thresholding character region 
and affinity scores.

<img width="1172" alt="teaser" src="./figures/craft_example.gif">

## Getting started

- Install using pip for Linux and Mac:

```console
pip install git+https://github.com/develooper1994/craft_text_detector
```
or
```download and install
python setup.py install
```

### Basic Usage

```python
# import package
import craft_text_detector as craft

# set image path and export folder directory
image_path = 'figures/idcard.png'
output_dir = 'outputs/'

# apply craft text detection and export detected regions to output directory
prediction_result = craft.detect_text(image_path,output_dir,cuda=False,crop_type="poly")
```

### Advanced Usage

```python
# import package
import craft_text_detector as craft

# set image path and export folder directory
image_path = 'figures/idcard.png'
output_dir = 'outputs/'

# read image
image = craft.read_image(image_path)

# load models
craft_net = craft.load_craftnet_model()
refine_net = craft.load_refinenet_model()

# perform prediction
prediction_result = craft.get_prediction(image=image,craft_net=craft_net,refine_net=refine_net,text_threshold=0.7,link_threshold=0.4,low_text=0.4,long_size=1280,cuda=True,show_time=True)

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
```
