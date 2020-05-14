https://pypi.org/project/craft-text-detector/
# CRAFT: Character-Region Awareness For Text detection

Packaged, Pytorch-based, easy to use, cross-platform version of the CRAFT text detector | 
[Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) | [Supplementary](https://youtu.be/HI8MzpY8KMI)

**[Youngmin Baek](mailto:youngmin.baek@navercorp.com), Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee.**
 
## Overview

PyTorch implementation for CRAFT text detector that effectively detect text area by exploring 
each character region and affinity between characters. The bounding box of texts are obtained 
by simply finding minimum bounding rectangles on binary map after thresholding character region 
and affinity scores.

<img width="1172" alt="teaser" src="./figures/craft_example.gif">

## Getting started
### Install dependencies
#### Requirements
- torch>=0.4.1
- torchvision>=0.2.1
- opencv-python>=3.4.8.29
- scipy>=1.3.2
- gdown>=3.10.1
- setuptools~=46.2.0
- numpy~=1.18.1
- check requiremtns.txt
```
pip install -r requirements.txt
```

#### Install using pip for Linux and Mac:

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
prediction_result = craft.detect_text(image_path,output_dir, cuda=False, crop_type="poly")
```

### Advanced Usage

```python
# import package
import craft_text_detector as craft

# set image path and export folder directory
image_name = 'idcard.png'
image_path = 'figures/' + image_name
output_dir = 'outputs/'

# read image
image = craft.read_image(image_path)

# load models
craft_net = craft.craft_detector(image=image, refiner=False, cuda=True)

# perform prediction
text_threshold = 0.9
link_threshold = 0.2
low_text = 0.2
cuda = True  # False
show_time = False
# perform prediction
prediction_result = craft_net(image=image,
                         text_threshold=0.7,
                         link_threshold=0.4,
                         low_text=0.4,
                         target_size=1280,
                         show_time=True)
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
