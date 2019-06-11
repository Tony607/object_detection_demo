"""
## Example to benchmark SSD mobileNet V2 on Neural Compute stick.
```
python openvino_inference_benchmark.py\
     --model-dir ./models/ssd_mobilenet_v2_custom_trained/FP16\
     --device MYRIAD\
     --img ../test/15.jpg
```
"""

import os
import sys
import time
import glob
import platform
from PIL import Image
import numpy as np

# Check path like C:\Intel\computer_vision_sdk\python\python3.5 or ~/intel/computer_vision_sdk/python/python3.5 exists in PYTHONPATH.
is_win = "windows" in platform.platform().lower()
""" 
# OpenVINO 2018.
if is_win:
    message = "Please run `C:\\Intel\\computer_vision_sdk\\bin\\setupvars.bat` before running this."
else:
    message = "Add the following line to ~/.bashrc and re-run.\nsource ~/intel/computer_vision_sdk/bin/setupvars.sh"
"""

# OpenVINO 2019.
if is_win:
    message = 'Please run "C:\Program Files (x86)\IntelSWTools\openvino_2019.1.133\bin\setupvars.bat" before running this.'
else:
    message = "Add the following line to ~/.bashrc and re-run.\nsource /opt/intel/openvino/bin/setupvars.sh"

try:
    from openvino import inference_engine as ie
    from openvino.inference_engine import IENetwork, IEPlugin
except Exception as e:
    exception_type = type(e).__name__
    print(
        "The following error happened while importing Python API module:\n[ {} ] {}".format(
            exception_type, e
        )
    )
    sys.exit(1)


def pre_process_image(imagePath, img_shape):
    """pre process an image from image path.
    
    Arguments:
        imagePath {str} -- input image file path.
        img_shape {tuple} -- Target height and width as a tuple.
    
    Returns:
        np.array -- Preprocessed image.
    """

    # Model input format
    assert isinstance(img_shape, tuple) and len(img_shape) == 2

    n, c, h, w = [1, 3, img_shape[0], img_shape[1]]
    image = Image.open(imagePath)
    processed_img = image.resize((h, w), resample=Image.BILINEAR)

    processed_img = np.array(processed_img).astype(np.uint8)

    # Change data layout from HWC to CHW
    processed_img = processed_img.transpose((2, 0, 1))
    processed_img = processed_img.reshape((n, c, h, w))

    return processed_img, np.array(image)


if __name__ == "__main__":
    import argparse

    # python argparse_test.py 5 -v --color RED
    parser = argparse.ArgumentParser(description="OpenVINO Inference speed benchmark.")
    # parser.add_argument("-v", "--verbose", help="increase output verbosity",
    #                     action="store_true")
    parser.add_argument(
        "--model-dir",
        help="Directory where the OpenVINO IR .xml and .bin files exist.",
        type=str,
    )
    parser.add_argument(
        "--device", help="Device to run inference: GPU, CPU or MYRIAD", type=str
    )
    parser.add_argument("--img", help="Path to a sample image to inference.", type=str)
    args = parser.parse_args()

    # Directory to model xml and bin files.
    output_dir = args.model_dir
    assert os.path.isdir(output_dir), "`{}` does not exist".format(output_dir)

    # Devices: GPU (intel), CPU or MYRIAD
    plugin_device = args.device
    # Converted model take fixed size image as input,
    # we simply use same size for image width and height.
    img_height = 300

    DATA_TYPE_MAP = {"GPU": "FP16", "CPU": "FP32", "MYRIAD": "FP16"}
    assert (
        plugin_device in DATA_TYPE_MAP
    ), "Unsupported device: `{}`, not found in `{}`".format(
        plugin_device, list(DATA_TYPE_MAP.keys())
    )

    # Path to a sample image to inference.
    img_fname = args.img
    assert os.path.isfile(img_fname)

    # Plugin initialization for specified device and load extensions library if specified.
    plugin_dir = None
    model_xml = glob.glob(os.path.join(output_dir, "*.xml"))[-1]
    model_bin = glob.glob(os.path.join(output_dir, "*.bin"))[-1]
    # Devices: GPU (intel), CPU, MYRIAD
    plugin = IEPlugin(plugin_device, plugin_dirs=plugin_dir)
    # Read IR
    net = IENetwork(model=model_xml, weights=model_bin)
    assert len(net.inputs.keys()) == 1
    assert len(net.outputs) == 1
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    # Load network to the plugin
    exec_net = plugin.load(network=net)
    del net

    # Run inference
    img_shape = (img_height, img_height)
    processed_img, image = pre_process_image(img_fname, img_shape)
    res = exec_net.infer(inputs={input_blob: processed_img})

    print(res["DetectionOutput"].shape)

    probability_threshold = 0.5
    preds = [
        pred for pred in res["DetectionOutput"][0][0] if pred[2] > probability_threshold
    ]

    for pred in preds:
        class_label = pred[1]
        probability = pred[2]
        print(
            "Predict class label:{}, with probability: {}".format(
                class_label, probability
            )
        )

    times = []
    for i in range(20):
        start_time = time.time()
        res = exec_net.infer(inputs={input_blob: processed_img})
        delta = time.time() - start_time
        times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1 / mean_delta
    print('average(sec):{:.3f},fps:{:.2f}'.format(mean_delta,fps))
