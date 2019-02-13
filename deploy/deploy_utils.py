from PIL import Image
import numpy as np

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

    return processed_img
