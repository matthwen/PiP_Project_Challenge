# -*- coding: utf-8 -*-
import os
from matplotlib import pyplot as plt
import numpy as np


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets, and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots(2, 2)
    ax[1, 1].remove()

    for i in range(len(inputs)):
        ax[0, 0].clear()
        ax[0, 0].set_title('input')
        ax[0, 0].imshow(inputs[i, 0], cmap=plt.cm.gray, interpolation='none')
        ax[0, 0].set_axis_off()
        ax[0, 1].clear()
        ax[0, 1].set_title('targets')
        ax[0, 1].imshow(targets[i], cmap=plt.cm.gray, interpolation='none')
        ax[0, 1].set_axis_off()
        ax[1, 0].clear()
        ax[1, 0].set_title('predictions')
        ax[1, 0].imshow(predictions[i], cmap=plt.cm.gray, interpolation='none')
        ax[1, 0].set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=1000)
    del fig


def crop(image_array, crop_size, crop_center):
    # if not isinstance(image_array, np.ndarray):
    #    raise ValueError("Input image is not a numpy array")
    # if len(image_array.shape) != 2:
    #    raise ValueError("numpy array is not a 2d array")
    if len(crop_size) != 2:
        raise ValueError("crop size does not contain 2 Objects")
    if len(crop_center) != 2:
        raise ValueError("crop center does not contain 2 Objects")
    if crop_size[0] % 2 == 0 or crop_size[1] % 2 == 0:
        raise ValueError("crop size is not all odd numbers")
    crop_x_len = crop_size[0]
    crop_y_len = crop_size[1]
    crop_x_start = crop_center[0] - ((crop_x_len - 1) // 2)
    crop_x_end = crop_center[0] + ((crop_x_len - 1) // 2)
    crop_y_start = crop_center[1] - ((crop_y_len - 1) // 2)
    crop_y_end = crop_center[1] + ((crop_y_len - 1) // 2)
    if crop_x_start < 20 or crop_y_start < 20 or crop_x_end > image_array.shape[0] - 20 or \
            crop_y_end > image_array.shape[1] - 20:
        raise ValueError(
            "minimal distance between the to-be cropped-out rectangle and the border of image_array is less than 20 pixels")

    crop_array = np.zeros(image_array.shape, dtype=image_array.dtype)
    for i in range(crop_x_len):
        for j in range(crop_y_len):
            crop_array[i + crop_x_start][j + crop_y_start] = 1

    target_array = image_array[crop_x_start:crop_x_end + 1, crop_y_start:crop_y_end + 1]
    image_array = image_array * (1 - crop_array)
    return image_array, crop_array, target_array


def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    # for line in sorted(tensors.items()):
    for line in tensors.items():
        print('{}\t{}'.format(*line))
