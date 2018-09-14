"""
Orders training data to evenly distribute class coverage for each training fold.

Author: Jake Lepere
Date: 09/03/2018
"""
import numpy as np

def order_input(images, masks, depths, num_folds=5, by='mask_count'):
    """
    Evenly spreads out the input data to maximize class coverage for all training folds.

    Args:
      images: the images
      masks: the masks
      depths: the depths
      num_folds: the number of validation folds
      by: arrange by 'mask_count' or 'depth'
    """

    # number of training examples
    num_examples = images.shape[0]

    # sort the data by weighting the mask and depth, where depth is the tie breaker if the mask sum is the same
    min_depth = np.amin(depths)
    max_depth = np.amax(depths)
    sorted_indexes = np.argsort([np.sum(masks[i]) + (depths[i] - min_depth)/(max_depth - min_depth) for i in range(num_examples)])

    # indexes after ordering
    ordered_indexes = []

    # order the indexes
    for i in range(num_folds):
        
        # starting index
        j = i

        # j = (i, i+num_folds, i+2*num_folds, i+3*num_fords, ...)
        while j < num_examples:
            
            # add index
            ordered_indexes.append(j)

            # increment
            j += num_folds

    # return ordered training data
    return images[ordered_indexes], masks[ordered_indexes], depths[ordered_indexes]

