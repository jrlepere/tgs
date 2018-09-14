"""
Test the run length encoding in TGS.
"""

import csv
import cv2
from tgs import TGS
from glob import glob
import numpy as np
from os.path import basename


# load and return the run lengths per training file in a dictionary
def load_run_lengths(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)    # skip header
        return dict((rows[0], rows[1]) for rows in reader)


# load and return the training masks
def load_masks(mask_filenames):
    num_examples = len(mask_filenames)
    masks = np.zeros((num_examples, 101, 101, 1), dtype=np.float32)
    for i in range(num_examples):
        masks[i] = np.reshape(cv2.imread(mask_filenames[i], cv2.IMREAD_GRAYSCALE), (101, 101, 1))
    return masks/255


if __name__ == '__main__':
    
    # creat TGS object
    tgs = TGS(num_partitions=1)

    # load training run lengths provided
    run_lengths = load_run_lengths('./train.csv')
    
    # load the masks
    mask_filenames = glob('./train/masks/*png')
    masks = load_masks(mask_filenames)

    # counter for the number of errors
    errs = 0

    # compare the actual vs tgs computed run length
    for i in range(len(mask_filenames)):
        if run_lengths[basename(mask_filenames[i])[:-4]] != tgs._get_run_length_encoding(masks[i], min_acceptable=0.50):
            errs += 1
    
    # print error count
    print('ERRORS: %d' % errs)

