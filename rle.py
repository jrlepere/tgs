
def rle(mask, min_acceptable=0.50):
    """
    Calculates the run length encoding of the mask

    Args:
      mask: the mask to encode
      min_acceptable: the minimum acceptable pixel value to be considered in the final mask [0,1]
    """
        
    # string for the run lengths
    run_lengths = ''

    # current index and start initialize
    current_index = 1
    current_start = 1

    # for each pixel in the mask flattened by column major
    for p in mask.flatten(order='F'):
        if p < min_acceptable:
            # run length over, if there was one
            if current_start < current_index:
                # there was, write it
                run_lengths += str(current_start) + ' ' + str(current_index - current_start) + ' '
            # reset current start to next iteration
            current_start = current_index + 1
        # increment index
        current_index += 1

    # check if mask ended in a run length
    if current_start < current_index:
        run_lengths += str(current_start) + ' ' + str(current_index - current_start) + ' '
        
    # return run length encodings
    return run_lengths[:-1]

