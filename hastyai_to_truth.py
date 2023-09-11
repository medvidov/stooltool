import os
import sys
import glob
import pickle
import imageio
import numpy as np
from itertools import zip_longest

def hastyai_to_truth(input_dir: str, output_path: str):
    """
    Converts the output of Hasty AI to the correct truth input to our SegNet for image segmentation.
    Saves out the converted truth where the key is the image uuid, and the value is the converted truth data.

    Input:
        input_dir: string path to input directory containing images for Hasty AI website
        output_path: string path to output directory to contain converted truth images for use with SegNet

    Output:
        None
    """
    # Load all of the truth data
    data = []
    ordering = []
    for image_path in glob.glob(os.path.join(input_dir, '*.png')):
        data.append(imageio.imread(image_path))
        ordering.append(image_path.split('/')[-1])

    # Convert the truth data
    data = np.array(data)
    for i, image in enumerate(data):
        temp_data = np.zeros(image.shape + (1,), dtype=np.uint8)
        y_indices, x_indices = np.where(image!=0)
        for y, x in zip_longest(y_indices, x_indices):
            temp_data[y][x] = 1
        print('Converted image ' + str(i+1) + '/' + str(data.size))
        data[i] = temp_data
    
    # Save the converted truth data
    converted_data_dictionary = {filename.split('.')[0]: truth_data for filename, truth_data in zip_longest(ordering, data)}
    if '.pkl' not in output_path:
        output_path += '.pkl'
    with open(output_path, 'wb') as output_file:
        pickle.dump(converted_data_dictionary, output_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python hastyai_to_truth.py [INPUT DIR PATH] [OUTPUT PATH]')
        sys.exit(0)
    hastyai_to_truth(sys.argv[1], sys.argv[2])   