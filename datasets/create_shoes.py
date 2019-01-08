import os
import sys
import re
import zipfile

import numpy as np
import h5py

import requests
from PIL import Image

outfile = 'shoes.hdf5'
image_file = 'img_shoes.zip'
attr_file = 'list_attr_shoes.txt'

def main():
    # Parse labels
    with open(attr_file, 'r') as lines:
        lines = [l.strip() for l in lines]
        num_images = int(lines[0])

        label_names = re.split(' ', lines[1])
        label_names = np.array(label_names, dtype=object)
        num_labels = len(label_names)

        lines = lines[2:]
        labels = np.ndarray((num_images, num_labels), dtype='uint8')
        for i in range(num_images):
            label = [int(l) for l in re.split(' ', lines[i])[1:]]
            label = np.maximum(0, label).astype(np.uint8)
            labels[i] = label

    ## Parse images
    with zipfile.ZipFile(image_file, 'r', zipfile.ZIP_DEFLATED) as zf:
        image_files = [f for f in zf.namelist()]
        image_files = sorted(image_files)
        image_files = list(filter(lambda f: f.endswith('.jpg'), image_files))

        num_images = len(image_files)
        print('%d images' % (num_images))

        image_data = np.ndarray((num_images, 64, 64, 3), dtype='uint8')
        for i, f in enumerate(image_files):
            image = Image.open(zf.open(f, 'r')).resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
            image = np.asarray(image, dtype='uint8')
            image_data[i] = image
            print('%d / %d' % (i + 1, num_images), end='\r', flush=True)

    # Create HDF5 file
    h5 = h5py.File(outfile, 'w')
    string_dt = h5py.special_dtype(vlen=str)
    dset = h5.create_dataset('images', data=image_data, dtype='uint8')
    dset = h5.create_dataset('label_names', data=label_names, dtype=string_dt)
    dset = h5.create_dataset('labels', data=labels, dtype='uint8')

    h5.flush()
    h5.close()

if __name__ == '__main__':
    main()
