#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tif_to_nii

command line executable to convert a directory of tif images
(from one image) to a nifti image stacked along a user-specified axis

call as: python tif_to_nii.py /path/to/tif/ /path/to/nifti
(append optional arguments to the call as desired)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
"""

import argparse
from glob import glob
import os
import sys

from PIL import Image
import nibabel as nib
import numpy as np


def arg_parser():
    parser = argparse.ArgumentParser(description="merge 2d tif images into a 3d image")
    parser.add_argument("img_dir", type=str, help="path to tiff image directory")
    parser.add_argument(
        "out_dir", type=str, help="path to output the corresponding tif image slices"
    )
    parser.add_argument(
        "-a", "--axis", type=int, default=2, help="axis on which to stack the 2d images"
    )
    return parser


def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == ".gz":
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def main():
    try:
        args = arg_parser().parse_args()
        fns = sorted(glob(os.path.join(args.img_dir, "*.tif*")))
        imgs = []
        for fn in fns:
            _, base, ext = split_filename(fn)
            img = np.asarray(Image.open(fn)).astype(np.float32).squeeze()
            if img.ndim != 2:
                raise Exception(
                    f"Only 2D data supported. File {base}{ext} has dimension {img.ndim}."
                )
            imgs.append(img)
        img = np.stack(imgs, axis=args.axis)
        nib.Nifti1Image(img, None).to_filename(
            os.path.join(args.out_dir, f"{base}.nii.gz")
        )
        return 0
    except Exception as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
