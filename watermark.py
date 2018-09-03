#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser


class Watermark:
    def __init__(self, blocksize, threshold):
        self.blocksize = blocksize
        self.threshold = threshold

    def read_image(self, image_path):
        # read image as ndarray
        self.img = plt.imread(image_path)

    def segment_block_3d(self):
        # assert image rank
        assert len(self.img.shape) == 3
        # get number of colors
        colors = self.img.shape[-1]
        # pad image so that we can divide into N x N blocks evenly
        ypad = self.blocksize - (self.img.shape[0] % self.blocksize)
        xpad = self.blocksize - (self.img.shape[1] % self.blocksize)
        img = np.pad(self.img, pad_width=((0, ypad), (0, xpad), (0, 0)),
                     mode='constant', constant_values=0)
        # numpy magic trick to segment image into N x N blocks
        block_shape = (self.img.shape[0] // self.blocksize,
                       self.img.shape[1] // self.blocksize,
                       self.blocksize, self.blocksize, colors)
        pixel_size = self.img.strides[-1]
        _, a, b, c, d = block_shape
        stride_size = (a*b*c*d*pixel_size, b*c*d*pixel_size, c*d*pixel_size,
                      d*pixel_size, pixel_size)
        blocks = np.lib.stride_tricks.as_strided(self.img, shape=block_shape,
                                                 strides=stride_size)
        return blocks

    def binarize(self, blocks):
        # Compute multi-dimensional SVD
        lambdas = np.linalg.svd(blocks, compute_uv=False)
        # Get first lambda
        lambda1 = lambdas[:, :, 0]
        lambda1[lambda1 == 0] = self.blocksize
        # Compute binary
        divisor = (1 / lambda1).reshape((lambda1.shape[0], lambda1.shape[1], 1,
                                        1, lambda1.shape[-1]))
        binary = (divisor * blocks) < self.threshold
        # Reassemble blocks into image
        binary = np.dstack(
            [binary[:, :, :, :, n].reshape((blocks.shape[0] * self.blocksize,
                                            blocks.shape[1] * self.blocksize))
             for n in range(blocks.shape[-1])]
        )
        return binary

    def encrypt(self, image_path, message):
        self.read_image(image_path)
        binary = self.binarize(self.segment_block_3d())
        plt.imshow(np.any(binary, axis=-1))
        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description="Implements fragile watermarking")
    parser.add_argument('-i', '--input', help="Path to image")
    parser.add_argument('-b', '--blocksize', help="Number of dimensions of \
                        block", default=4)
    parser.add_argument('-t', '--threshold', help="Binarization threshold",
                        default=0.24)
    args = parser.parse_args()

    wm = Watermark(args.blocksize, args.threshold)
    wm.encrypt(args.input, None)
