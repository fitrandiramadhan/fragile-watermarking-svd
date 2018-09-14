#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser


class Watermark:
    def __init__(self, blocksize, threshold, show=False):
        self.blocksize = blocksize
        self.threshold = threshold
        self.show = show

    def read_image(self, image_path):
        # read image as ndarray
        return plt.imread(image_path)

    def pad_image(self, image):
        # pad image so that we can divide into N x N blocks evenly
        ypad = self.blocksize - (image.shape[0] % self.blocksize)
        xpad = self.blocksize - (image.shape[1] % self.blocksize)
        return np.pad(
            image,
            pad_width=((0, ypad), (0, xpad), (0, 0)),
            mode='constant',
            constant_values=0)

    def segment_block_3d(self, image):
        # assert image rank
        assert len(image.shape) == 3
        # get number of colors
        colors = image.shape[-1]
        # numpy magic trick to segment image into N x N blocks
        block_shape = (image.shape[0] // self.blocksize,
                       image.shape[1] // self.blocksize, self.blocksize,
                       self.blocksize, colors)
        pixel_size = image.strides[-1]
        _, a, b, c, d = block_shape
        stride_size = (a * b * c * d * pixel_size, b * c * d * pixel_size,
                       c * d * pixel_size, d * pixel_size, pixel_size)
        return np.lib.stride_tricks.as_strided(
            image, shape=block_shape, strides=stride_size)

    def binarize(self, blocks):
        # Compute multi-dimensional SVD
        lambdas = np.linalg.svd(blocks, full_matrices=False, compute_uv=False)
        # Get first lambda
        lambda1 = lambdas[:, :, 0]
        lambda1[lambda1 == 0] = self.blocksize
        # Compute binary
        divisor = (1 / lambda1).reshape((lambda1.shape[0], lambda1.shape[1], 1,
                                         1, lambda1.shape[-1]))
        binary = (divisor * blocks) >= self.threshold
        # Reassemble blocks into image
        binary = np.dstack([
            binary[:, :, :, :, n].reshape((blocks.shape[0] * self.blocksize,
                                           blocks.shape[1] * self.blocksize))
            for n in range(blocks.shape[-1])
        ])
        return binary

    def embed(self, padded_image, binary):
        # Replace LSB with binarized image:
        # (a & ~1) | b replaces LSB with b regardless of value of b
        return np.bitwise_or(padded_image, binary)

    def encrypt(self, src, dest=None):
        # Read image and replace LSB with 0
        # a & ~1 replace LSB with 0
        img = np.bitwise_and(plt.imread(src), ~1).astype(np.uint8)
        # Pad image
        pad = self.pad_image(img)
        # Compute binarized image
        binary = self.binarize(self.segment_block_3d(pad))
        if self.show:
            plt.imshow(np.all(binary, axis=-1))
            plt.show()
        # Embed watermark
        watermark = self.embed(pad, binary)
        # Return to original dimensions
        x, y = img.shape[:2]
        if not dest:
            return watermark[:x, :y]
        else:
            plt.imsave(dest, watermark[:x, :y])

    def decrypt(self, src, dest=None):
        """
        Inspect whether image has been tampered with.

        Parameters
        ----------
        src : image file path to 'authentic' image
        ref : image file path to file to inspect
        """
        # Read image
        img = plt.imread(src)
        # Extract LSB: a & 1
        lsb = np.bitwise_and(img, 1).astype(bool)
        # a & ~1 replace LSB with 0
        img = np.bitwise_and(img, ~1).astype(np.uint8)
        # Pad image
        pad = self.pad_image(img)
        # Compute binarized image
        x, y = img.shape[:2]
        binary = self.binarize(self.segment_block_3d(pad))[:x, :y]

        xor = np.bitwise_xor(binary, lsb)
        if self.show:
            plt.imshow(np.any(xor, axis=-1))
            plt.show()
        elif dest:
            plt.imsave(dest, xor)


if __name__ == '__main__':
    parser = ArgumentParser(description="Implements fragile watermarking")
    parser.add_argument('-i', '--input', help="Path to original image")
    parser.add_argument('-o', '--output', help="Path to watermarked image")
    parser.add_argument(
        '-d',
        '--decrypt',
        action='store_true',
        help="Whether to decrypt or encrypt image. Defaults to encryption")
    parser.add_argument(
        '-b', '--blocksize', help="Number of dimensions of block", default=4)
    parser.add_argument(
        '-t', '--threshold', help="Binarization threshold", default=0.25)
    parser.add_argument(
        '--show', action='store_false', help="Show binary image")
    args = parser.parse_args()

    wm = Watermark(int(args.blocksize), float(args.threshold), args.show)
    if args.decrypt:
        wm.decrypt(args.input, args.output)
    else:
        wm.encrypt(args.input, args.output)
