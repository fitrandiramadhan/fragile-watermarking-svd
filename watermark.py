#!/usr/bin/env python
import numpy as np
import imageio as iio
import matplotlib
from argparse import ArgumentParser
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


class Watermark:
    def __init__(self, blocksize, threshold, show=False):
        self.blocksize = blocksize
        self.threshold = threshold
        self.show = show

    def read_image(self, image_path):
        # read image as ndarray
        return iio.imread(image_path)

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

    def arnold(self, img, binary, key1, key2):
        assert img.shape[0] == img.shape[1]

    def embed(self, padded_image, binary):
        # Replace LSB with binarized image:
        # (a & ~1) | b replaces LSB with b regardless of value of b
        return np.bitwise_or(padded_image, binary)

    def encrypt(self, src, dest=None):
        # Read image and replace LSB with 0
        if type(src) == str:
            img = iio.imread(src)
        else:
            img = src
        # a & ~1 replace LSB with 0
        img[:] = np.bitwise_and(img, ~1)
        # Pad image
        pad = self.pad_image(img)
        # Compute binarized image
        binary = self.binarize(self.segment_block_3d(pad))
        # Embed watermark
        watermark = self.embed(pad, binary)
        # Return to original dimensions
        x, y = img.shape[:2]
        img[:] = watermark[:x, :y]
        if self.show:
            plt.imshow(np.all(binary, axis=-1))
            plt.show()
        elif dest:
            iio.imwrite(dest, img)
        else:
            return img

    def decrypt(self, src, dest=None):
        """
        Inspect whether image has been tampered with.

        Parameters
        ----------
        src : image file path to 'authentic' image
        ref : image file path to file to inspect
        """
        # Read image
        if type(src) == str:
            img = iio.imread(src)
        else:
            img = src
        # Extract LSB: a & 1
        lsb = np.bitwise_and(img, 1)
        # a & ~1 replace LSB with 0
        img[:] = np.bitwise_and(img, ~1)
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
            iio.imwrite(dest, xor)
        else:
            return xor

    def gif_encrypt(self, src, dest):
        imgs = [self.encrypt(n) for n in iio.mimread(src)]
        iio.mimwrite(dest, imgs)

    def gif_decrypt(self, src, dest):
        imgs = [self.decrypt(n) for n in iio.mimread(src)]
        iio.mimwrite(dest, imgs)


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
        '--show', action='store_true', help="Show binary image")
    args = parser.parse_args()

    wm = Watermark(int(args.blocksize), float(args.threshold), args.show)
    if args.decrypt:
        wm.gif_decrypt(args.input, args.output)
    else:
        wm.gif_encrypt(args.input, args.output)
