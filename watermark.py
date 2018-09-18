#!/usr/bin/env python
import os
import numpy as np
import imageio as iio
import matplotlib
from argparse import ArgumentParser
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


class Watermark:
    def __init__(self, blocksize, threshold, show=False, key=64):
        self.blocksize = blocksize
        self.threshold = threshold
        self.show = show
        self.transforms = 1
        self.arn_x = None
        self.arn_y = None
        self.key = key

    def read_image(self, image_path):
        # read image as ndarray
        return iio.imread(image_path)

    def pad_image(self, image):
        if image.shape[0] == image.shape[
                1] and image.shape[0] % self.blocksize == 0:
            return image
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

    def arnold(self, img, n=0):
        # Original coordinates of image
        x, y = np.meshgrid(
            np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
        if n != self.transforms or self.arn_x is None or self.arn_y is None:
            # Temporary array to store intermediate results
            tmp_x = x.copy()
            tmp_y = y.copy()
            # Arrays to store results of tranformation
            arn_x = np.empty(x.shape, x.dtype)
            arn_y = np.empty(y.shape, y.dtype)
            # Transform n number of times
            for _ in range(n):
                arn_x[:] = (2 * tmp_x + tmp_y) % img.shape[0]
                arn_y[:] = (tmp_x + tmp_y) % img.shape[1]
                tmp_x[:] = arn_x
                tmp_y[:] = arn_y
            self.arn_x = arn_x
            self.arn_y = arn_y
            self.transforms = n
        else:
            tmp_x = self.arn_x
            tmp_y = self.arn_y
        # Rearrange pixels in place
        if n > 0:
            img[tmp_x, tmp_y] = img[x, y]

    def embed(self, padded_image, binary):
        # Replace LSB with binarized image:
        # (a & ~1) | b replaces LSB with b regardless of value of b
        return np.bitwise_or(padded_image, binary)

    def encrypt(self, src, n_trans=0):
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
        # Encrypt binarized image with Arnold cat map
        self.arnold(binary, n_trans)
        # Embed watermark
        watermark = self.embed(pad, binary)
        # Return to original dimensions
        x, y = img.shape[:2]
        img[:] = watermark[:x, :y]
        if self.show:
            plt.imshow(np.all(binary, axis=-1))
            plt.show()
        else:
            return img

    def decrypt(self, src, n_trans=0):
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
        # Decrypt using Arnold cat map
        self.arnold(lsb, n_trans)
        # Localize alterations (if any)
        xor = np.bitwise_xor(binary, lsb)
        if self.show:
            plt.imshow(np.any(xor, axis=-1))
            plt.show()
        else:
            return xor

    def encrypt_4d(self, src, n_trans=0):
        """
        Runs encryption on videos. Assumes a square matrix divisible by 4.
        """
        # Stack images
        vid = np.stack(src)
        vid[:] = np.bitwise_and(vid, ~1)


    def gif_encrypt(self, src, dest=None):
        ext = os.path.splitext(src)[-1]
        if ext in ['.gif', '.mp4']:
            imgs = iio.mimread(src)
            imgs = [self.encrypt(n) for n in imgs]
            if not self.show:
                iio.mimwrite(dest, imgs, format='FFMPEG')
        else:
            img = iio.imread(src)
            img = self.encrypt(img, n_trans=self.key)
            if not self.show:
                iio.imwrite(dest, img)

    def gif_decrypt(self, src, dest=None):
        ext = os.path.splitext(src)[-1]
        if ext in ['.gif', '.mp4']:
            imgs = iio.mimread(src)
            imgs = [self.decrypt(n, n_trans=(192 - len(imgs))) for n in imgs]
            if not self.show:
                iio.mimwrite(dest, imgs)
        else:
            img = iio.imread(src)
            img = self.decrypt(img, n_trans=(192 - self.key))
            if not self.show:
                iio.imwrite(dest, img)


if __name__ == '__main__':
    parser = ArgumentParser(description="Implements fragile watermarking")
    parser.add_argument('-i', '--input', help="Path to original image")
    parser.add_argument(
        '-o', '--output', help="Path to watermarked image", default=None)
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
    parser.add_argument(
        '-k', '--key', help="Number of arnold transforms to perform",
        default=64
    )
    args = parser.parse_args()

    if not args.output:
        args.show = True
    wm = Watermark(int(args.blocksize), float(args.threshold), args.show,
                   int(args.key))
    if args.decrypt:
        wm.gif_decrypt(args.input, args.output)
    else:
        wm.gif_encrypt(args.input, args.output)
