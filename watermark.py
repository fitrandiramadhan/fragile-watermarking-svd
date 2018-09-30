#!/usr/bin/env python
import os
import numpy as np
import imageio as iio
import skvideo.io as skv
import matplotlib
import logging
from argparse import ArgumentParser
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class Watermark:
    def __init__(self, blocksize, threshold, show=False, key=64):
        self.blocksize = blocksize
        self.threshold = threshold
        self.show = show
        self.transforms = 1
        self.arn_x = None
        self.arn_y = None
        self.key = key

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
            plt.imshow(np.all(xor, axis=-1))
            plt.show()
        return xor * 192

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

    def encrypt_img(self, src, dest):
        img = self.encrypt(src)
        iio.imwrite(dest, img)

    def decrypt_img(self, src, dest):
        img = self.decrypt(src)
        iio.imwrite(dest, img)

    def encrypt_vid(self, src, dest):
        """
        Runs encryption on videos. Assumes a square matrix divisible by 4.
        """
        vid = skv.vread(src)
        # set LSB to 0
        vid[:] = np.bitwise_and(vid, ~1)
        binary = self.binarize_4d(vid)
        self.arnold_4d(binary, n=len(vid))
        # embed watermark
        log.info('Embedding watermark and saving')
        vid[:] = np.bitwise_or(vid, binary)
        params = {'-c:v': 'ffv1'}
        with skv.FFmpegWriter(dest, outputdict={'-c:v': 'ffv1'}) as f:
            [f.writeFrame(vid[n]) for n in range(len(vid))]

    def decrypt_vid(self, src, dest):
        vid = skv.vread(src)
        # extract LSB and then set to 0
        lsb = np.bitwise_and(vid, 1).astype(bool)
        vid[:] = np.bitwise_and(vid, ~1)
        binary = self.binarize_4d(vid)
        self.arnold_4d(lsb, n=(192 - len(vid)))
        # localize
        xor = np.bitwise_xor(binary, lsb)
        params = {'-c:v': 'ffv1'}
        with skv.FFmpegWriter(dest, outputdict={'-c:v': 'ffv1'}) as f:
            [f.writeFrame(xor[n] * 192) for n in range(len(vid))]

    def tamper_vid(self, src, dest, n_tamper=20, save=False):
        vid = skv.vread(src)
        t, x, y, _ = vid.shape
        rand_t = np.random.randint(t, size=n_tamper)
        rand_x = np.random.randint(x, size=n_tamper)
        rand_y = np.random.randint(y, size=n_tamper)
        rand_z = np.random.randint(255, size=n_tamper)
        for n in range(n_tamper):
            vid[rand_t[n], rand_x[n]:rand_y[n], rand_x[n]:
                rand_y[n], :] = rand_z[n]
        with skv.FFmpegWriter(dest, outputdict={'-c:v': 'ffv1'}) as f:
            [f.writeFrame(vid[n]) for n in range(len(vid))]
        if save:
            tmp = np.zeros(vid.shape, dtype=vid.dtype)
            for n in range(n_tamper):
                tmp[rand_t[n], rand_x[n]:rand_y[n], rand_x[n]:
                    rand_y[n], :] = rand_z[n]
            with skv.FFmpegWriter(
                    dest.replace('.avi', '_tmp.avi'),
                    outputdict={'-c:v': 'ffv1'}) as f:
                [f.writeFrame(tmp[n]) for n in range(len(tmp))]

    def binarize_4d(self, vid):
        assert len(vid.shape) == 4
        log.info('Segmenting into blocks of size %d' % self.blocksize)
        frames, x, y, colors = vid.shape
        blksz = self.blocksize
        pixsz = vid.strides[-1]
        block_shape = (frames, x // blksz, y // blksz, blksz, blksz, colors)
        _, a, b, c, d, e = block_shape
        stride_size = (a * b * c * d * e * pixsz, b * c * d * e * pixsz,
                       c * d * e * pixsz, d * e * pixsz, e * pixsz, pixsz)
        blocks = np.lib.stride_tricks.as_strided(
            vid, shape=block_shape, strides=stride_size)
        log.info('Computing SVD')
        lambdas = np.linalg.svd(blocks, full_matrices=False, compute_uv=False)
        log.info('Thresholding')
        lambda1 = lambdas[:, :, :, 0]
        lambda1[lambda1 == 0] = self.blocksize
        divisor = (1 / lambda1).reshape(
            (lambda1.shape[0], lambda1.shape[1], lambda1.shape[2], 1, 1,
             lambda1.shape[-1]))
        binary = (divisor * blocks) >= self.threshold
        return np.lib.stride_tricks.as_strided(binary, shape=vid.shape)

    def arnold_4d(self, vid, n=0):
        log.info('Calculating Arnold transform')
        x, y = np.meshgrid(
            np.arange(vid.shape[1]), np.arange(vid.shape[2]), indexing='ij')
        if n != self.transforms or self.arn_x is None or self.arn_y is None:
            tmp_x = x.copy()
            tmp_y = y.copy()
            arn_x = np.empty(x.shape, x.dtype)
            arn_y = np.empty(y.shape, y.dtype)
            for _ in range(n):
                arn_x[:] = (2 * tmp_x + tmp_y) % vid.shape[1]
                arn_y[:] = (tmp_x + tmp_y) % vid.shape[2]
                tmp_x[:] = arn_x
                tmp_y[:] = arn_y
            self.arn_x = arn_x
            self.arn_y = arn_y
            self.transforms = n
        else:
            tmp_x = self.arn_x
            tmp_y = self.arn_y
        if n > 0:
            vid[:, tmp_x, tmp_y] = vid[:, x, y]

    def score(self, ref_path, hyp_path):
        from sklearn.metrics import f1_score
        ref = skv.vread(ref_path)
        hyp = skv.vread(hyp_path)
        score = f1_score(
            np.any(ref, axis=-1).flatten(),
            np.any(hyp, axis=-1).flatten())
        print(score)


if __name__ == '__main__':
    parser = ArgumentParser(description="Implements fragile watermarking")
    parser.add_argument('-i', '--input', help="Path to original image")
    parser.add_argument(
        '-o', '--output', help="Path to watermarked image", default=None)
    parser.add_argument(
        '-v',
        '--video',
        action='store_true',
        help="Whether to assume video. Defaults to false (image)")
    parser.add_argument(
        '-b', '--blocksize', help="Number of dimensions of block", default=4)
    parser.add_argument(
        '-t', '--threshold', help="Binarization threshold", default=0.25)
    parser.add_argument(
        '-m',
        '--mode',
        help="Mode to operate in: encrypt, decrypt, tamper",
        default='encrypt')
    parser.add_argument(
        '--show', action='store_true', help="Show binary image")
    parser.add_argument(
        '-k',
        '--key',
        help="Number of arnold transforms to perform",
        default=64)
    args = parser.parse_args()

    if not args.output:
        args.show = True
    wm = Watermark(
        int(args.blocksize), float(args.threshold), args.show, int(args.key))
    if args.video:
        if args.mode == 'encrypt':
            wm.encrypt_vid(args.input, args.output)
        elif args.mode == 'decrypt':
            wm.decrypt_vid(args.input, args.output)
        elif args.mode == 'tamper':
            wm.tamper_vid(args.input, args.output, save=True)
        elif args.mode == 'score':
            wm.score(args.input, args.output)
    else:
        if args.mode == 'encrypt':
            wm.encrypt_img(args.input, args.output)
        elif args.mode == 'decrypt':
            wm.decrypt_img(args.input, args.output)
        elif args.mode == 'score':
            wm.score(args.input, args.output)
