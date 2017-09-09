from __future__ import absolute_import, division, print_function

__all__ = []

from lsst.utils import continueClass
from ._fits import (Fits, ImageWriteOptions, ImageCompressionOptions, ImageScalingOptions,
                    compressionAlgorithmToString, scalingAlgorithmToString)

@continueClass
class Fits:
    def __enter__(self):
        return self

    def __exit__(self, cls, exc, traceback):
        self.closeFile()


@continueClass
class ImageWriteOptions:
    def __repr__(self):
        return "%s(compression=%r, scaling=%r)" % (self.__class__.__name__, self.compression, self.scaling)


@continueClass
class ImageCompressionOptions:
    def __repr__(self):
        return ("%s(algorithm=%r, tiles=%r, quantizeLevel=%f" %
                (self.__class__.__name__, compressionAlgorithmToString(self.algorithm),
                 self.tiles.tolist(), self.quantizeLevel))


@continueClass
class ImageScalingOptions:
    def __repr__(self):
        return ("%s(algorithm=%r, bitpix=%d, maskPlanes=%s, seed=%d, quantizeLevel=%f, quantizePad=%f, "
                "fuzz=%s, bscale=%f, bzero=%f" %
                (self.__class__.__name__, scalingAlgorithmToString(self.algorithm), self.bitpix,
                 self.maskPlanes, self.seed, self.quantizeLevel, self.quantizePad, self.fuzz,
                 self.bscale, self.bzero))
