__all__ = []

from lsst.utils import continueClass
from .fits import (Fits, ImageWriteOptions, ImageCompressionOptions, ImageScalingOptions,
                   compressionAlgorithmToString, scalingAlgorithmToString)


@continueClass  # noqa: F811
class Fits:
    def __enter__(self):
        return self

    def __exit__(self, cls, exc, traceback):
        self.closeFile()


@continueClass  # noqa: F811
class ImageWriteOptions:
    def __repr__(self):
        return f"{self.__class__.__name__}(compression={self.compression!r}, scaling={self.scaling!r})"


@continueClass  # noqa: F811
class ImageCompressionOptions:
    def __repr__(self):
        return (f"{self.__class__.__name__}(algorithm={compressionAlgorithmToString(self.algorithm)!r}, "
                f"tiles={self.tiles.tolist()!r}, quantizeLevel={self.quantizeLevel:f})")


@continueClass  # noqa: F811
class ImageScalingOptions:
    def __repr__(self):
        return (f"{self.__class__.__name__}(algorithm={scalingAlgorithmToString(self.algorithm)!r}, "
                f"bitpix={self.bitpix}, maskPlanes={self.maskPlanes}, seed={self.seed} "
                f"quantizeLevel={self.quantizeLevel}, quantizePad={self.quantizePad}, "
                f"fuzz={self.fuzz}, bscale={self.bscale}, bzero={self.bzero})")
