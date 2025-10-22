# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

import astropy.io.fits
import numpy as np

import lsst.afw.image.testUtils  # for TestCase monkey-patch side-effects.  # noqa: F401
from lsst.afw.fits import DitherAlgorithm
from lsst.afw.image import (
    ExposureF,
    Image,
    ImageFitsReader,
    MaskedImage,
    MaskedImageFitsReader,
    makeMaskedImageFromArrays,
)
from lsst.geom import Box2I, Extent2I, Point2I
from lsst.utils.tests import TestCase, getTempFilePath


class ImageFitsTestCase(TestCase):
    """Tests for serializing Image, Mask, and MaskedImage to FITS."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(500)

    def sprinkle(self, masked_image: MaskedImage, value: float | int, count: int, mask_plane: str) -> None:
        """set a small number of randomly-selected pixels to the given value.

        Parameters
        ----------
        masked_image : `MaskedImage`
            Masked image to modify in place.  The image plane and mask plane
            are modified; the variance is not.
        value : `int` or `float`
            Value of the modified pixels.
        count : `int`
            Number of pixels to modify.
        mask_plane : `str`
            Name of a mask plane to add and set.
        """
        bit = masked_image.mask.addMaskPlane(mask_plane)
        if not count:
            return
        x = self.rng.integers(0, masked_image.width, count, dtype=int)
        y = self.rng.integers(0, masked_image.height, count, dtype=int)
        masked_image.image.array[y, x] = value
        masked_image.mask.array[y, x] = 1 << bit

    def set_nontrivial_xy0(self, image: MaskedImage) -> MaskedImage:
        """Set the origin of the given `Image` or `MaskedImage` to something
        other than (0, 0).
        """
        image.setXY0(Point2I(image.width // 3, image.height // 2))
        return image

    def make_int_image(
        self,
        dtype: np.typing.DTypeLike,
        noise_min: int,
        noise_max: int,
        *,
        hot_value: int = 0,
        hot_count: int = 50,
        cold_value: int = 0,
        cold_count: int = 50,
        shape: tuple[int, int] = (256, 225),
        variance: float | None = None,
    ) -> MaskedImage:
        """Make an integer test image.

        Parameters
        ----------
        dtype : `numpy.dtype`
            Numpy type object for the image's pixels.
        noise_min : `int`
            Minimum value (inclusive) for the discrete uniform noise that fills
            most of the image.
        noise_max : `int`
            Maximum value (inclusive) for the discrete uniform noise that fills
            most of the image.
        hot_value : `int`, optional
            Value for hot pixels to `sprinkle` into the image.
        hot_count : `int`, optional
            Number of hot pixels to `sprinkle` into the image.
        cold_value : `int`, optional
            Value for cold pixels to `sprinkle` into the image.
        cold_count : `int`, optional
            Number of cold pixels to `sprinkle` into the image.
        shape : `tuple`, optional
            Dimensions of the image (y, x).
        variance : `float`, optional
            Constant value for the variance plane.  Default is to use the
            actual variance of the added noise.

        Returns
        -------
        masked_image : `MaskedImage`
            Test masked image.  Sprinkled pixels are masked as "HOT" or "COLD".
        """
        array = self.rng.integers(noise_min, noise_max, shape, dtype, endpoint=True)
        masked_image = makeMaskedImageFromArrays(array)
        masked_image.variance.array = (
            variance if variance is not None else (noise_max - noise_min) ** 2 / 12.0
        )
        self.sprinkle(masked_image, hot_value, hot_count, mask_plane="HOT")
        self.sprinkle(masked_image, cold_value, cold_count, mask_plane="COLD")
        return self.set_nontrivial_xy0(masked_image)

    def make_float_image(
        self,
        dtype: np.typing.DTypeLike,
        noise_sigma: float = 1.0,
        noise_mean: float = 0.0,
        *,
        hot_value: float = 20000.0,
        hot_count: int = 50,
        cold_value: float = -20000.0,
        cold_count: int = 50,
        nan_count: int = 50,
        shape: tuple[int, int] = (256, 256),
    ) -> MaskedImage:
        """Make a floating-point test image.

        Parameters
        ----------
        dtype : `numpy.dtype`
            Numpy type object for the image's pixels.
        noise_sigma : `float`
            Standard deviation of the Gaussian noise distribution that fills
            most of the image.
        noise_mean : `int`
            Mean of the Gaussian noise distribution that fills most of the
            image.
        hot_value : `float`, optional
            Value for hot pixels to `sprinkle` into the image.
        hot_count : `float`, optional
            Number of hot pixels to `sprinkle` into the image.
        cold_value : `float`, optional
            Value for cold pixels to `sprinkle` into the image.
        cold_count : `float`, optional
            Number of cold pixels to `sprinkle` into the image.
        nan_count : `float`, optional
            Number of NaN pixels to `sprinkle` into the image.
        shape : `tuple`, optional
            Dimensions of the image (y, x).

        Returns
        -------
        masked_image : `MaskedImage`
            Test masked image.  Sprinkled pixels are masked as "HOT", "COLD",
            or `NaN`, and the variance plane is set to a constant
            ``nan_sigma**2``.
        """
        array = self.rng.normal(noise_mean, noise_sigma, shape).astype(dtype)
        masked_image = makeMaskedImageFromArrays(array)
        masked_image.variance.array = noise_sigma**2
        self.sprinkle(masked_image, hot_value, hot_count, mask_plane="HOT")
        self.sprinkle(masked_image, cold_value, cold_count, mask_plane="COLD")
        self.sprinkle(masked_image, np.nan, nan_count, mask_plane="NAN")
        return self.set_nontrivial_xy0(masked_image)

    @contextmanager
    def roundtrip_image_reader(
        self, image: Image, compression: Mapping[str, object] | None = None, original_fits: bool = False
    ) -> Iterator[tuple[ImageFitsReader, astropy.io.fits.HDUList]]:
        """Return a context manager that writes and reads a test image.

        Parameters
        ----------
        image : `Image`
            Image to write and then read.
        compression : `collections.abc.Mapping` or `None`, optional
            How to compress the image (`None` for no compression), as a
            mapping compatible with `Image.writeFitsWithOptions`.
        original_fits : `bool`, optional
            If `True`, open the astropy FITS object with
            ``disable_image_compression=True``, interpreting all compressed
            image HDUs as binary tables and leaving their headers unchanged.

        Returns
        -------
        context : `contextlib.AbstractContextManager
            A context manager that, when entered, returns:

            - reader (`ImageFitsReader`): a reader for the saved image;
            - fits (`astropy.io.fits.HDUList`): an Astropy FITS object.
        """
        with getTempFilePath(".fits") as filename:
            image.writeFitsWithOptions(filename, compression)
            with astropy.io.fits.open(filename, disable_image_compression=original_fits) as fits:
                yield ImageFitsReader(filename), fits

    @contextmanager
    def roundtrip_masked_image_reader(
        self,
        masked_image: MaskedImage,
        compression: Mapping[str, object] | None = None,
        original_fits: bool = False,
    ) -> Iterator[tuple[MaskedImageFitsReader, astropy.io.fits.HDUList]]:
        """Return a context manager that writes and reads a test masked image.

        Parameters
        ----------
        image : `MaskedImage`
            Masked image to write and then read.
        compression : `collections.abc.Mapping` or `None`, optional
            How to compress the image (`None` for no compression), as a
            mapping compatible with `MaskedImage.writeFitsWithOptions`.
        original_fits : `bool`, optional
            If `True`, open the astropy FITS object with
            ``disable_image_compression=True``, interpreting all compressed
            image HDUs as binary tables and leaving their headers unchanged.

        Returns
        -------
        context : `contextlib.AbstractContextManager
            A context manager that, when entered, returns:

            - reader (`MaskedImageFitsReader`): a reader for the saved image;
            - fits (`astropy.io.fits.HDUList`): an Astropy FITS object.
        """
        with getTempFilePath(".fits") as filename:
            masked_image.writeFitsWithOptions(filename, compression)
            with astropy.io.fits.open(filename, disable_image_compression=original_fits) as fits:
                yield MaskedImageFitsReader(filename), fits

    @contextmanager
    def check_roundtrip_image_invariants(
        self,
        image: Image,
        compression: Mapping[str, object] | None = None,
        safe_dtypes: Sequence[np.typing.DTypeLike] = (),
        *,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> Iterator[Image]:
        """Return a context manager that writes and reads a test image and then
        runs a battery of tests.

        Parameters
        ----------
        image : `Image`
            Image to write and then read.
        compression : `collections.abc.Mapping` or `None`, optional
            How to compress the image (`None` for no compression), as a
            mapping compatible with `Image.writeFitsWithOptions`.
        safe_dtypes : `~collections.abc.Sequence` [`numpy.dtype`], optional
            Other pixel types that the image can be safely read back in as
            without any loss of precision or range.
        rtol : `float`, optional
            Relative tolerance for floating-point comparisons between our
            reader and astropy.  Defaults to zero.
        atol : `float`, optional
            Absolute tolerance for floating-point comparisons between our
            reader and astropy.  Defaults to zero.

        Returns
        -------
        context : `contextlib.AbstractContextManager
            A context manager that returns the roundtripped `Image` when
            entered.

        Notes
        -----
        This method tests:

         - reading just the bounding box;
         - reading the pixel type;
         - reading a subimage;
         - consistency with reading with astropy (by comparing to the
           roundtripped image, not the original);
         - consistency with reading into other pixel types (``safe_dtypes``).

        It does not test that the roundtripped image matches the original (that
        is for calling code to do).
        """
        hdu = 0 if compression is None else 1
        with self.subTest(compression=compression):
            with self.roundtrip_image_reader(image, compression) as (reader, fits):
                self.assertEqual(reader.readDType(), image.array.dtype)
                self.assertEqual(reader.readBBox(), image.getBBox())
                roundtripped = reader.read()
                np.testing.assert_allclose(roundtripped.array, fits[hdu].data, atol=atol, rtol=rtol)
                subbox = Box2I(
                    image.getXY0() + Extent2I(image.width // 5, image.height // 6),
                    image.getXY0() + Extent2I(4 * image.width // 5, 5 * image.height // 6),
                )
                subrt = reader.read(bbox=subbox)
                self.assertEqual(subrt.getBBox(), subbox)
                self.assertImagesEqual(subrt, roundtripped[subbox])
                for dtype in [image.dtype, *safe_dtypes]:
                    np.testing.assert_array_equal(
                        reader.readArray(dtype=np.dtype(dtype)), roundtripped.array.astype(dtype)
                    )
                    for dtype in [image.dtype, *safe_dtypes]:
                        np.testing.assert_array_equal(
                            reader.readArray(dtype=np.dtype(dtype)), roundtripped.array.astype(dtype)
                        )
                yield roundtripped

    @contextmanager
    def check_roundtrip_masked_image_invariants(
        self,
        masked_image: MaskedImage,
        compression: Mapping[str, object] | None = None,
        safe_dtypes: Sequence[np.typing.DTypeLike] = (),
        *,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> Iterator[MaskedImage]:
        """Return a context manager that writes and reads a test masked image
        and then runs a battery of tests.

        Parameters
        ----------
        masked_image : `MaskedImage`
            Masked image to write and then read.
        compression : `collections.abc.Mapping` or `None`, optional
            How to compress the image (`None` for no compression), as a
            mapping compatible with `Image.writeFitsWithOptions`.
        safe_dtypes : `~collections.abc.Sequence` [`numpy.dtype`], optional
            Other image plane pixel types that the image can be safely read
            back in as without any loss of precision or range.
        rtol : `float`, optional
            Relative tolerance for floating-point comparisons between our
            reader and astropy.  Defaults to zero.
        atol : `float`, optional
            Absolute tolerance for floating-point comparisons between our
            reader and astropy.  Defaults to zero.

        Returns
        -------
        context : `contextlib.AbstractContextManager
            A context manager that returns the roundtripped `Image` when
            entered.

        Notes
        -----
        This method tests:

         - reading just the bounding box;
         - reading the pixel type;
         - reading a subimage;
         - consistency with reading with astropy (by comparing to the
           roundtripped image, not the original);
         - consistency with reading into other pixel types (``safe_dtypes``).

        It does not test that the roundtripped image matches the original (that
        is for calling code to do).
        """
        with self.subTest(compression=compression):
            with self.roundtrip_masked_image_reader(masked_image, compression) as (reader, fits):
                self.assertEqual(reader.readImageDType(), masked_image.image.array.dtype)
                self.assertEqual(reader.readMaskDType(), masked_image.mask.array.dtype)
                self.assertEqual(reader.readVarianceDType(), masked_image.variance.array.dtype)
                self.assertEqual(reader.readBBox(), masked_image.getBBox())
                roundtripped = reader.read()
                np.testing.assert_allclose(roundtripped.image.array, fits["IMAGE"].data, rtol=rtol, atol=atol)
                np.testing.assert_allclose(roundtripped.mask.array, fits["MASK"].data, rtol=rtol, atol=atol)
                np.testing.assert_allclose(
                    roundtripped.variance.array, fits["VARIANCE"].data, rtol=rtol, atol=atol
                )
                subbox = Box2I(
                    masked_image.getXY0() + Extent2I(masked_image.width // 5, masked_image.height // 6),
                    masked_image.getXY0()
                    + Extent2I(4 * masked_image.width // 5, 5 * masked_image.height // 6),
                )
                subrt = reader.read(bbox=subbox)
                self.assertEqual(subrt.getBBox(), subbox)
                self.assertMaskedImagesEqual(subrt, roundtripped[subbox])
                for dtype in [masked_image.image.dtype, *safe_dtypes]:
                    np.testing.assert_array_equal(
                        reader.readImageArray(dtype=np.dtype(dtype)), roundtripped.image.array.astype(dtype)
                    )
                    for dtype in [masked_image.image.dtype, *safe_dtypes]:
                        np.testing.assert_array_equal(
                            reader.readImageArray(dtype=np.dtype(dtype)),
                            roundtripped.image.array.astype(dtype),
                        )
                yield roundtripped

    def check_exact_roundtrip(
        self,
        masked_image: MaskedImage,
        compression: Mapping[str, object] | None = None,
        *,
        safe_dtypes: Sequence[np.typing.DTypeLike] = (),
    ) -> None:
        """Test that a masked image round-trips through serialization exactly.

        Parameters
        ----------
        masked_image : `MaskedImage`
            Masked image to write and then read.  Roundtripping just the image
            plane on its own is also tested.
        compression : `collections.abc.Mapping` or `None`, optional
            How to compress the image (`None` for no compression), as a
            mapping compatible with `MaskedImage.writeFitsWithOptions`.
        safe_dtypes : `~collections.abc.Sequence` [`numpy.dtype`], optional
            Other image plane pixel types that the image can be safely read
            back in as without any loss of precision or range.
        """
        with self.check_roundtrip_image_invariants(masked_image.image, compression, safe_dtypes) as image_rt:
            self.assertImagesEqual(image_rt, masked_image.image)
        with self.check_roundtrip_masked_image_invariants(
            masked_image, compression, safe_dtypes=safe_dtypes
        ) as masked_image_rt:
            self.assertMaskedImagesEqual(masked_image_rt, masked_image)

    def check_quantized_roundtrip(
        self,
        masked_image: MaskedImage,
        *,
        compression: Mapping[str, object],
        roundtrip_atol: float,
        safe_dtypes: Sequence[np.typing.DTypeLike] = (),
    ) -> None:
        """Test that a masked image round-trips through serialization with some
        loss of precision expected.

        Parameters
        ----------
        masked_image : `MaskedImage`
            Masked image to write and then read.  Roundtripping just the image
            plane on its own is also tested.
        compression : `collections.abc.Mapping`
            How to compress the image, as a mapping compatible with
            `MaskedImage.writeFitsWithOptions`.
        roundtrip_atol : `float`
            Absolute tolerance for comparisons between the input and output
            image.
        safe_dtypes : `~collections.abc.Sequence` [`numpy.dtype`], optional
            Other image plane pixel types that the image can be safely read
            back in as without any loss of precision or range.
        """
        with self.check_roundtrip_image_invariants(masked_image.image, compression, safe_dtypes) as image_rt:
            self.assertImagesAlmostEqual(image_rt, masked_image.image, atol=roundtrip_atol)
        with self.check_roundtrip_masked_image_invariants(
            masked_image, compression, safe_dtypes=safe_dtypes
        ) as masked_image_rt:
            self.assertMaskedImagesAlmostEqual(masked_image_rt, masked_image, atol=roundtrip_atol)

    def test_u16(self) -> None:
        """Test FITS serialization (including compression) for uint16."""
        masked_image = self.make_int_image(
            np.uint16, noise_min=256, noise_max=512, cold_value=1, hot_value=32750
        )
        safe_dtypes = (np.int32, np.uint64, np.float32, np.float64)
        self.check_exact_roundtrip(masked_image, safe_dtypes=safe_dtypes)
        self.check_exact_roundtrip(
            masked_image, {"image": {}, "mask": {}, "variance": {}}, safe_dtypes=safe_dtypes
        )
        self.check_exact_roundtrip(
            masked_image,
            {
                "image": {"algorithm": "GZIP_1"},
                "mask": {"algorithm": "GZIP_1"},
                "variance": {"algorithm": "GZIP_1"},
            },
            safe_dtypes=safe_dtypes,
        )
        self.check_exact_roundtrip(
            masked_image,
            {
                "image": {"algorithm": "GZIP_2"},
                "mask": {"algorithm": "GZIP_2"},
                "variance": {"algorithm": "GZIP_2"},
            },
            safe_dtypes=safe_dtypes,
        )
        self.check_exact_roundtrip(
            masked_image,
            {
                "image": {"algorithm": "RICE_1"},
                "mask": {"algorithm": "RICE_1"},
                "variance": {},
            },
            safe_dtypes=safe_dtypes,
        )

    def test_i32(self) -> None:
        """Test FITS serialization (including compression) for int32."""
        masked_image = self.make_int_image(
            np.int32, noise_min=-(1 << 21), noise_max=(1 << 22), cold_value=-(1 << 24), hot_value=(1 << 25)
        )
        safe_dtypes = (np.float64,)
        self.check_exact_roundtrip(masked_image, safe_dtypes=safe_dtypes)
        self.check_exact_roundtrip(
            masked_image, {"image": {}, "mask": {}, "variance": {}}, safe_dtypes=safe_dtypes
        )
        self.check_exact_roundtrip(
            masked_image,
            {
                "image": {"algorithm": "GZIP_1"},
                "mask": {"algorithm": "GZIP_1"},
                "variance": {"algorithm": "GZIP_1"},
            },
            safe_dtypes=safe_dtypes,
        )
        self.check_exact_roundtrip(
            masked_image,
            {
                "image": {"algorithm": "GZIP_2"},
                "mask": {"algorithm": "GZIP_2"},
                "variance": {"algorithm": "GZIP_2"},
            },
            safe_dtypes=safe_dtypes,
        )
        self.check_exact_roundtrip(
            masked_image,
            {
                "image": {"algorithm": "RICE_1"},
                "mask": {"algorithm": "RICE_1"},
                "variance": {},
            },
            safe_dtypes=safe_dtypes,
        )

    def test_u64(self) -> None:
        """Test FITS serialization for uint64."""
        masked_image = self.make_int_image(
            np.uint64,
            noise_min=256,
            noise_max=(1 << 42),
            cold_value=1,
            hot_value=(1 << 50),
            # We need to lie about the variance because the actual variance
            # of this uniform distribution is too big to be represented as
            # float32.
            variance=50.0,
        )
        self.check_exact_roundtrip(masked_image)
        # CFITSIO does not support compressing uint64, so neither do we.

    def test_f32_lossless(self) -> None:
        """Test uncompressed FITS serialization and lossless compression for
        float32.
        """
        masked_image = self.make_float_image(
            np.float32, noise_mean=100.0, noise_sigma=15.0, cold_value=-(1 << 24), hot_value=(1 << 25)
        )
        safe_dtypes = (np.float64,)
        self.check_exact_roundtrip(masked_image, compression=None, safe_dtypes=safe_dtypes)
        self.check_exact_roundtrip(
            masked_image, {"image": {}, "mask": {}, "variance": {}}, safe_dtypes=safe_dtypes
        )
        self.check_exact_roundtrip(
            masked_image,
            {
                "image": {"algorithm": "GZIP_1"},
                "mask": {"algorithm": "GZIP_1"},
                "variance": {"algorithm": "GZIP_1"},
            },
            safe_dtypes=safe_dtypes,
        )
        self.check_exact_roundtrip(
            masked_image,
            {
                "image": {"algorithm": "GZIP_2"},
                "mask": {"algorithm": "GZIP_2"},
                "variance": {"algorithm": "GZIP_2"},
            },
            safe_dtypes=safe_dtypes,
        )

    def test_f32_lossy_no_mask(self) -> None:
        """Test quantized compression of float32 with no bad pixels other than
        NaNs.
        """
        masked_image = self.make_float_image(
            # Don't add hot or cold pixels because we won't be using a mask to
            # keep them out of the stdev estimation.
            np.float32,
            noise_mean=100.0,
            noise_sigma=15.0,
            cold_count=0,
            hot_count=0,
        )
        lossy = {"algorithm": "RICE_1", "quantization": {"seed": 1}}
        lossless = {"algorithm": "GZIP_2"}
        for dither in DitherAlgorithm.__members__.keys():
            lossy["quantization"]["dither"] = dither
            lossy["quantization"]["level"] = 30.0
            lossy["quantization"]["scaling"] = "STDEV_CFITSIO"
            self.check_quantized_roundtrip(
                masked_image,
                compression={"image": lossy, "mask": lossless, "variance": lossy},
                roundtrip_atol=0.5,  # target is 30 steps per sigma, and sigma=15.
            )
            lossy["quantization"]["scaling"] = "MANUAL"
            lossy["quantization"]["level"] = 0.25
            self.check_quantized_roundtrip(
                masked_image,
                compression={"image": lossy, "mask": lossless, "variance": lossy},
                roundtrip_atol=0.25,
            )

    def test_f32_lossy_masked(self) -> None:
        """Test quantized compression of float 32 with hot and cold pixels."""
        masked_image = self.make_float_image(
            np.float32, noise_mean=100.0, noise_sigma=15.0, hot_value=1e5, cold_value=-1e4
        )
        lossy = {
            "algorithm": "RICE_1",
            "quantization": {"level": 30.0, "mask_planes": ["HOT", "COLD", "NAN"], "seed": 1},
        }
        lossless = {"algorithm": "GZIP_2"}
        for dither in DitherAlgorithm.__members__.keys():
            lossy["quantization"]["dither"] = dither
            lossy["quantization"]["scaling"] = "STDEV_MASKED"
            self.check_quantized_roundtrip(
                masked_image,
                compression={"image": lossy, "mask": lossless, "variance": lossy},
                roundtrip_atol=0.5,  # target is 30 steps per sigma, and sigma=15.
            )
            lossy["quantization"]["scaling"] = "RANGE"
            with self.subTest(dither=dither, scaling="RANGE"):
                self.check_quantized_roundtrip(
                    masked_image,
                    compression={"image": lossy, "mask": lossless, "variance": lossy},
                    roundtrip_atol=1e-7,  # since the dynamic range is tiny, this is actually barely lossy
                )

    def test_f64_lossless(self) -> None:
        """Test uncompressed FITS serialization and lossless compression for
        float36.
        """
        masked_image = self.make_float_image(
            np.float64, noise_mean=100.0, noise_sigma=15.0, cold_value=-(1 << 24), hot_value=(1 << 25)
        )
        self.check_exact_roundtrip(masked_image, compression=None)
        self.check_exact_roundtrip(masked_image, {"image": {}, "mask": {}, "variance": {}})
        self.check_exact_roundtrip(
            masked_image,
            {
                "image": {"algorithm": "GZIP_1"},
                "mask": {"algorithm": "GZIP_1"},
                "variance": {"algorithm": "GZIP_1"},
            },
        )
        self.check_exact_roundtrip(
            masked_image,
            {
                "image": {"algorithm": "GZIP_2"},
                "mask": {"algorithm": "GZIP_2"},
                "variance": {"algorithm": "GZIP_2"},
            },
        )

    def test_f64_lossy_no_mask(self) -> None:
        """Test quantized compression of float64 with no bad pixels other than
        NaNs.
        """
        masked_image = self.make_float_image(
            # Don't add hot or cold pixels because we won't be using a mask to
            # keep them out of the stdev estimation.
            np.float64,
            noise_mean=100.0,
            noise_sigma=15.0,
            cold_count=0,
            hot_count=0,
        )
        lossy = {"algorithm": "RICE_1", "quantization": {"seed": 1}}
        lossless = {"algorithm": "GZIP_2"}
        for dither in DitherAlgorithm.__members__.keys():
            lossy["quantization"]["dither"] = dither
            lossy["quantization"]["level"] = 30.0
            lossy["quantization"]["scaling"] = "STDEV_CFITSIO"
            self.check_quantized_roundtrip(
                masked_image,
                compression={"image": lossy, "mask": lossless, "variance": lossy},
                roundtrip_atol=0.5,  # target is 30 steps per sigma, and sigma=15.
            )
            lossy["quantization"]["scaling"] = "MANUAL"
            lossy["quantization"]["level"] = 0.25
            self.check_quantized_roundtrip(
                masked_image,
                compression={"image": lossy, "mask": lossless, "variance": lossy},
                roundtrip_atol=0.25,
            )

    def test_f64_lossy_masked(self) -> None:
        """Test quantized compression of float64 with hot and cold pixels."""
        masked_image = self.make_float_image(
            np.float64, noise_mean=100.0, noise_sigma=15.0, hot_value=1e5, cold_value=-1e4
        )
        lossy = {
            "algorithm": "RICE_1",
            "quantization": {"level": 30.0, "mask_planes": ["HOT", "COLD", "NAN"], "seed": 1},
        }
        lossless = {"algorithm": "GZIP_2"}
        for dither in DitherAlgorithm.__members__.keys():
            lossy["quantization"]["dither"] = dither
            lossy["quantization"]["scaling"] = "STDEV_MASKED"
            self.check_quantized_roundtrip(
                masked_image,
                compression={"image": lossy, "mask": lossless, "variance": lossy},
                roundtrip_atol=0.5,  # target is 30 steps per sigma, and sigma=15.
            )
            lossy["quantization"]["scaling"] = "RANGE"
            with self.subTest(dither=dither, scaling="RANGE"):
                self.check_quantized_roundtrip(
                    masked_image,
                    compression={"image": lossy, "mask": lossless, "variance": lossy},
                    roundtrip_atol=1e-7,  # since the dynamic range is tiny, this is actually barely lossy
                )

    def check_tile_shape(
        self, image: Image, compression: Mapping[str, object], ztile1: int, ztile2: int
    ) -> None:
        with self.roundtrip_image_reader(image, compression=compression, original_fits=True) as (_, fits):
            hdu = fits[1]
            self.assertEqual(hdu.header["ZTILE1"], ztile1)
            self.assertEqual(hdu.header["ZTILE2"], ztile2)

    def test_tile_shapes(self) -> None:
        """Test customizing the tile size in FITS compression."""
        masked_image = self.make_float_image(
            # Don't add hot or cold pixels because we won't be using a mask to
            # keep them out of the stdev estimation.
            np.float32,
            noise_mean=100.0,
            noise_sigma=15.0,
            cold_count=0,
            hot_count=0,
        )
        lossy = {
            "image": {
                "algorithm": "RICE_1",
                "quantization": {
                    "dither": "NO_DITHER",
                    "scaling": "STDEV_CFITSIO",
                    "level": 30.0,
                    "seed": 1,
                },
            }
        }
        self.check_tile_shape(masked_image.image, lossy, ztile1=masked_image.width, ztile2=1)
        lossy["image"]["tile_width"] = 1
        lossy["image"]["tile_height"] = 0
        self.check_tile_shape(masked_image.image, lossy, ztile1=1, ztile2=masked_image.height)
        lossy["image"]["tile_width"] = 25
        lossy["image"]["tile_height"] = 17
        self.check_tile_shape(masked_image.image, lossy, ztile1=25, ztile2=17)

    # I can't get CFITSIO to make the correction described by this test; if we
    # update the header key after asking it to write the image, it gets into
    # a weird state that prevents (at least) appending binary table HDUs.
    @unittest.expectedFailure
    def test_no_rice_one(self) -> None:
        """Test that we've corrected CFITSIO's non-standard writing of RICE_ONE
        instead of RICE_1 when the dither mode is SUBTRACTIVE_DITHER_2.

        RICE_ONE is an old pre-standard key that is already in enough images
        that it's reasonable for readers to accept it, but it is not correct
        for CFITSIO to be writing it.
        """
        masked_image = self.make_float_image(
            # Don't add hot or cold pixels because we won't be using a mask to
            # keep them out of the stdev estimation.
            np.float32,
            noise_mean=100.0,
            noise_sigma=15.0,
            cold_count=0,
            hot_count=0,
        )
        lossy = {
            "image": {
                "algorithm": "RICE_1",
                "quantization": {
                    "dither": "SUBTRACTIVE_DITHER_2",
                    "scaling": "STDEV_CFITSIO",
                    "level": 30.0,
                    "seed": 1,
                },
            },
            "mask": {
                "algorithm": "GZIP_2",
            },
            "variance": {
                "algorithm": "RICE_1",
                "quantization": {
                    "dither": "SUBTRACTIVE_DITHER_2",
                    "scaling": "STDEV_CFITSIO",
                    "level": 30.0,
                    "seed": 1,
                },
            },
        }
        with self.roundtrip_image_reader(masked_image.image, lossy, original_fits=True) as (_, fits):
            self.assertEqual(len(fits), 2)
            self.assertEqual(fits[1].header["ZCMPTYPE"], "RICE_1")
        with self.roundtrip_masked_image_reader(masked_image, lossy, original_fits=True) as (_, fits):
            self.assertEqual(len(fits), 4)
            self.assertEqual(fits[1].header["ZCMPTYPE"], "RICE_1")
            self.assertEqual(fits[2].header["ZCMPTYPE"], "GZIP_2")
            self.assertEqual(fits[3].header["ZCMPTYPE"], "RICE_1")

    def test_compressed_exposure(self) -> None:
        masked_image = self.make_float_image(
            np.float32, noise_mean=100.0, noise_sigma=15.0, hot_value=1e5, cold_value=-1e4
        )
        exposure = ExposureF(masked_image)
        lossy = {
            "algorithm": "RICE_1",
            "quantization": {
                "dither": "SUBTRACTIVE_DITHER_2",
                "scaling": "STDEV_MASKED",
                "mask_planes": ["HOT", "COLD", "NAN"],
                "level": 30.0,
                "seed": 1,
            },
        }
        lossless = {"algorithm": "GZIP_2"}
        with getTempFilePath(".fits") as filename:
            exposure.writeFitsWithOptions(
                filename,
                {"image": lossy, "mask": lossless, "variance": lossy},
            )
            roundtripped = ExposureF(filename)
        self.assertMaskedImagesAlmostEqual(masked_image, roundtripped.maskedImage, atol=0.25)


if __name__ == "__main__":
    unittest.main()
