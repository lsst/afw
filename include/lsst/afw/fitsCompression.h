// -*- lsst-c++ -*-
#ifndef LSST_AFW_fitsCompression_h_INCLUDED
#define LSST_AFW_fitsCompression_h_INCLUDED

#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "ndarray.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Mask.h"


namespace lsst {
namespace afw {
namespace fits {

class Fits;

/// FITS compression algorithms.
///
/// See the standard for details.
///
/// These have a trailing slash to avoid conflict with CFITSIO's preprocessor
/// macros (the trailing slash is dropped in the Python bindings).  Not all
/// FITS compression algorithms are supported.
enum class CompressionAlgorithm {
    GZIP_1_,
    GZIP_2_,
    RICE_1_
};

/// FITS quantization algorithms.
///
/// See the standard for details.
///
/// These have a trailing slash to avoid conflict with CFITSIO's preprocessor
/// macros (the trailing slash is dropped in the Python bindings).
enum class DitherAlgorithm {
    NO_DITHER_,
    SUBTRACTIVE_DITHER_1_,
    SUBTRACTIVE_DITHER_2_,
};

/// Algorithms used to compute the scaling factor used in quantization.
enum class ScalingAlgorithm {
    RANGE,           ///< Scale to preserve dynamic range with bad pixels msasked out.
    STDEV_MASKED,    ///< Scale based on the standard deviation with bad pixels masked out.
    STDEV_CFITSIO,   ///< Let CFITSIO work out the scaling (per-tile; does not respect mask planes)
    MANUAL,          ///< Scale set manually.
};

/// Options controlling quantization for image compression with FITS.
struct QuantizationOptions {
    /// The method used to dither floating point values with random noise.
    DitherAlgorithm dither = DitherAlgorithm::NO_DITHER_;

    /// The algorithm used to determine the scaling for quantization.
    ScalingAlgorithm scaling = ScalingAlgorithm::STDEV_MASKED;

    /// Mask planes to ignore when doing statistics.
    ///
    /// This is used by all STDEV_ algorithms except STDEV_CFITSIO.
    std::vector<std::string> mask_planes = {"NO_DATA"};

    /// Target quantization level.
    ///
    /// Interpretation depends on the scaling algorithm.
    /// For RANGE, this is ignored.
    /// For STDEV_*, this is the number of quantization levels within the standard deviation.
    /// For MANUAL, this is just ZSCALE, the factor the compressed integer values should be scaled by to
    /// (approximately) recover the original floating-point values.
    float level = 0;

    /// Random seed used for dithering.
    ///
    /// Must be between 1 and 10000 (inclusive) to be set explicitly; 0 signals
    /// that a random seed should be generated from the current time, and
    /// negative sets a random seed based on the checksum of the first tile.
    int seed = 0;

    /// Whether this quantization configuration would make use of a Mask
    /// and the configured mask_planes.
    bool uses_mask() const {
        switch (scaling) {
            case ScalingAlgorithm::RANGE:
            case ScalingAlgorithm::STDEV_MASKED:
                return true;
            case ScalingAlgorithm::STDEV_CFITSIO:
            case ScalingAlgorithm::MANUAL:
                break;
        };
        return false;
    }
};

/// Options controlling image compression with FITS.
struct CompressionOptions {

    /// The compression algorithm to use.
    CompressionAlgorithm algorithm = CompressionAlgorithm::GZIP_2_;

    //@{
    /// Shape of a compression tile.
    ///
    /// Zeros are replaced by the length of the image in that dimension.
    std::size_t tile_width = 0;
    std::size_t tile_height = 1;
    //@}

    /// Options for quantizing a floating point image (i.e. lossy compression).
    std::optional<QuantizationOptions> quantization = std::nullopt;

    /// Whether this compression configuration would make use of a Mask
    /// and the configured mask_planes.
    bool uses_mask() const {
        return quantization.has_value() && quantization.value().uses_mask();
    }
};

}  // namespace fits
}  // namespace afw
}  // namespace lsst

#endif  // ifndef LSST_AFW_fitsCompression_h_INCLUDED
