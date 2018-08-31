// -*- lsst-c++ -*-

#include "fitsio.h"
extern "C" {
#include "fitsio2.h"
}

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Random.h"

#include "lsst/afw/fitsCompression.h"

extern float* fits_rand_value;     // Random numbers, defined in cfitsio
int const N_RESERVED_VALUES = 10;  // Number of reserved values for float --> bitpix=32 conversions (cfitsio)

namespace lsst {
namespace afw {
namespace fits {

ImageCompressionOptions::CompressionAlgorithm compressionAlgorithmFromString(std::string const& name) {
    if (name == "NONE") return ImageCompressionOptions::NONE;
    if (name == "GZIP") return ImageCompressionOptions::GZIP;
    if (name == "GZIP_SHUFFLE") return ImageCompressionOptions::GZIP_SHUFFLE;
    if (name == "RICE") return ImageCompressionOptions::RICE;
    if (name == "HCOMPRESS")
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, "HCOMPRESS is unsupported");
    if (name == "PLIO") return ImageCompressionOptions::PLIO;
    throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, "Unrecognised compression algorithm: " + name);
}

std::string compressionAlgorithmToString(ImageCompressionOptions::CompressionAlgorithm algorithm) {
    switch (algorithm) {
        case ImageCompressionOptions::NONE:
            return "NONE";
        case ImageCompressionOptions::GZIP:
            return "GZIP";
        case ImageCompressionOptions::GZIP_SHUFFLE:
            return "GZIP_SHUFFLE";
        case ImageCompressionOptions::RICE:
            return "RICE";
        case ImageCompressionOptions::PLIO:
            return "PLIO";
        default:
            std::ostringstream os;
            os << "Unrecognized compression algorithm: " << algorithm;
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, os.str());
    }
}

ImageCompressionOptions::CompressionAlgorithm compressionAlgorithmFromCfitsio(int cfitsio) {
    switch (cfitsio) {
        case 0:
            return ImageCompressionOptions::NONE;
        case RICE_1:
            return ImageCompressionOptions::RICE;
        case GZIP_1:
            return ImageCompressionOptions::GZIP;
        case GZIP_2:
            return ImageCompressionOptions::GZIP_SHUFFLE;
        case PLIO_1:
            return ImageCompressionOptions::PLIO;
        case HCOMPRESS_1:
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                              "Unsupported compression algorithm: HCOMPRESS_1");
        default:
            std::ostringstream os;
            os << "Unrecognized cfitsio compression: " << cfitsio;
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, os.str());
    }
}

int compressionAlgorithmToCfitsio(ImageCompressionOptions::CompressionAlgorithm algorithm) {
    switch (algorithm) {
        case ImageCompressionOptions::NONE:
            return 0;
        case ImageCompressionOptions::GZIP:
            return GZIP_1;
        case ImageCompressionOptions::GZIP_SHUFFLE:
            return GZIP_2;
        case ImageCompressionOptions::RICE:
            return RICE_1;
        case ImageCompressionOptions::PLIO:
            return PLIO_1;
        default:
            std::ostringstream os;
            os << "Unrecognized compression algorithm: " << algorithm;
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, os.str());
    }
}

ImageCompressionOptions::ImageCompressionOptions(ImageCompressionOptions::CompressionAlgorithm algorithm_,
                                                 int rows, float quantizeLevel_)
        : algorithm(algorithm_), tiles(ndarray::allocate(MAX_COMPRESS_DIM)), quantizeLevel(quantizeLevel_) {
    tiles[0] = 0;
    tiles[1] = rows;
    for (int ii = 2; ii < MAX_COMPRESS_DIM; ++ii) tiles[ii] = 1;
}

ImageScalingOptions::ScalingAlgorithm scalingAlgorithmFromString(std::string const& name) {
    if (name == "NONE") return ImageScalingOptions::NONE;
    if (name == "RANGE") return ImageScalingOptions::RANGE;
    if (name == "STDEV_POSITIVE") return ImageScalingOptions::STDEV_POSITIVE;
    if (name == "STDEV_NEGATIVE") return ImageScalingOptions::STDEV_NEGATIVE;
    if (name == "STDEV_BOTH") return ImageScalingOptions::STDEV_BOTH;
    throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, "Unrecognized scaling algorithm: " + name);
}

std::string scalingAlgorithmToString(ImageScalingOptions::ScalingAlgorithm algorithm) {
    switch (algorithm) {
        case ImageScalingOptions::NONE:
            return "NONE";
        case ImageScalingOptions::RANGE:
            return "RANGE";
        case ImageScalingOptions::STDEV_POSITIVE:
            return "STDEV_POSITIVE";
        case ImageScalingOptions::STDEV_NEGATIVE:
            return "STDEV_NEGATIVE";
        case ImageScalingOptions::STDEV_BOTH:
            return "STDEV_BOTH";
        default:
            std::ostringstream os;
            os << "Unrecognized scaling algorithm: " << algorithm;
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, os.str());
    }
}

ImageScalingOptions::ImageScalingOptions(ScalingAlgorithm algorithm_, int bitpix_,
                                         std::vector<std::string> const& maskPlanes_, int seed_,
                                         float quantizeLevel_, float quantizePad_, bool fuzz_, double bscale_,
                                         double bzero_)
        : algorithm(algorithm_),
          bitpix(bitpix_),
          fuzz(fuzz_),
          seed(std::abs(seed_ - 1) % (N_RANDOM - 1) +
               1),  // zero is bad (cfitsio uses non-deterministic method)
          maskPlanes(maskPlanes_),
          quantizeLevel(quantizeLevel_),
          quantizePad(quantizePad_),
          bscale(bscale_),
          bzero(bzero_) {}

namespace {

/// Calculate median and standard deviation for an image
template <typename T, int N>
std::pair<T, T> calculateMedianStdev(ndarray::Array<T const, N, N> const& image,
                                     ndarray::Array<bool, N, N> const& mask) {
    std::size_t num = 0;
    auto const& flatMask = ndarray::flatten<1>(mask);
    for (auto mm = flatMask.begin(); mm != flatMask.end(); ++mm) {
        if (!*mm) ++num;
    }
    ndarray::Array<T, 1, 1> array = ndarray::allocate(num);
    auto const& flatImage = ndarray::flatten<1>(image);
    auto mm = ndarray::flatten<1>(mask).begin();
    auto aa = array.begin();
    for (auto ii = flatImage.begin(); ii != flatImage.end(); ++ii, ++mm) {
        if (*mm) continue;
        *aa = *ii;
        ++aa;
    }

    // Quartiles; from https://stackoverflow.com/a/11965377/834250
    auto const q1 = num / 4;
    auto const q2 = num / 2;
    auto const q3 = q1 + q2;
    std::nth_element(array.begin(), array.begin() + q1, array.end());
    std::nth_element(array.begin() + q1 + 1, array.begin() + q2, array.end());
    std::nth_element(array.begin() + q2 + 1, array.begin() + q3, array.end());

    T const median = num % 2 ? array[num / 2] : 0.5 * (array[num / 2] + array[num / 2 - 1]);
    // No, we're not doing any interpolation for the lower and upper quartiles.
    // We're estimating the noise, so it doesn't need to be super precise.
    T const lq = array[q1];
    T const uq = array[q3];
    return std::make_pair(median, 0.741 * (uq - lq));
}

/// Calculate min and max for an image
template <typename T, int N>
std::pair<T, T> calculateMinMax(ndarray::Array<T const, N, N> const& image,
                                ndarray::Array<bool, N, N> const& mask) {
    T min = std::numeric_limits<T>::max(), max = std::numeric_limits<T>::min();
    auto mm = ndarray::flatten<1>(mask).begin();
    auto const& flatImage = ndarray::flatten<1>(image);
    for (auto ii = flatImage.begin(); ii != flatImage.end(); ++ii, ++mm) {
        if (*mm) continue;
        if (!std::isfinite(*ii)) continue;
        if (*ii > max) max = *ii;
        if (*ii < min) min = *ii;
    }
    return std::make_pair(min, max);
}

// Return range of values for target BITPIX
template <typename T>
double rangeForBitpix(int bitpix, bool cfitsioPadding) {
    if (bitpix == 0) {
        bitpix = detail::Bitpix<T>::value;
    }
    double range = std::pow(2.0, bitpix) - 1;  // Range of values for target BITPIX
    if (cfitsioPadding) {
        range -= N_RESERVED_VALUES;
    }
    return range;
}

}  // anonymous namespace

template <typename T, int N>
ImageScale ImageScalingOptions::determineFromRange(ndarray::Array<T const, N, N> const& image,
                                                   ndarray::Array<bool, N, N> const& mask, bool isUnsigned,
                                                   bool cfitsioPadding) const {
    auto minMax = calculateMinMax(image, mask);
    T const min = minMax.first;
    T const max = minMax.second;
    if (min == max) return ImageScale(bitpix, 1.0, min);
    double range = rangeForBitpix<T>(bitpix, cfitsioPadding);
    range -= 2;  // To allow for rounding and fuzz at either end
    double const bscale = static_cast<T>((max - min) / range);
    double bzero = static_cast<T>(isUnsigned ? min : min + 0.5 * range * bscale);
    if (cfitsioPadding) {
        bzero -= bscale * N_RESERVED_VALUES;
    }
    bzero -= bscale;  // Allow for rounding and fuzz on the low end
    return ImageScale(bitpix, bscale, bzero);
}

template <typename T, int N>
ImageScale ImageScalingOptions::determineFromStdev(ndarray::Array<T const, N, N> const& image,
                                                   ndarray::Array<bool, N, N> const& mask, bool isUnsigned,
                                                   bool cfitsioPadding) const {
    auto stats = calculateMedianStdev(image, mask);
    auto const median = stats.first, stdev = stats.second;
    double const bscale = static_cast<T>(stdev / quantizeLevel);

    /// Use min/max-based bzero if we can possibly fit everything in
    auto minMax = calculateMinMax(image, mask);
    T const min = minMax.first;
    T const max = minMax.second;
    double range = rangeForBitpix<T>(bitpix, cfitsioPadding);  // Range of values for target BITPIX
    double const numUnique = (max - min) / bscale;             // Number of unique values

    double imageVal;  // Value on image
    long diskVal;     // Corresponding quantized value
    if (numUnique < range) {
        imageVal = median;
        diskVal = cfitsioPadding ? 0.5 * N_RESERVED_VALUES : 0;
    } else {
        switch (algorithm) {
            case ImageScalingOptions::STDEV_POSITIVE:
                // Put (median - N sigma) at the lowest possible value: predominantly positive images
                imageVal = median - quantizePad * stdev;
                diskVal = (bitpix == 8) ? 0 : -(1L << (bitpix - 1));  // Lowest value: -2^(bitpix-1)
                if (cfitsioPadding) diskVal -= N_RESERVED_VALUES;
                break;
            case ImageScalingOptions::STDEV_NEGATIVE:
                // Put (median + N sigma) at the highest possible value: predominantly negative images
                imageVal = median + quantizePad * stdev;
                diskVal = (bitpix == 8) ? 255 : (1L << (bitpix - 1)) - 1;  // Lowest value: 2^(bitpix-1)-1
                break;
            case ImageScalingOptions::STDEV_BOTH:
                // Put median right in the middle: images with an equal abundance of positive and negative
                // values
                imageVal = median;
                diskVal = cfitsioPadding ? 0.5 * N_RESERVED_VALUES : 0;
                break;
            default:
                std::abort();  // Programming error: should never get here
        }
    }

    double bzero = static_cast<T>(imageVal - bscale * diskVal);
    return ImageScale(bitpix, bscale, bzero);
}

/// Scaling zero-point, set according to pixel type
template <typename T, class Enable = void>
struct Bzero {
    static double constexpr value = 0.0;
};

// uint64 version
// 'double' doesn't have sufficient bits to represent the appropriate BZERO,
// so let cfitsio handle it.
template <>
struct Bzero<std::uint64_t> {
    static double constexpr value = 0.0;
};

// Unsigned integer version
template <typename T>
struct Bzero<T, typename std::enable_if<std::numeric_limits<T>::is_integer &&
                                        !std::numeric_limits<T>::is_signed>::type> {
    static double constexpr value = (std::numeric_limits<T>::max() >> 1) + 1;
};

#ifndef DOXYGEN  // suppress a bogus Doxygen complaint about an documented symbol
template <typename T, int N>
ImageScale ImageScalingOptions::determine(ndarray::Array<T const, N, N> const& image,
                                          ndarray::Array<bool, N, N> const& mask) const {
    if (std::is_integral<T>::value && (bitpix != 0 || bitpix != detail::Bitpix<T>::value) &&
        algorithm != NONE) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                          "Image scaling not supported for integral types");
    }
    bool const isUnsigned = bitpix == 8 || (bitpix == 0 && detail::Bitpix<T>::value == 8);
    bool const cfitsioPadding = !std::numeric_limits<T>::is_integer && bitpix == 32;
    switch (algorithm) {
        case NONE:
            return ImageScale(detail::Bitpix<T>::value, 1.0, Bzero<T>::value);
        case RANGE:
            return determineFromRange(image, mask, isUnsigned, cfitsioPadding);
        case MANUAL:
            return ImageScale(bitpix, bscale, bzero);
        case ImageScalingOptions::STDEV_POSITIVE:
        case ImageScalingOptions::STDEV_NEGATIVE:
        case ImageScalingOptions::STDEV_BOTH:
            return determineFromStdev(image, mask, isUnsigned, cfitsioPadding);
        default:
            std::abort();  // should never get here
    }
}
#endif

namespace {

/// Random number generator used by cfitsio
///
/// We use the exact same random numbers that cfitsio generates, and
/// copy the implementation for the indexing.
class CfitsioRandom {
public:
    /// Ctor
    CfitsioRandom(int seed) : _seed(seed) {
        assert(seed != 0 && seed < N_RANDOM);
        fits_init_randoms();
        resetForTile(0);
    }

    /// Reset the indices for the i-th tile
    void resetForTile(int iTile) {
        _start = (iTile + _seed - 1) % N_RANDOM;
        reseed();
    }

    /// Get the next value
    float getNext() {
        float const value = fits_rand_value[_index];
        increment();
        return value;
    }

    /// Increment the indices
    void increment() {
        ++_index;
        if (_index == N_RANDOM) {
            ++_start;
            if (_start == N_RANDOM) {
                _start = 0;
            }
            reseed();
        }
    }

    /// Generate a flattened image of random numbers
    template <typename T>
    ndarray::Array<T, 1, 1> forImage(typename ndarray::Array<T const, 2, 2>::Index const& shape,
                                     ndarray::Array<long, 1> const& tiles) {
        std::size_t const xSize = shape[1], ySize = shape[0];
        ndarray::Array<T, 1, 1> out = ndarray::allocate(xSize * ySize);
        std::size_t const xTileSize = tiles[0] <= 0 ? xSize : tiles[0];
        std::size_t const yTileSize = tiles[1] < 0 ? ySize : (tiles[1] == 0 ? 1 : tiles[1]);
        int const xNumTiles = std::ceil(xSize / static_cast<float>(xTileSize));
        int const yNumTiles = std::ceil(ySize / static_cast<float>(yTileSize));
        for (int iTile = 0, yTile = 0; yTile < yNumTiles; ++yTile) {
            int const yStart = yTile * yTileSize;
            int const yStop = std::min(yStart + yTileSize, ySize);
            for (int xTile = 0; xTile < xNumTiles; ++xTile, ++iTile) {
                int const xStart = xTile * xTileSize;
                int const xStop = std::min(xStart + xTileSize, xSize);
                resetForTile(iTile);
                for (int y = yStart; y < yStop; ++y) {
                    auto iter = out.begin() + y * xSize + xStart;
                    for (int x = xStart; x < xStop; ++x, ++iter) {
                        *iter = static_cast<T>(getNext());
                    }
                }
            }
        }
        return out;
    }

private:
    /// Start the run of indices over with the new seed value
    void reseed() { _index = static_cast<int>(fits_rand_value[_start] * 500); }

    int _seed;   // Initial seed
    int _start;  // Starting index for tile; "iseed" in cfitsio
    int _index;  // Index of next value; "nextrand" in cfitsio
};

}  // anonymous namespace

template <typename T>
std::shared_ptr<detail::PixelArrayBase> ImageScale::toFits(ndarray::Array<T const, 2, 2> const& image,
                                                           bool forceNonfiniteRemoval, bool fuzz,
                                                           ndarray::Array<long, 1> const& tiles,
                                                           int seed) const {
    if (!std::numeric_limits<T>::is_integer && bitpix < 0) {
        if (bitpix != detail::Bitpix<T>::value) {
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                              "Floating-point images may not be converted to different floating-point types");
        }
        if (bscale != 1.0 || bzero != 0.0) {
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                              "Scaling may not be applied to floating-point images");
        }
    }

    if (bitpix < 0 || (bitpix == 0 && !std::numeric_limits<T>::is_integer) ||
        (bscale == 1.0 && bzero == 0.0 && !fuzz)) {
        if (!forceNonfiniteRemoval) {
            // Type conversion only
            return detail::makePixelArray(bitpix, ndarray::Array<T const, 1, 1>(ndarray::flatten<1>(image)));
        }
        if (!std::numeric_limits<T>::is_integer) {
            ndarray::Array<T, 1, 1> out = ndarray::allocate(image.getNumElements());
            auto outIter = out.begin();
            auto const& flatImage = ndarray::flatten<1>(image);
            for (auto inIter = flatImage.begin(); inIter != flatImage.end(); ++inIter, ++outIter) {
                *outIter = std::isfinite(*inIter) ? *inIter : std::numeric_limits<T>::max();
            }
            return detail::makePixelArray(bitpix, out);
        }
        // Fall through for explicit scaling
    }

    // Note: BITPIX=8 treated differently, since it uses unsigned values; the rest use signed */
    double min = bitpix == 8 ? 0 : -std::pow(2.0, bitpix - 1);
    double max = bitpix == 8 ? 255 : (std::pow(2.0, bitpix - 1) - 1.0);

    if (!std::numeric_limits<T>::is_integer && bitpix == 32) {
        // cfitsio saves space for N_RESERVED_VALUES=10 values at the low end
        min += N_RESERVED_VALUES;
    }

    double const scale = 1.0 / bscale;
    std::size_t const num = image.getNumElements();
    bool const applyFuzz = fuzz && !std::numeric_limits<T>::is_integer && bitpix > 0;
    ndarray::Array<double, 1, 1> out;
    if (applyFuzz) {
        if (tiles.isEmpty()) {
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                              "Tile sizes must be provided if fuzzing is desired");
        }
        out = CfitsioRandom(seed).forImage<double>(image.getShape(), tiles);
    } else {
        out = ndarray::allocate(num);
    }
    auto outIter = out.begin();
    auto const& flatImage = ndarray::flatten<1>(image);
    for (auto inIter = flatImage.begin(); inIter != flatImage.end(); ++inIter, ++outIter) {
        double value = (*inIter - bzero) * scale;
        if (!std::isfinite(value)) {
            // This choice of "max" for non-finite and overflow pixels is mainly cosmetic --- it has to be
            // something, and "min" would produce holes in the cores of bright stars.
            *outIter = blank;
            continue;
        }
        if (applyFuzz) {
            // Add random factor [0.0,1.0): adds a variance of 1/12,
            // but preserves the expectation value given the floor()
            value += *outIter;
        }
        *outIter = (value < min ? blank : (value > max ? blank : std::floor(value)));
    }
    return detail::makePixelArray(bitpix, out);
}

template <typename T>
ndarray::Array<T, 2, 2> ImageScale::fromFits(ndarray::Array<T, 2, 2> const& image) const {
    ndarray::Array<T, 2, 2> memory = ndarray::allocate(image.getShape());
    memory.deep() = bscale * image + bzero;
    return memory;
}

// Explicit instantiation
#define INSTANTIATE(TYPE)                                                                                    \
    template ImageScale ImageScalingOptions::determine<TYPE, 2>(                                             \
            ndarray::Array<TYPE const, 2, 2> const& image, ndarray::Array<bool, 2, 2> const& mask) const;    \
    template std::shared_ptr<detail::PixelArrayBase> ImageScale::toFits<TYPE>(                               \
            ndarray::Array<TYPE const, 2, 2> const&, bool, bool, ndarray::Array<long, 1> const&, int) const; \
    template ndarray::Array<TYPE, 2, 2> ImageScale::fromFits<TYPE>(ndarray::Array<TYPE, 2, 2> const&) const;

INSTANTIATE(std::uint8_t);
INSTANTIATE(std::uint16_t);
INSTANTIATE(std::int16_t);
INSTANTIATE(std::uint32_t);
INSTANTIATE(std::int32_t);
INSTANTIATE(std::uint64_t);
INSTANTIATE(std::int64_t);
INSTANTIATE(boost::float32_t);
INSTANTIATE(boost::float64_t);

}  // namespace fits
}  // namespace afw
}  // namespace lsst