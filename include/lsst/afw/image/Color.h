// -*- lsst-c++ -*-
/*
 * Capture the colour of an object
 */

#ifndef LSST_AFW_IMAGE_COLOR_H
#define LSST_AFW_IMAGE_COLOR_H

#include <cmath>
#include <limits>
#include <string>

namespace lsst {
namespace afw {
namespace image {

// TO DO: Change description here.
/**
 * Describe the colour of a source
 *
 * We need a concept of colour more general than "g - r" in order to calculate e.g. atmospheric dispersion
 * or a source's PSF
 *
 * @note This is very much just a place holder until we work out what we need.  A full SED may be required,
 * in which case a constructor from an SED name might be appropriate, or a couple of colours, or ...
 */
class Color final {
public:
    /// Default: indeterminate color
    Color() noexcept
        : _color_value(std::numeric_limits<double>::quiet_NaN()),
        _color_type(),
        _indeterminate(true) {}

    /// Fully-specified color: both a numeric value and its type string
    Color(double color_value, std::string const & color_type) noexcept
        : _color_value(color_value), _color_type(color_type), _indeterminate(false) {}

    Color(Color const &) = default;
    Color(Color &&) = default;
    Color &operator=(Color const &) = default;
    Color &operator=(Color &&) = default;
    ~Color() noexcept = default;

    /// Whether this Color was default‑constructed (i.e. has no value/type).
    bool isIndeterminate() const noexcept { return _indeterminate; }

    /// The numeric color value; only valid if !isIndeterminate().
    double getColorValue() const noexcept { return _color_value; }

    /// The color type string (e.g. "g-r"); only valid if !isIndeterminate().
    std::string const & getColorType() const noexcept { return _color_type; }

    /**
     * Equality comparison for colors.
     *
     * Indeterminate colors compare equal to each other; fully-specified
     * colors compare by both value and type.
     */
    bool operator==(Color const & other) const noexcept {
        if (_indeterminate && other._indeterminate) {
            return true;
        }
        if (_indeterminate != other._indeterminate) {
            return false;
        }
        return _color_value == other._color_value && _color_type == other._color_type;
    }
    bool operator!=(Color const & other) const noexcept { return !(*this == other); }

private:
    double      _color_value;
    std::string _color_type;
    bool        _indeterminate;
};

}  // namespace image
}  // namespace afw
}  // namespace lsst

#endif