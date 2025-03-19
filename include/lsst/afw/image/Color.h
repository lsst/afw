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

/**
 * Describe the colour of a source
 *
 * Inputs are color type (a string, for exemple 'g-r') and its color value (float).
 *
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
