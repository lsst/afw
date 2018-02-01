// -*- lsst-c++ -*-
/*
 * Capture the colour of an object
 */

#ifndef LSST_AFW_IMAGE_COLOR_H
#define LSST_AFW_IMAGE_COLOR_H

#include <cmath>
#include <limits>
#include "lsst/afw/image/Filter.h"

namespace lsst {
namespace afw {
namespace image {

/**
 * Describe the colour of a source
 *
 * We need a concept of colour more general than "g - r" in order to calculate e.g. atmospheric dispersion
 * or a source's PSF
 *
 * @note This is very much just a place holder until we work out what we need.  A full SED may be required,
 * in which case a constructor from an SED name might be appropriate, or a couple of colours, or ...
 */
class Color {
public:
    explicit Color(double g_r = std::numeric_limits<double>::quiet_NaN()) : _g_r(g_r) {}

    Color(Color const &) = default;
    Color(Color &&) = default;
    Color &operator=(Color const &) = default;
    Color &operator=(Color &&) = default;
    ~Color() = default;

    /// Whether the color is the special value that indicates that it is unspecified.
    bool isIndeterminate() const { return std::isnan(_g_r); }

    //@{
    /**
     *  Equality comparison for colors
     *
     *  Just a placeholder like everything else, but we explicitly let indeterminate colors compare
     *  as equal.
     *
     *  In the future, we'll probably want some way of doing fuzzy comparisons on colors, but then
     *  we'd have to define some kind of "color difference" matric, and it's not worthwhile doing
     *  that yet.
     */
    bool operator==(Color const& other) const {
        return (isIndeterminate() && other.isIndeterminate()) || other._g_r == _g_r;
    }
    bool operator!=(Color const& other) const { return !operator==(other); }
    //@}

    /** Return the effective wavelength for this object in the given filter
     */
    double getLambdaEff(Filter const&  ///< The filter in question
                        ) const {
        return 1000 * _g_r;
    }

private:
    double _g_r;
};
}
}
}  // lsst::afw::image

#endif
