// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
/**
 * @brief Approximate values for a set of x,y vector<>s
 * @ingroup afw
 */
#include <limits>
#include <algorithm>
#include "boost/format.hpp"
#include "boost/shared_ptr.hpp"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Approximate.h"

namespace lsst {
namespace ex = pex::exceptions;
namespace afw {
namespace math {

/************************************************************************************************************/

template<typename PixelT>
class ApproximateChebyshev : public Approximate<PixelT> {
    template<typename T>
    friend PTR(Approximate<T>)
    makeApproximate(std::vector<double> const &x, std::vector<double> const &y,
                    image::MaskedImage<T> const& im,
                    ApproximateControl::Style const& style);
public:
    virtual ~ApproximateChebyshev();
    virtual double approximate(double const x, double const y) const;
private:
    ApproximateChebyshev(std::vector<double> const &x, std::vector<double> const &y,
                         image::MaskedImage<PixelT> const& im, ApproximateControl::Style const& style);
};

template<typename PixelT>
ApproximateChebyshev<PixelT>::ApproximateChebyshev(
        std::vector<double> const &x,            ///< the x-values of points
        std::vector<double> const &y,            ///< the y-values of points
        image::MaskedImage<PixelT> const& im,    ///< The values at (x, y)
        ApproximateControl::Style const& style   ///< desired approximation algorithm
                                                  ) : Approximate<PixelT>(x, y, style)
{
}

template<typename PixelT>
ApproximateChebyshev<PixelT>::~ApproximateChebyshev() {
}

template<typename PixelT>
double ApproximateChebyshev<PixelT>::approximate(double const x, double const y) const
{
    return 0.0;
}

/************************************************************************************************************/
/**
 * A factory function to make Approximate objects
 */
template<typename PixelT>
PTR(Approximate<PixelT>)
makeApproximate(std::vector<double> const &x,            ///< the x-values of points
                std::vector<double> const &y,            ///< the y-values of points
                image::MaskedImage<PixelT> const& im,    ///< The values at (x, y)
                ApproximateControl::Style const& style   /// < desired approximation algorithm
               )
{
    switch (style) {
      case ApproximateControl::CHEBYSHEV:
        return PTR(Approximate<PixelT>)(new ApproximateChebyshev<PixelT>(x, y, im, style));
      default:
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          str(boost::format("Unknown ApproximationStyle: %d")
                              % style));
    }
}
/*
 * Explicit instantiations
 *
 * \cond
 */
#define INSTANTIATE(PIXEL_T)                                          \
    template                                                          \
    PTR(Approximate<PIXEL_T>) makeApproximate(                        \
        std::vector<double> const &x, std::vector<double> const &y,   \
        image::MaskedImage<PIXEL_T> const& im,                        \
        ApproximateControl::Style const& style)

INSTANTIATE(float);
INSTANTIATE(int);

// \endcond

}}}
