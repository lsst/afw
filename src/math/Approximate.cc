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
#include <numeric>
#include "Eigen/Core"
#include "Eigen/LU"
#include "boost/format.hpp"
#include "boost/shared_ptr.hpp"
#include "lsst/utils/ieee.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Approximate.h"

namespace lsst {
namespace ex = pex::exceptions;
namespace afw {
namespace math {

/// \brief ctor
ApproximateControl::ApproximateControl(Style style, ///< Type of approximation
                                       int orderX,  ///< Order of approximation to use in x-direction
                                       int orderY   ///< Order of approximation to use in y-direction
                                       ) :
    _style(style), _orderX(orderX), _orderY(orderY < 0 ? orderX : orderY) {
    if (_orderX != _orderY) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          str(boost::format("X- and Y-orders must be equal (%d != %d) "
                                            "due to a limitation in math::Chebyshev1Function2")
                              % _orderX % _orderY));
    }
}

/************************************************************************************************************/

namespace {
/**
 * \brief Specialisation of Approximate in Chebyshev polynomials
 */
template<typename PixelT>
class ApproximateChebyshev : public Approximate<PixelT> {
    template<typename T>
    friend PTR(Approximate<T>)
    math::makeApproximate(std::vector<double> const &xVec, std::vector<double> const &yVec,
                          image::MaskedImage<T> const& im, geom::Box2I const& bbox,
                          ApproximateControl const& ctrl);
public:
    virtual ~ApproximateChebyshev();
private:
    math::Chebyshev1Function2<double> _poly;

    ApproximateChebyshev(std::vector<double> const &xVec, std::vector<double> const &yVec,
                         image::MaskedImage<PixelT> const& im, geom::Box2I const& bbox,
                         ApproximateControl const& ctrl);
    virtual PTR(image::Image<typename Approximate<PixelT>::OutPixelT>)
            doGetImage(int orderX, int orderY) const;
    virtual PTR(image::MaskedImage<typename Approximate<PixelT>::OutPixelT>)
            doGetMaskedImage(int orderX, int orderY) const;
};

/************************************************************************************************************/

namespace {
    // N.b. physically inlining these routines into ApproximateChebyshev
    // causes clang++ 3.1 problems; I suspect a clang bug (see http://llvm.org/bugs/show_bug.cgi?id=14162)
    inline void
    solveMatrix_Eigen(Eigen::MatrixXd &a,
                      Eigen::VectorXd &b,
                      Eigen::Map<Eigen::VectorXd> &c
                     ) {
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(a);
        c = lu.solve(b);
    }
}

/**
 * \brief Fit a grid of points to a afw::math::Chebyshev1Function2D
 */
template<typename PixelT>
ApproximateChebyshev<PixelT>::ApproximateChebyshev(
        std::vector<double> const &xVec,      ///< the x-values of points
        std::vector<double> const &yVec,      ///< the y-values of points
        image::MaskedImage<PixelT> const& im, ///< The values at (xVec, yVec)
        geom::Box2I const& bbox,              ///< Range where approximation should be valid
        ApproximateControl const& ctrl        ///< desired approximation algorithm
                                                  )
    : Approximate<PixelT>(xVec, yVec, bbox, ctrl),
      _poly(math::Chebyshev1Function2<double>(ctrl.getOrderX(), geom::Box2D(bbox)))    
{
#if !defined(NDEBUG)
    {
        std::vector<double> const& coeffs = _poly.getParameters();
        assert(std::accumulate(coeffs.begin(), coeffs.end(), 0.0) == 0.0); // i.e. coeffs is initialised to 0.0
    }
#endif
    int const nTerm = _poly.getNParameters();        // number of terms in polynomial
    int const nData = im.getWidth()*im.getHeight(); // number of data points
    /*
     * N.b. in the comments,
     *      i     runs over the 0..nTerm-1 coefficients
     *      alpha runs over the 0..nData-1 data points
     */

    /*
     * We need the value of the polynomials evaluated at every data point, so it's more
     * efficient to pre-calculate the values:  termCoeffs[i][alpha]
     */
    std::vector<std::vector<double> > termCoeffs(nTerm);

    for (int i = 0; i != nTerm; ++i) {
        termCoeffs[i].reserve(nData);
    }

    for (int iy = 0; iy != im.getHeight(); ++iy) {
        double const y = yVec[iy];

        for (int ix = 0; ix != im.getWidth(); ++ix) {
            double const x = xVec[ix];

            for (int i = 0; i != nTerm; ++i) {
                _poly.setParameter(i, 1.0);
                termCoeffs[i].push_back(_poly(x, y));
                _poly.setParameter(i, 0.0);
            }            
        }
    }
    // We'll solve A*c = b
    Eigen::MatrixXd A; A.setZero(nTerm, nTerm);    // We'll solve A*c = b
    Eigen::VectorXd b; b.setZero(nTerm);
    /*
     * Go through the data accumulating the values of the A and b matrix/vector
     */
    int alpha = 0;
    for (int iy = 0; iy != im.getHeight(); ++iy) {
        for (typename image::MaskedImage<PixelT>::const_x_iterator ptr = im.row_begin(iy),
                 end = im.row_end(iy); ptr != end; ++ptr, ++alpha) {
            double const val = ptr.image();
            double const ivar = 1/ptr.variance();
            if (!lsst::utils::isfinite(val + ivar)) {
                continue;
            }

            for (int i = 0; i != nTerm; ++i) {
                double const c_i = termCoeffs[i][alpha];
                double const tmp = c_i*ivar;

                b(i) += val*tmp;
                A(i, i) += c_i*tmp;
                for (int j = 0; j < i; ++j) {
                    double const c_j = termCoeffs[j][alpha];
                    A(i, j) += c_j*tmp;
                }
            }
        }
    }
    // We only filled out the lower triangular part of A
    for (int j = 0; j != nTerm; ++j) {
        for (int i = j + 1; i != nTerm; ++i) {
            A(j, i) = A(i, j);
        }
    }
    /*
     * OK, now all we ned do is solve that...
     */
    std::vector<double> cvec(nTerm);
    Eigen::Map<Eigen::VectorXd> c(&cvec[0], nTerm); // N.b. c shares memory with cvec

    solveMatrix_Eigen(A, b, c);
    
    _poly.setParameters(cvec);
}

/// \brief dtor
template<typename PixelT>
ApproximateChebyshev<PixelT>::~ApproximateChebyshev() {
}

/**
 * \brief worker function for getImage
 *
 * If orderX/orderY are specified the expansion will be truncated to that order
 *
 * \note As in the ApproximateControl ctor, the x- and y-orders must be equal
 */
template<typename PixelT>
PTR(image::Image<typename Approximate<PixelT>::OutPixelT>)
ApproximateChebyshev<PixelT>::doGetImage(int orderX,  ///< Order of approximation to use in x-direction
                                         int orderY   ///< Order of approximation to use in y-direction
                                        ) const
{
    if (orderX < 0) orderX = Approximate<PixelT>::_ctrl.getOrderX();
    if (orderY < 0) orderY = Approximate<PixelT>::_ctrl.getOrderY();

    math::Chebyshev1Function2<double> poly =
        (orderX == Approximate<PixelT>::_ctrl.getOrderX() &&
         orderY == Approximate<PixelT>::_ctrl.getOrderY()) ? _poly : _poly.truncate(orderX);
    
    typedef typename image::Image<typename Approximate<PixelT>::OutPixelT> ImageT;

    PTR(ImageT) im(new ImageT(Approximate<PixelT>::_bbox));
    for (int iy = 0; iy != im->getHeight(); ++iy) {
        double const y = iy;

        int ix = 0;
        for (typename ImageT::x_iterator ptr = im->row_begin(iy),
                 end = im->row_end(iy); ptr != end; ++ptr, ++ix) {
            double const x = ix;

            *ptr = poly(x, y);
        }
    }

    return im;
}
/**
 * \brief Return a MaskedImage
 *
 *
 * If orderX/orderY are specified the expansion will be truncated to that order
 *
 * \note As in the ApproximateControl ctor, the x- and y-orders must be equal
 */
template<typename PixelT>
PTR(image::MaskedImage<typename Approximate<PixelT>::OutPixelT>)
ApproximateChebyshev<PixelT>::doGetMaskedImage(
        int orderX,                     ///< Order of approximation to use in x-direction
        int orderY                      ///< Order of approximation to use in y-direction
                                              ) const
{
    typedef typename image::MaskedImage<typename Approximate<PixelT>::OutPixelT> MImageT;

    PTR(MImageT) mi(new MImageT(Approximate<PixelT>::_bbox));
    PTR(typename MImageT::Image) im = mi->getImage();

    for (int iy = 0; iy != im->getHeight(); ++iy) {
        double const y = iy;

        int ix = 0;
        for (typename MImageT::Image::x_iterator ptr = im->row_begin(iy),
                 end = im->row_end(iy); ptr != end; ++ptr, ++ix) {
            double const x = ix;

            *ptr = _poly(x, y);
        }
    }

    return mi;
}
}

/************************************************************************************************************/
/**
 * \brief A factory function to make Approximate objects
 */
template<typename PixelT>
PTR(Approximate<PixelT>)
makeApproximate(std::vector<double> const &x,            ///< the x-values of points
                std::vector<double> const &y,            ///< the y-values of points
                image::MaskedImage<PixelT> const& im,    ///< The values at (x, y)
                geom::Box2I const& bbox,                 ///< Range where approximation should be valid
                ApproximateControl const& ctrl           ///< desired approximation algorithm
               )
{
    switch (ctrl.getStyle()) {
      case ApproximateControl::CHEBYSHEV:
        return PTR(Approximate<PixelT>)(new ApproximateChebyshev<PixelT>(x, y, im, bbox, ctrl));
      default:
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          str(boost::format("Unknown ApproximationStyle: %d") % ctrl.getStyle()));
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
        geom::Box2I const& bbox,                                      \
        ApproximateControl const& ctrl)

INSTANTIATE(float);
//INSTANTIATE(int);

// \endcond

}}}
