// -*- lsst-c++ -*-

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
 * \file
 * @brief Class representing an invertible transform of a pixelized image 
 */

#ifndef LSST_AFW_GEOM_XYTRANSFORM_H
#define LSST_AFW_GEOM_XYTRANSFORM_H

#include <string>
#include "boost/shared_ptr.hpp"
#include "lsst/pex/exceptions.h"
#include "lsst/daf/base.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/geom/ellipses.h"

namespace lsst {
namespace afw {
namespace geom {


/**
 * @brief XYTransform: virtual base class which represents a "pixel domain to pixel domain" transform
 *
 * An example would be a camera distortion.
 * By comparison, class Wcs represents a "pixel domain to celestial" transform.
 *
 * We allow XYTransforms to operate either in the pixel coordinate system of an individual
 * detector, or in the global focal plane coordinate system (with units mm rather than pixel
 * counts).  The flag XYTransform::_inFpCoordinateSystem distinguishes these two cases, so that
 * we can throw an exception if the transform is applied in the wrong coordinate system.  This
 * way of keeping track of the coordinate system is sort of clunky and will be improved later.
 */
class XYTransform : public daf::base::Citizen
{
public:
    typedef afw::geom::Point2D Point2D;
    typedef afw::geom::ellipses::Quadrupole Quadrupole;
    typedef afw::geom::AffineTransform AffineTransform;

    XYTransform(bool inFpCoordinateSystem);
    virtual ~XYTransform() { }

    /// returns a deep copy
    virtual PTR(XYTransform) clone() const = 0;

    /// returns a "deep inverse" in this sense that the forward+inverse transforms do not share state
    virtual PTR(XYTransform) invert() const;

    /**
     * @brief virtuals for forward and reverse transforms
     *
     * Both the pixel argument and the return value of these routines:
     *    - are in pixel units
     *    - have XY0 offsets included (i.e. caller may need to add XY0 to pixel 
     *         and subtract XY0 from return value if necessary)
     *
     * These routines are responsible for throwing exceptions if the 'pixel' arg 
     * is outside the domain of the transform.
     */
    virtual Point2D forwardTransform(Point2D const &pixel) const = 0;
    virtual Point2D reverseTransform(Point2D const &pixel) const = 0;
    
    /**
     * @brief linearized forward and reversed transforms
     *
     * These guys are virtual but not pure virtual; there is a default implementation which
     * calls forwardTransform() or reverseTransform() and takes finite differences with step 
     * size equal to one pixel.
     *
     * The following should always be satisfied for an arbitrary Point2D p
     * (and analogously for the reverse transform)
     *    this->forwardTransform(p) == this->linearizeForwardTransform(p)(p);
     */
    virtual AffineTransform linearizeForwardTransform(Point2D const &pixel) const;
    virtual AffineTransform linearizeReverseTransform(Point2D const &pixel) const;

    /// apply distortion to an (infinitesimal) quadrupole
    Quadrupole forwardTransform(Point2D const &pixel, Quadrupole const &q) const;
    Quadrupole reverseTransform(Point2D const &pixel, Quadrupole const &q) const;

    bool inFpCoordinateSystem() const { return _inFpCoordinateSystem; }

protected:
    bool _inFpCoordinateSystem;
};


/**
 * @brief IdentityXYTransform: Represents a trivial XYTransform satisfying f(x)=x.
 */
class IdentityXYTransform : public XYTransform
{
public:
    IdentityXYTransform(bool inFpCoordinateSystem);
    virtual ~IdentityXYTransform() { }
    
    virtual PTR(XYTransform) clone() const;
    virtual Point2D forwardTransform(Point2D const &pixel) const;
    virtual Point2D reverseTransform(Point2D const &pixel) const;
    virtual AffineTransform linearizeForwardTransform(Point2D const &pixel) const;
    virtual AffineTransform linearizeReverseTransform(Point2D const &pixel) const;
};


/**
 * @brief This class supplies a default ->invert() method which works for any XYTransform
 * (but can be overridden if something more efficient exists)
 */
class InvertedXYTransform : public XYTransform
{
public:
    InvertedXYTransform(CONST_PTR(XYTransform) base);
    virtual ~InvertedXYTransform() { }

    virtual PTR(XYTransform) clone() const;
    virtual PTR(XYTransform) invert() const;
    virtual Point2D forwardTransform(Point2D const &pixel) const;
    virtual Point2D reverseTransform(Point2D const &pixel) const;
    virtual AffineTransform linearizeForwardTransform(Point2D const &pixel) const;
    virtual AffineTransform linearizeReverseTransform(Point2D const &pixel) const;

protected:    
    CONST_PTR(XYTransform) _base;
};


/**
 * @brief This class wraps an AffineTransform to work like an XYTransform
 *
 */
class AffineXYTransform : public XYTransform
{
public:
    AffineXYTransform(AffineTransform const &affineTransform);
    virtual ~AffineXYTransform() { }

    virtual PTR(XYTransform) clone() const;
    virtual Point2D forwardTransform(Point2D const &position) const;
    virtual Point2D reverseTransform(Point2D const &position) const;
    virtual AffineTransform linearizeForwardTransform(Point2D const &position) const;
    virtual AffineTransform linearizeReverseTransform(Point2D const &position) const;

protected:    
    AffineTransform _forwardAffineTransform;
    AffineTransform _reverseAffineTransform;
};


/**
 * @brief RadialXYTransform: represents a purely radial polynomial distortion, up to 6th order.
 *
 * Note: this transform is always in the focal plane coordinate system but can be
 * combined with DetectorXYTransform below to get the distortion for an individual detector.
 */
class RadialXYTransform : public XYTransform
{
public:
    RadialXYTransform(std::vector<double> const &coeffs, bool coefficientsDistort=true);
    virtual ~RadialXYTransform() { }

    virtual PTR(XYTransform) clone() const;
    virtual PTR(XYTransform) invert() const;
    virtual Point2D forwardTransform(Point2D const &pixel) const;
    virtual Point2D reverseTransform(Point2D const &pixel) const;
    virtual AffineTransform linearizeForwardTransform(Point2D const &pixel) const;
    virtual AffineTransform linearizeReverseTransform(Point2D const &pixel) const;

    /**
     * @brief These static member functions operate on polynomials represented by vector<double>.
     *
     * They are intended mainly as helpers for the virtual member functions above, but are declared
     * public since there are also some unit tests which call them.
     */
    static std::vector<double>  polyInvert(std::vector<double> const &coeffs);
    static double               polyEval(std::vector<double> const &coeffs, double x);
    static Point2D              polyEval(std::vector<double> const &coeffs, Point2D const &p);
    static double               polyEvalDeriv(std::vector<double> const &coeffs, double x);

    static AffineTransform      polyEvalJacobian(std::vector<double> const &coeffs, 
                                                 Point2D const &p);

    static double               polyEvalInverse(std::vector<double> const &coeffs, 
                                                std::vector<double> const &icoeffs, double x);

    static Point2D              polyEvalInverse(std::vector<double> const &coeffs, 
                                                std::vector<double> const &icoeffs, 
                                                Point2D const &p);

    static AffineTransform      polyEvalInverseJacobian(std::vector<double> const &coeffs, 
                                                        std::vector<double> const &icoeffs, 
                                                        Point2D const &p);

    static AffineTransform      makeAffineTransform(double x, double y, double f, double g);

protected:
    std::vector<double> _coeffs;
    std::vector<double> _icoeffs;
    bool _coefficientsDistort;
};


}}}

#endif
