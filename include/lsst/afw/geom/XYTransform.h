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
 * @brief Class representing an invertible 2D transform
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
 * @brief Virtual base class for 2D transforms
 */
class XYTransform : public daf::base::Citizen
{
public:
    typedef afw::geom::Point2D Point2D;
    typedef afw::geom::AffineTransform AffineTransform;

    explicit XYTransform();
    virtual ~XYTransform() { }

    /// returns a deep copy
    virtual PTR(XYTransform) clone() const = 0;

    /// returns a "deep inverse" in this sense that the forward+inverse transforms do not share state
    virtual PTR(XYTransform) invert() const;

    /**
     * @brief virtuals for forward and reverse transforms
     *
     * These routines are responsible for throwing exceptions if the 'point' arg 
     * is outside the domain of the transform.
     */
    virtual Point2D forwardTransform(Point2D const &point) const = 0;
    virtual Point2D reverseTransform(Point2D const &point) const = 0;
    
    /**
     * @brief linearized forward and reversed transforms
     *
     * These are virtual but not pure virtual; there is a default implementation which
     * calls forwardTransform() or reverseTransform() and takes finite differences with step 
     * size equal to one.
     *
     * The following should always be satisfied for an arbitrary Point2D p
     * (and analogously for the reverse transform)
     *    this->forwardTransform(p) == this->linearizeForwardTransform(p)(p);
     */
    virtual AffineTransform linearizeForwardTransform(Point2D const &point) const;
    virtual AffineTransform linearizeReverseTransform(Point2D const &point) const;
};


/**
 * @brief A trivial XYTransform satisfying f(x)=x.
 */
class IdentityXYTransform : public XYTransform
{
public:
    IdentityXYTransform();
    
    virtual PTR(XYTransform) clone() const;
    virtual Point2D forwardTransform(Point2D const &point) const;
    virtual Point2D reverseTransform(Point2D const &point) const;
    virtual AffineTransform linearizeForwardTransform(Point2D const &point) const;
    virtual AffineTransform linearizeReverseTransform(Point2D const &point) const;
};


/**
 * @brief Wrap an XYTransform, swapping forward and reverse transforms.
 */
class InvertedXYTransform : public XYTransform
{
public:
    InvertedXYTransform(CONST_PTR(XYTransform) base);

    virtual PTR(XYTransform) clone() const;
    /** @brief Return the wrapped XYTransform */
    virtual PTR(XYTransform) invert() const;
    virtual Point2D forwardTransform(Point2D const &point) const;
    virtual Point2D reverseTransform(Point2D const &point) const;
    virtual AffineTransform linearizeForwardTransform(Point2D const &point) const;
    virtual AffineTransform linearizeReverseTransform(Point2D const &point) const;

protected:    
    CONST_PTR(XYTransform) _base;
};


/**
 * @brief Wrap a sequence of multiple XYTransforms
 *
 * forwardTransform executes transformList[i].forwardTransform in order 0, 1, 2..., e.g.
 *  
 * MultiXYTransform.forwardTransform(p) =
 *   transformList[n].forwardTransform(...(transformList[1].forwardTransform(transformList[0].forwardTransform(p))...)
 */
class MultiXYTransform : public XYTransform
{
public:
    typedef std::vector<CONST_PTR(XYTransform)> TransformList;
    MultiXYTransform(TransformList const &transformList);
    virtual PTR(XYTransform) clone() const;
    virtual Point2D forwardTransform(Point2D const &point) const;
    virtual Point2D reverseTransform(Point2D const &point) const;
    virtual AffineTransform linearizeForwardTransform(Point2D const &point) const;
    virtual AffineTransform linearizeReverseTransform(Point2D const &point) const;
    TransformList getTransformList() const { return _transformList; }
private:
    TransformList _transformList;
};

/**
 * @brief Wrap an AffineTransform
 *
 */
class AffineXYTransform : public XYTransform
{
public:
    AffineXYTransform(AffineTransform const &affineTransform);

    virtual PTR(XYTransform) clone() const;
    virtual Point2D forwardTransform(Point2D const &position) const;
    virtual Point2D reverseTransform(Point2D const &position) const;
    virtual AffineTransform linearizeForwardTransform(Point2D const &position) const;
    virtual AffineTransform linearizeReverseTransform(Point2D const &position) const;
    /// get underlying forward AffineTransform
    AffineTransform getForwardTransform() const;
    /// get underlying reverse AffineTransform
    AffineTransform getReverseTransform() const;

protected:    
    AffineTransform _forwardAffineTransform;
    AffineTransform _reverseAffineTransform;
};


/**
 * @brief A purely radial polynomial distortion, up to 6th order.
 *
 * forwardTransform(pt) = pt * scale
 * where:
 * - scale = (coeffs[1] r + coeffs[2] r^2 + ...) / r
 * - r = magnitude of pt
 *
 * @warning reverseTransform will fail if the polynomial is too far from linear (ticket #3152)
 *
 * @throw lsst::pex::exceptions::InvalidParameterError if coeffs.size() > 0 and any of
 * the following are true: coeffs.size() == 1, coeffs[0] != 0 or coeffs[1] == 0
 */
class RadialXYTransform : public XYTransform
{
public:
    RadialXYTransform(
        std::vector<double> const &coeffs   ///< radial polynomial coefficients;
            ///< if size == 0 then gives the identity transformation;
            ///< otherwise must satisfy: size > 1, coeffs[0] == 0, and coeffs[1] != 0
    );

    virtual PTR(XYTransform) clone() const;
    virtual PTR(XYTransform) invert() const;
    virtual Point2D forwardTransform(Point2D const &point) const;
    virtual Point2D reverseTransform(Point2D const &point) const;
    virtual AffineTransform linearizeForwardTransform(Point2D const &point) const;
    virtual AffineTransform linearizeReverseTransform(Point2D const &point) const;
    std::vector<double> getCoeffs() const { return _coeffs; }

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
};


}}}

#endif
