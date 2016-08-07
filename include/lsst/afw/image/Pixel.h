/*
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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

#if !defined(LSST_AFW_IMAGE_PIXEL_H)
#define LSST_AFW_IMAGE_PIXEL_H

#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>

namespace lsst { namespace afw { namespace image { namespace pixel {

template <typename, typename, typename, typename, typename> class BinaryExpr;

template <typename> struct exprTraits;

template <typename> struct bitwise_or;
template <typename> struct variance_divides;
template <typename> struct variance_multiplies;
template <typename> struct variance_plus;

/************************************************************************************************************/
/**
 * \file
 * \brief Classes to provide utility functions for a "Pixel" to get at image/mask/variance operators
 *
 * These classes allow us to manipulate the tuples returned by MaskedImage iterators/locators as if they were
 * POD.  This provides convenient syntactic sugar, but it also permits us to write generic algorithms to
 * manipulate MaskedImages as well as Images
 *
 * We need SinglePixel as well as Pixel as the latter is just a reference to a pixel in an image, and we need
 * to be able to build temporary values too
 *
 * We use C++ template expressions to manipulate Pixel and SinglePixel; this permits us to avoid making
 * SinglePixel inherit from Pixel (or vice versa) and allows the compiler to do a much better job of
 * optimising mixed-mode expressions --- basically, it no longer needs to create SinglePixels as Pixels that
 * have their own storage but use the reference members of Pixel to get work done.  There may be better ways,
 * but this way works, and g++ 4.0.1 (and maybe other compilers/versions) failed to optimise the previous
 * solution very well.
 */

/// A single %pixel of the same type as a MaskedImage
template<typename _ImagePixelT, typename _MaskPixelT, typename _VariancePixelT=double>
class SinglePixel : public detail::MaskedImagePixel_tag {
public:
    template<typename, typename, typename> friend class Pixel;
    template<typename T> friend class PixelTypeTraits;

    typedef _ImagePixelT ImagePixelT;
    typedef _MaskPixelT MaskPixelT;
    typedef _VariancePixelT VariancePixelT;

    SinglePixel(double const image, int mask=0, double const variance=0) :
        _image(image), _mask(mask), _variance(variance) {}
    SinglePixel(int const image, int mask=0, double const variance=0) :
        _image(image), _mask(mask), _variance(variance) {}

    template<typename rhsExpr>
    SinglePixel(rhsExpr const& rhs) : _image(rhs.image()), _mask(rhs.mask()), _variance(rhs.variance()) {}

    ImagePixelT image() const { return _image; }
    MaskPixelT mask() const { return _mask; }
    VariancePixelT variance() const { return _variance; }

private:
    /// Default Ctor
    ///
    /// Can be called by PixelTypeTraits<SinglePixel>::padValue()
    SinglePixel() :
        _image(std::numeric_limits<_ImagePixelT>::has_quiet_NaN ?
               std::numeric_limits<_ImagePixelT>::quiet_NaN() : 0),
        _mask(0),
        _variance(std::numeric_limits<_VariancePixelT>::has_quiet_NaN ?
                  std::numeric_limits<_VariancePixelT>::quiet_NaN() : 0)
        {}

    ImagePixelT _image;
    MaskPixelT _mask;
    VariancePixelT _variance;
};

/// Pixel type traits
template<typename PixelT>
struct PixelTypeTraits
{
    /// The quantity to use when a pixel value is undefined
    static inline const PixelT padValue() {
        return
            std::numeric_limits<PixelT>::has_quiet_NaN ?
            std::numeric_limits<PixelT>::quiet_NaN()
            : 0;
    }
};

/// Specialization for a %pixel of a MaskedImage
template<typename _ImagePixelT, typename _MaskPixelT, typename _VariancePixelT>
struct PixelTypeTraits<SinglePixel<_ImagePixelT, _MaskPixelT, _VariancePixelT> >
{
    typedef SinglePixel<_ImagePixelT, _MaskPixelT, _VariancePixelT> PixelT;

    /// The quantity to use when a pixel value is undefined
    static inline const PixelT padValue() {
        return PixelT();
    }
};

/// Return a SinglePixel
///
/// This function is useful as function overloading will choose the correct return type
/// (cf. std::make_pair()
template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT> makeSinglePixel(ImagePixelT x, MaskPixelT m, VariancePixelT v) {
    return SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT>(x, m, v);
}

/// A %pixel of a MaskedImage
template<typename _ImagePixelT, typename _MaskPixelT, typename _VariancePixelT=double>
class Pixel : public detail::MaskedImagePixel_tag {
public:
    typedef _ImagePixelT ImagePixelT;
    typedef _MaskPixelT MaskPixelT;
    typedef _VariancePixelT VariancePixelT;

    /// Construct a Pixel from references to its image/mask/variance components
#if 0
    Pixel(ImagePixelT& image, MaskPixelT& mask, VariancePixelT& variance) :
        _image(image), _mask(mask), _variance(variance) {}
#else
    //
    // This constructor casts away const.  This should be fixed by making const Pixels.
    //
    Pixel(ImagePixelT const& image, MaskPixelT const& mask=0x0, VariancePixelT const& variance=0) :
        _image(const_cast<ImagePixelT&>(image)),
        _mask(const_cast<MaskPixelT&>(mask)),
        _variance(const_cast<VariancePixelT&>(variance)) {
    }
#endif

    Pixel(SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT>& rhs) :
        _image(rhs._image), _mask(rhs._mask), _variance(rhs._variance) {}

    Pixel operator=(Pixel const& rhs) { // the following template won't stop the compiler trying to generate operator=
        _variance = rhs.variance();     // evaluate before we update image()
        _image = rhs.image();
        _mask = rhs.mask();

        return *this;
    }
    /// Assign a Pixel by evaluating an expression
    ///
    /// We use C++ template expressions to build a compile-time parse tree to evaluate
    /// Pixel expressions;  this is where we evaluate the rhs and set the Pixel's values
    template<typename rhsExpr>
    Pixel operator=(rhsExpr const& rhs) {
        _variance = rhs.variance();     // evaluate before we update image()
        _image = rhs.image();
        _mask = rhs.mask();

        return *this;
    }

    /// set the image part of a Pixel to rhs_image (the mask and variance are set to 0)
    Pixel operator=(double const& rhs_image) {
        _image = rhs_image;
        _mask = 0;
        _variance = 0;

        return *this;
    }

    /// set the image part of a Pixel to rhs_image (the mask and variance are set to 0)
    Pixel operator=(int const& rhs_image) {
        _image = rhs_image;
        _mask = 0;
        _variance = 0;

        return *this;
    }
    /// Return the %image part of a Pixel
    ImagePixelT image() const { return _image; }
    /// Return the mask part of a Pixel
    MaskPixelT mask() const { return _mask; }
    /// Return the variance part of a Pixel
    VariancePixelT variance() const { return _variance; }
    //
    // Logical operators.  We don't need to construct BinaryExpr for them
    // as efficiency isn't a concern.
    //
    /// Return true iff two pixels are equal (in all three of image, mask, and variance)
    template<typename T1>
    friend bool operator==(Pixel const& lhs, T1 const& rhs) {
        return lhs.image() == rhs.image() && lhs.mask() == rhs.mask() && lhs.variance() == rhs.variance();
    }

    /// Return true iff two pixels are unequal (in at least one of image, mask, and variance)
    template<typename T1>
    friend bool operator!=(Pixel const& lhs, T1 const& rhs) {
        return !(lhs == rhs);
    }

    //
    // Provide friend versions of the op= operators to permit argument promotion on their first arguments
    //
    /// Evaluate e1 += e2, and return e1
    template<typename ExprT>
    friend Pixel operator+=(Pixel const& e1, ExprT const& e2) {
        Pixel tmp(e1);                  // n.b. shares storage with e1 but gets around "const" (which is required)
        tmp = BinaryExpr<Pixel, ExprT,
            std::plus<ImagePixelT>, bitwise_or<MaskPixelT>, variance_plus<VariancePixelT> >(tmp, e2);
        return tmp;
    }

    /// Evaluate e1 -= e2, and return e1
    template<typename ExprT>
    friend Pixel operator-=(Pixel const& e1, ExprT const& e2) {
        Pixel tmp(e1);                  // n.b. shares storage with e1 but gets around "const" (which is required)
        tmp = BinaryExpr<Pixel, ExprT,
            std::minus<ImagePixelT>, bitwise_or<MaskPixelT>, variance_plus<VariancePixelT> >(tmp, e2);
        return tmp;
    }

    /// Evaluate e1 *= e2, and return e1
    template<typename ExprT>
    friend Pixel operator*=(Pixel const& e1, ExprT const& e2) {
        Pixel tmp(e1);                  // n.b. shares storage with e1 but gets around "const" (which is required)
        tmp = BinaryExpr<Pixel, ExprT,
            std::multiplies<ImagePixelT>, bitwise_or<MaskPixelT>, variance_multiplies<VariancePixelT> >(tmp, e2);
        return tmp;
    }

    /// Evaluate e1 /= e2, and return e1
    template<typename ExprT>
    friend Pixel operator/=(Pixel const& e1, ExprT const& e2) {
        Pixel tmp(e1);                  // n.b. shares storage with e1 but gets around "const" (which is required)
        tmp = BinaryExpr<Pixel, ExprT,
            std::divides<ImagePixelT>, bitwise_or<MaskPixelT>, variance_divides<VariancePixelT> >(tmp, e2);
        return tmp;
    }

private:
    ImagePixelT& _image;
    MaskPixelT& _mask;
    VariancePixelT& _variance;
};

/************************************************************************************************************/
/// A traits class to return the types of the %image/mask/variance
template <typename ExprT>
struct exprTraits {
    typedef ExprT expr_type;
    typedef typename ExprT::ImagePixelT ImagePixelT;
    typedef typename ExprT::MaskPixelT MaskPixelT;
    typedef typename ExprT::VariancePixelT VariancePixelT;
};

/// A specialisation of exprTraits for \c double
template <>
struct exprTraits<double> {
    typedef double ImagePixelT;
    typedef int MaskPixelT;
    typedef double VariancePixelT;
    typedef SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT> expr_type;
};

/// A specialisation of exprTraits for \c float
template <>
struct exprTraits<float> {
    typedef float ImagePixelT;
    typedef exprTraits<double>::MaskPixelT MaskPixelT;
    typedef exprTraits<double>::VariancePixelT VariancePixelT;
    typedef SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT> expr_type;
};

/// A specialisation of exprTraits for \c int
template <>
struct exprTraits<int> {
    typedef int ImagePixelT;
    typedef exprTraits<double>::MaskPixelT MaskPixelT;
    typedef exprTraits<double>::VariancePixelT VariancePixelT;
    typedef SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT> expr_type;
};

/// A specialisation of exprTraits for \c unsigned short
template <>
struct exprTraits<unsigned short> {
    typedef int ImagePixelT;
    typedef exprTraits<double>::MaskPixelT MaskPixelT;
    typedef exprTraits<double>::VariancePixelT VariancePixelT;
    typedef SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT> expr_type;
};

/************************************************************************************************************/
///
/// \brief A noop functor (useful for e.g. masks and variances when changing the sign of the image)
///
template <typename T1>
struct noop : public std::unary_function<T1, T1> {
    T1 operator()(const T1& x) const {
        return x;
    }
};

///
/// \brief bitwise_or doesn't seem to be in std::
///
/// \note We provide a single-operand version for when the right-hand-side of an expression is a scalar, not a
/// masked pixel,
///
template <typename T1>
struct bitwise_or : public std::binary_function<T1, T1, T1> {
    T1 operator()(const T1& x, const T1& y) const {
        return (x | y);
    }
    T1 operator()(const T1& x) const {
        return x;
    }
};

///
/// \brief Calculate the variance when we divide two Pixels
///
/// \note We provide a single-operand version for when the right-hand-side of an expression is a scalar, not a
/// masked pixel,
///
template <typename T1>
struct variance_divides {
    T1 operator()(T1 const& x, T1 const& y, T1 const& vx, T1 const& vy) const {
        T1 const x2 = x*x;
        T1 const y2 = y*y;
        T1 const iy2 = 1.0/y2;
        return x2*vy*iy2*iy2 + vx*iy2;
    }

    T1 operator()(T1 const&, T1 const& y, T1 const& vx) const {
        return vx/(y*y);
    }
};
///
/// \brief Calculate the variance when we multiply two Pixels
///
/// \note We provide a single-operand version for when the right-hand-side of an expression is a scalar, not a
/// masked pixel,
///
template <typename T1>
struct variance_multiplies {
    T1 operator()(T1 const& x, T1 const& y, T1 const& vx, T1 const& vy) const {
        T1 const x2 = x*x;
        T1 const y2 = y*y;
        return x2*vy + y2*vx;
    }

    T1 operator()(T1 const&, T1 const& y, T1 const& vx) const {
        return vx*y*y;
    }
};
///
/// \brief Calculate the variance when we add (or subtract) two Pixels
///
/// \note We provide a single-operand version for when the right-hand-side of an expression is a scalar, not a
/// masked pixel,
///
template <typename T1>
struct variance_plus {
    T1 operator()(T1 const&, T1 const&, T1 const& vx, T1 const& vy) const {
        return vx + vy;
    }

    T1 operator()(T1 const&, T1 const&, T1 const& vx) const {
        return vx;
    }
};
///
/// \brief The variance of the sum of a pair of correlated pixels
///
/// The covariance is modelled as alpha*sqrt(var_x*var_y)
///
/// \note We provide a single-operand version for when the right-hand-side of an expression is a scalar, not a
/// masked pixel,
///
template <typename T1>
struct variance_plus_covar {
    variance_plus_covar(double alpha=0) : _alpha(alpha) {}

    T1 operator()(T1 const&, T1 const&, T1 const& vx, T1 const& vy) const {
        return vx + vy + 2*_alpha*sqrt(vx*vy);
    }
    T1 operator()(T1 const&, T1 const&, T1 const& vx) const {
        return vx;
    }
private:
    double _alpha;
};

/************************************************************************************************************/
/// Class for representing Unary operations
template <typename ExprT1, typename ImageBinOp, typename MaskBinOp, typename VarianceBinOp>
class UnaryExpr {
public:
    typedef typename exprTraits<ExprT1>::ImagePixelT ImagePixelT;
    typedef typename exprTraits<ExprT1>::MaskPixelT MaskPixelT;
    typedef typename exprTraits<ExprT1>::VariancePixelT VariancePixelT;
    /// a unary expression, with three functors to represent the %image/mask/variance operations
    UnaryExpr(ExprT1 e1,
              ImageBinOp imageOp=ImageBinOp(), MaskBinOp maskOp=MaskBinOp(), VarianceBinOp varOp=VarianceBinOp()) :
        _expr1(e1), _imageOp(imageOp), _maskOp(maskOp), _varOp(varOp) {}

    /// evaluate the %image part of the expression
    ImagePixelT image() const {
        return _imageOp(_expr1.image());
    }

    /// evaluate the mask part of the expression
    MaskPixelT mask() const {
        return _maskOp(_expr1.mask());
    }

    /// evaluate the variance part of the expression
    VariancePixelT variance() const {
        return _varOp(_expr1.variance());
    }
private:
    typename exprTraits<ExprT1>::expr_type _expr1;
    ImageBinOp _imageOp;
    MaskBinOp _maskOp;
    VarianceBinOp _varOp;
};

/// Class for representing binary operations
template <typename ExprT1, typename ExprT2, typename ImageBinOp, typename MaskBinOp, typename VarianceBinOp>
class BinaryExpr {
public:
    typedef typename exprTraits<ExprT1>::ImagePixelT ImagePixelT;
    typedef typename exprTraits<ExprT1>::MaskPixelT MaskPixelT;
    typedef typename exprTraits<ExprT1>::VariancePixelT VariancePixelT;
    /// A binary operation, with three functors to represent the %image/mask/variance operations
    BinaryExpr(ExprT1 e1, ExprT2 e2,
               ImageBinOp imageOp=ImageBinOp(), MaskBinOp maskOp=MaskBinOp(), VarianceBinOp varOp=VarianceBinOp()) :
        _expr1(e1), _expr2(e2), _imageOp(imageOp), _maskOp(maskOp), _varOp(varOp) {}

    /// A binary operation, with three functors to represent the %image/mask/variance operations and an extra double argument
    BinaryExpr(ExprT1 e1, ExprT2 e2, double const alpha,
               ImageBinOp imageOp=ImageBinOp(), MaskBinOp maskOp=MaskBinOp(), VarianceBinOp =VarianceBinOp()) :
        _expr1(e1), _expr2(e2), _imageOp(imageOp), _maskOp(maskOp), _varOp(VarianceBinOp(alpha)) {}
    /// evaluate the %image part of the expression
    ImagePixelT image() const {
        return _imageOp(_expr1.image(), _expr2.image());
    }

    /// evaluate the mask part of the expression
    MaskPixelT mask() const {
        return _maskOp(_expr1.mask(), _expr2.mask());
    }

    /// evaluate the variance part of the expression
    VariancePixelT variance() const {
        return _varOp(_expr1.image(), _expr2.image(), _expr1.variance(), _expr2.variance());
    }
private:
    typename exprTraits<ExprT1>::expr_type _expr1;
    typename exprTraits<ExprT2>::expr_type _expr2;
    ImageBinOp _imageOp;
    MaskBinOp _maskOp;
    VarianceBinOp _varOp;
};

/// Partial specialization of BinaryExpr when ExprT2 is a double (i.e no mask/variance part)
///
/// \todo Could use a traits class to handle all scalar types
template <typename ExprT1, typename ImageBinOp, typename MaskBinOp, typename VarianceBinOp>
class BinaryExpr<ExprT1, double, ImageBinOp, MaskBinOp, VarianceBinOp> {
public:
    typedef typename exprTraits<ExprT1>::ImagePixelT ImagePixelT;
    typedef typename exprTraits<ExprT1>::MaskPixelT MaskPixelT;
    typedef typename exprTraits<ExprT1>::VariancePixelT VariancePixelT;
    /// A binary operation, with three functors to represent the %image/mask/variance operations
    BinaryExpr(ExprT1 e1, double e2,
               ImageBinOp imageOp=ImageBinOp(), MaskBinOp maskOp=MaskBinOp(), VarianceBinOp varOp=VarianceBinOp()) :
        _expr1(e1), _expr2(e2), _imageOp(imageOp), _maskOp(maskOp), _varOp(varOp) {}

    /// A binary operation, with three functors to represent the %image/mask/variance operations and an extra double argument
    BinaryExpr(ExprT1 e1, double e2, double const alpha,
               ImageBinOp imageOp=ImageBinOp(), MaskBinOp maskOp=MaskBinOp(), VarianceBinOp=VarianceBinOp()) :
        _expr1(e1), _expr2(e2), _imageOp(imageOp), _maskOp(maskOp), _varOp(VarianceBinOp(alpha)) {}
    /// evaluate the %image part of the expression
    ImagePixelT image() const {
        return _imageOp(_expr1.image(), _expr2);
    }

    /// evaluate the mask part of the expression
    MaskPixelT mask() const {
        return _maskOp(_expr1.mask());
    }

    /// evaluate the variance part of the expression
    VariancePixelT variance() const {
        return _varOp(_expr1.image(), _expr2, _expr1.variance());
    }
private:
    typename exprTraits<ExprT1>::expr_type _expr1;
    double _expr2;
    ImageBinOp _imageOp;
    MaskBinOp _maskOp;
    VarianceBinOp _varOp;
};

/************************************************************************************************************/
/// Template for -e1
template <typename ExprT1>
UnaryExpr<ExprT1,
          std::negate<typename exprTraits<ExprT1>::ImagePixelT>,
          noop<typename exprTraits<ExprT1>::MaskPixelT>,
          noop<typename exprTraits<ExprT1>::VariancePixelT> > operator-(ExprT1 e1) {
    return UnaryExpr<ExprT1,
                  std::negate<typename exprTraits<ExprT1>::ImagePixelT>,
                  noop<typename exprTraits<ExprT1>::MaskPixelT>,
                  noop<typename exprTraits<ExprT1>::VariancePixelT> >(e1);
}

//------------------------------------------
/// Template for (e1 + e2)
template <typename ExprT1,typename ExprT2>
BinaryExpr<ExprT1, ExprT2,
           std::plus<typename exprTraits<ExprT1>::ImagePixelT>,
           bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
           variance_plus<typename exprTraits<ExprT1>::VariancePixelT> > operator+(ExprT1 e1, ExprT2 e2) {
    return BinaryExpr<ExprT1, ExprT2,
        std::plus<typename exprTraits<ExprT1>::ImagePixelT>,
        bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
        variance_plus<typename exprTraits<ExprT1>::VariancePixelT> >(e1,e2);
}

/// template for e1 += e2
template <typename ExprT1,typename ExprT2>
ExprT1 operator+=(ExprT1& e1, ExprT2 e2) {
    e1 = BinaryExpr<ExprT1, ExprT2,
        std::plus<typename exprTraits<ExprT1>::ImagePixelT>,
        bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
        variance_plus<typename exprTraits<ExprT1>::VariancePixelT> >(e1,e2);
    return e1;
}

//
// Implementations of add that work for arithmetic or MaskedImage pixels
//
// The choice is made on the basis of std::is_arithmetic
namespace {
    template <typename ExprT1,typename ExprT2>
    ExprT1 doPlus(ExprT1 e1, ExprT2 e2,
                  double const,
                  boost::mpl::true_) {
        return e1 + e2;
    }

    template <typename ExprT1,typename ExprT2>
    BinaryExpr<ExprT1, ExprT2,
               std::plus<typename exprTraits<ExprT1>::ImagePixelT>,
               bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
               variance_plus_covar<typename exprTraits<ExprT1>::VariancePixelT> > doPlus(ExprT1 e1, ExprT2 e2,
                                                                                         double const alpha,
                                                                                        boost::mpl::false_) {
        return BinaryExpr<ExprT1, ExprT2,
            std::plus<typename exprTraits<ExprT1>::ImagePixelT>,
            bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
            variance_plus_covar<typename exprTraits<ExprT1>::VariancePixelT> >(e1, e2, alpha);
    }
}

/// \brief Like operator+(), but assume that covariance's 2*alpha*sqrt(vx*vy)
template<typename ExprT1, typename ExprT2>
inline ExprT1 plus(ExprT1& lhs,          ///< Left hand value
                   ExprT2 const& rhs,    ///< Right hand value
                   float covariance      ///< Assume that covariance is 2*alpha*sqrt(vx*vy) (if variances are known)
                 ) {
    return doPlus(lhs, rhs, covariance, typename std::is_arithmetic<ExprT1>::type());
}

//------------------------------------------
/// Template to evaluate (e1 - e2)
template <typename ExprT1,typename ExprT2>
BinaryExpr<ExprT1, ExprT2,
           std::minus<typename exprTraits<ExprT1>::ImagePixelT>,
           bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
           variance_plus<typename exprTraits<ExprT1>::VariancePixelT> > operator-(ExprT1 e1, ExprT2 e2) {
    return BinaryExpr<ExprT1, ExprT2,
        std::minus<typename exprTraits<ExprT1>::ImagePixelT>,
        bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
        variance_plus<typename exprTraits<ExprT1>::VariancePixelT> >(e1,e2);
}

/// Template to evaluate e1 -= e2
template <typename ExprT1,typename ExprT2>
ExprT1 operator-=(ExprT1& e1, ExprT2 e2) {
    e1 = BinaryExpr<ExprT1, ExprT2,
        std::minus<typename exprTraits<ExprT1>::ImagePixelT>,
        bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
        variance_plus<typename exprTraits<ExprT1>::VariancePixelT> >(e1,e2);
    return e1;
}

//------------------------------------------
/// Template to evaluate (e1 * e2)
template <typename ExprT1,typename ExprT2>
BinaryExpr<ExprT1, ExprT2,
           std::multiplies<typename exprTraits<ExprT1>::ImagePixelT>,
           bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
           variance_multiplies<typename exprTraits<ExprT1>::VariancePixelT> > operator*(ExprT1 e1, ExprT2 e2) {
    return BinaryExpr<ExprT1, ExprT2,
        std::multiplies<typename exprTraits<ExprT1>::ImagePixelT>,
        bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
        variance_multiplies<typename exprTraits<ExprT1>::VariancePixelT> >(e1,e2);
}

/// Template to evaluate e1 *= e2
template <typename ExprT1,typename ExprT2>
ExprT1 operator*=(ExprT1& e1, ExprT2 e2) {
    e1 = BinaryExpr<ExprT1, ExprT2,
        std::multiplies<typename exprTraits<ExprT1>::ImagePixelT>,
        bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
        variance_multiplies<typename exprTraits<ExprT1>::VariancePixelT> >(e1,e2);
    return e1;
}

//------------------------------------------
/// Template to evaluate (e1 / e2)
template <typename ExprT1,typename ExprT2>
BinaryExpr<ExprT1, ExprT2,
           std::divides<typename exprTraits<ExprT1>::ImagePixelT>,
           bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
           variance_divides<typename exprTraits<ExprT1>::VariancePixelT> > operator/(ExprT1 e1, ExprT2 e2) {
    return BinaryExpr<ExprT1, ExprT2,
        std::divides<typename exprTraits<ExprT1>::ImagePixelT>,
        bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
        variance_divides<typename exprTraits<ExprT1>::VariancePixelT> >(e1,e2);
}

/// Template to evaluate e1 /= e2
template <typename ExprT1,typename ExprT2>
ExprT1 operator/=(ExprT1& e1, ExprT2 e2) {
    e1 = BinaryExpr<ExprT1, ExprT2,
        std::divides<typename exprTraits<ExprT1>::ImagePixelT>,
        bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
        variance_divides<typename exprTraits<ExprT1>::VariancePixelT> >(e1,e2);
    return e1;
}

/************************************************************************************************************/
/// Print a SinglePixel
template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
std::ostream& operator<<(std::ostream &os, SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT> const& v) {
    return os << "(" << v.image() << ", " << v.mask() << ", " << v.variance() << ")";
}

/// Print a Pixel
template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
std::ostream& operator<<(std::ostream &os, Pixel<ImagePixelT, MaskPixelT, VariancePixelT> const& v) {
    return os << "(" << v.image() << ", " << v.mask() << ", " << v.variance() << ")";
}

/// Evaluate and print a BinaryExpr
template <typename ExprT1,typename ExprT2, typename BinOp, typename MaskBinOp, typename VarBinOp>
std::ostream& operator<<(std::ostream &os, BinaryExpr<ExprT1, ExprT2, BinOp, MaskBinOp, VarBinOp> const& v) {
    return os << "(" << v.image() << ", " << v.mask() << ", " << v.variance() << ")";
}

}}}}
#endif
