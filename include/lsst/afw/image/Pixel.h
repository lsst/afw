#if !defined(LSST_AFW_IMAGE_PIXEL_H)
#define LSST_AFW_IMAGE_PIXEL_H

#include <cmath>
#include <iostream>
#include <functional>

namespace lsst { namespace afw { namespace image { namespace pixel {

template <typename, typename, typename, typename, typename> class BinaryExpr;

template <typename> struct exprTraits;

template <typename> struct bitwise_or;
template <typename> struct variance_divides;
template <typename> struct variance_multiplies;
template <typename> struct variance_plus;

/************************************************************************************************************/
/// @brief Classes to provide utility functions for a "Pixel" to get at image/mask/variance operators
//
// These classes allow us to manipulate the tuples returned by MaskedImage iterators/locators as if they were
// POD.  This provides convenient syntactic sugar, but it also permits us to write generic algorithms to
// manipulate MaskedImages as well as Images
//
// We need SinglePixel as well as Pixel as the latter is just a reference to a pixel in an image, and we need
// to be able to build temporary values too
//
// We use C++ template expressions to manipulate Pixel and SinglePixel; this permits us to avoid making
// SinglePixel inherit from Pixel (or vice versa) and allows the compiler to do a much better job of
// optimising mixed-mode expressions --- basically, it no longer needs to create SinglePixels as Pixels that
// have their own storage but use the reference members of Pixel to get work done.  There may be better ways,
// but this way works, and g++ 4.0.1 (and maybe other compilers/versions) failed to optimise the previous
// solution very well.

template<typename _ImagePixelT, typename _MaskPixelT, typename _VariancePixelT=double>
class SinglePixel : detail::maskedImagePixel_tag {
public:
    template<typename, typename, typename> friend class Pixel;

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
    ImagePixelT _image;
    MaskPixelT _mask;
    VariancePixelT _variance;
};

template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT> makeSinglePixel(ImagePixelT x, MaskPixelT m, VariancePixelT v) {
    return SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT>(x, m, v);
}

template<typename _ImagePixelT, typename _MaskPixelT, typename _VariancePixelT=double>
class Pixel : detail::maskedImagePixel_tag {
public:
    typedef _ImagePixelT ImagePixelT;
    typedef _MaskPixelT MaskPixelT;
    typedef _VariancePixelT VariancePixelT;

#if 0
    Pixel(ImagePixelT& image, MaskPixelT& mask, VariancePixelT& variance) :
        _image(image), _mask(mask), _variance(variance) {}
#else
    //
    // This constructor casts away const.  This should be fixed by making const Pixels.
    //
    Pixel(ImagePixelT const& image, MaskPixelT const& mask, VariancePixelT const& variance) :
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

    template<typename rhsExpr>
    Pixel operator=(rhsExpr const& rhs) {
        _variance = rhs.variance();     // evaluate before we update image()
        _image = rhs.image();
        _mask = rhs.mask();

        return *this;
    }

    Pixel operator=(double const& rhs_image) {
        _image = rhs_image;
        _mask = 0;
        _variance = 0;

        return *this;
    }

    Pixel operator=(int const& rhs_image) {
        _image = rhs_image;
        _mask = 0;
        _variance = 0;

        return *this;
    }

    ImagePixelT image() const { return _image; }
    MaskPixelT mask() const { return _mask; }
    VariancePixelT variance() const { return _variance; }
    //
    // Logical operators.  We don't need to construct BinaryExpr for them
    // as efficiency isn't a concern.
    //
    template<typename T1>
    friend bool operator==(Pixel const& lhs, T1 const& rhs) {
        return lhs.image() == rhs.image() && lhs.mask() == rhs.mask() && lhs.variance() == rhs.variance();
    }
    
    template<typename T1>
    friend bool operator!=(Pixel const& lhs, T1 const& rhs) {
        return !(lhs == rhs);
    }

    //
    // Provide friend versions of the op= operators to permit argument promotion on their first arguments
    //
    template<typename ExprT>
    friend Pixel operator+=(Pixel const& e1, ExprT const& e2) {
        Pixel tmp(e1);                  // n.b. shares storage with e1 but gets around "const" (which is required)
        tmp = BinaryExpr<Pixel, ExprT,
            std::plus<ImagePixelT>, bitwise_or<MaskPixelT>, variance_plus<VariancePixelT> >(tmp, e2);
        return tmp;
    }

    template<typename ExprT>
    friend Pixel operator-=(Pixel const& e1, ExprT const& e2) {
        Pixel tmp(e1);                  // n.b. shares storage with e1 but gets around "const" (which is required)
        tmp = BinaryExpr<Pixel, ExprT,
            std::minus<ImagePixelT>, bitwise_or<MaskPixelT>, variance_plus<VariancePixelT> >(tmp, e2);
        return tmp;
    }


    template<typename ExprT>
    friend Pixel operator*=(Pixel const& e1, ExprT const& e2) {
        Pixel tmp(e1);                  // n.b. shares storage with e1 but gets around "const" (which is required)
        tmp = BinaryExpr<Pixel, ExprT,
            std::multiplies<ImagePixelT>, bitwise_or<MaskPixelT>, variance_multiplies<VariancePixelT> >(tmp, e2);
        return tmp;
    }

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

template <typename ExprT>
struct exprTraits {
    typedef ExprT expr_type;
    typedef typename ExprT::ImagePixelT ImagePixelT;
    typedef typename ExprT::MaskPixelT MaskPixelT;
    typedef typename ExprT::VariancePixelT VariancePixelT;
};

template <>
struct exprTraits<double> {
    typedef double ImagePixelT;
    typedef int MaskPixelT;
    typedef double VariancePixelT;
    typedef SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT> expr_type;
};

template <>
struct exprTraits<float> {
    typedef float ImagePixelT;
    typedef exprTraits<double>::MaskPixelT MaskPixelT;
    typedef exprTraits<double>::VariancePixelT VariancePixelT;
    typedef SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT> expr_type;
};

template <>
struct exprTraits<int> {
    typedef int ImagePixelT;
    typedef exprTraits<double>::MaskPixelT MaskPixelT;
    typedef exprTraits<double>::VariancePixelT VariancePixelT;
    typedef SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT> expr_type;
};

template <>
struct exprTraits<unsigned short> {
    typedef int ImagePixelT;
    typedef exprTraits<double>::MaskPixelT MaskPixelT;
    typedef exprTraits<double>::VariancePixelT VariancePixelT;
    typedef SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT> expr_type;
};

/************************************************************************************************************/
//
// Here's a noop (useful for e.g. masks and variances when changing the sign of the image)
//
template <typename T1>
struct noop : public std::unary_function<T1, T1> {
    T1 operator()(const T1& x) const {
        return x;
    }
};

//
// bitwise_or doesn't seem to be in std::
//
template <typename T1>
struct bitwise_or : public std::binary_function<T1, T1, T1> {
    T1 operator()(const T1& x, const T1& y) const {
        return (x | y);
    }
};
    
//
// Propagate the variance when we divide two Pixels
//
template <typename T1>
struct variance_divides {
    T1 operator()(T1 const& x, T1 const& y, T1 const& vx, T1 const& vy) const {
        T1 const x2 = x*x;
        T1 const y2 = y*y;
        return (x2*vy + y2*vx)/(y2*y2);
    }
};
//
// Propagate the variance when we multiply two Pixels
//
template <typename T1>
struct variance_multiplies {
    T1 operator()(T1 const& x, T1 const& y, T1 const& vx, T1 const& vy) const {
        T1 const x2 = x*x;
        T1 const y2 = y*y;
        return x2*vy + y2*vx;
    }
};
//
// Propagate the variance when we add (or subtract) two Pixels
//
template <typename T1>
struct variance_plus {
    T1 operator()(T1 const& x, T1 const& y, T1 const& vx, T1 const& vy) const {
        return vx + vy;
    }
};

template <typename T1>
struct variance_plus_covar {
    variance_plus_covar(double alpha=0) : _alpha(alpha) {}    

    T1 operator()(T1 const& x, T1 const& y, T1 const& vx, T1 const& vy) const {
        return vx + vy + 2*_alpha*sqrt(vx*vy);
    }
private:
    double _alpha;
};
                
/************************************************************************************************************/

template <typename ExprT1, typename ImageBinOp, typename MaskBinOp, typename VarianceBinOp>
class UnaryExpr {
public:
    typedef typename exprTraits<ExprT1>::ImagePixelT ImagePixelT;
    typedef typename exprTraits<ExprT1>::MaskPixelT MaskPixelT;
    typedef typename exprTraits<ExprT1>::VariancePixelT VariancePixelT;

    UnaryExpr(ExprT1 e1,
              ImageBinOp imageOp=ImageBinOp(), MaskBinOp maskOp=MaskBinOp(), VarianceBinOp varOp=VarianceBinOp()) :
        _expr1(e1), _imageOp(imageOp), _maskOp(maskOp), _varOp(varOp) {}
    
    ImagePixelT image() const {
        return _imageOp(_expr1.image());
    }

    MaskPixelT mask() const {
        return _maskOp(_expr1.mask());
    }

    VariancePixelT variance() const {
        return _varOp(_expr1.variance());
    }
private:
    typename exprTraits<ExprT1>::expr_type _expr1;
    ImageBinOp _imageOp;
    MaskBinOp _maskOp;
    VarianceBinOp _varOp;
};

template <typename ExprT1, typename ExprT2, typename ImageBinOp, typename MaskBinOp, typename VarianceBinOp>
class BinaryExpr {
public:
    typedef typename exprTraits<ExprT1>::ImagePixelT ImagePixelT;
    typedef typename exprTraits<ExprT1>::MaskPixelT MaskPixelT;
    typedef typename exprTraits<ExprT1>::VariancePixelT VariancePixelT;

    BinaryExpr(ExprT1 e1, ExprT2 e2,
               ImageBinOp imageOp=ImageBinOp(), MaskBinOp maskOp=MaskBinOp(), VarianceBinOp varOp=VarianceBinOp()) :
        _expr1(e1), _expr2(e2), _imageOp(imageOp), _maskOp(maskOp), _varOp(varOp) {}

    BinaryExpr(ExprT1 e1, ExprT2 e2, double const alpha,
               ImageBinOp imageOp=ImageBinOp(), MaskBinOp maskOp=MaskBinOp(), VarianceBinOp varOp=VarianceBinOp()) :
        _expr1(e1), _expr2(e2), _imageOp(imageOp), _maskOp(maskOp), _varOp(VarianceBinOp(alpha)) {}
    
    ImagePixelT image() const {
        return _imageOp(_expr1.image(), _expr2.image());
    }

    MaskPixelT mask() const {
        return _maskOp(_expr1.mask(), _expr2.mask());
    }

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

/************************************************************************************************************/

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

#if 1
template <typename ExprT1,typename ExprT2>
ExprT1 operator+=(ExprT1& e1, ExprT2 e2) {
    e1 = BinaryExpr<ExprT1, ExprT2,
        std::plus<typename exprTraits<ExprT1>::ImagePixelT>,
        bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
        variance_plus<typename exprTraits<ExprT1>::VariancePixelT> >(e1,e2);
    return e1;
}
#endif

//
// Implementations of add that work for arithmetic or MaskedImage pixels
//
// The choice is made on the basis of boost::is_arithmetic
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

/// @brief Like operator+(), but assume that covariance's 2*alpha*sqrt(vx*vy)
template<typename ExprT1, typename ExprT2>
inline ExprT1 plus(ExprT1& lhs,          ///< Left hand value
                   ExprT2 const& rhs,    ///< Right hand value
                   float covariance      ///< Assume that covariance is 2*alpha*sqrt(vx*vy) (if variances are known)
                 ) {
    return doPlus(lhs, rhs, covariance, typename boost::is_arithmetic<ExprT1>::type());
}

//------------------------------------------

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

#if 1
template <typename ExprT1,typename ExprT2>
ExprT1 operator-=(ExprT1& e1, ExprT2 e2) {
    e1 = BinaryExpr<ExprT1, ExprT2,
        std::minus<typename exprTraits<ExprT1>::ImagePixelT>,
        bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
        variance_plus<typename exprTraits<ExprT1>::VariancePixelT> >(e1,e2);
    return e1;
}
#endif

//------------------------------------------

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

#if 1
template <typename ExprT1,typename ExprT2>
ExprT1 operator*=(ExprT1& e1, ExprT2 e2) {
    e1 = BinaryExpr<ExprT1, ExprT2,
        std::multiplies<typename exprTraits<ExprT1>::ImagePixelT>,
        bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
        variance_multiplies<typename exprTraits<ExprT1>::VariancePixelT> >(e1,e2);
    return e1;
}
#endif

//------------------------------------------

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

#if 1
template <typename ExprT1,typename ExprT2>
ExprT1 operator/=(ExprT1& e1, ExprT2 e2) {
    e1 = BinaryExpr<ExprT1, ExprT2,
        std::divides<typename exprTraits<ExprT1>::ImagePixelT>,
        bitwise_or<typename exprTraits<ExprT1>::MaskPixelT>,
        variance_divides<typename exprTraits<ExprT1>::VariancePixelT> >(e1,e2);
    return e1;
}
#endif

/************************************************************************************************************/

template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
std::ostream& operator<<(std::ostream &os, SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT> const& v) {
    return os << "(" << v.image() << ", " << v.mask() << ", " << v.variance() << ")";
}

template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
std::ostream& operator<<(std::ostream &os, Pixel<ImagePixelT, MaskPixelT, VariancePixelT> const& v) {
    return os << "(" << v.image() << ", " << v.mask() << ", " << v.variance() << ")";
}

template <typename ExprT1,typename ExprT2, typename BinOp, typename MaskBinOp, typename VarBinOp>
std::ostream& operator<<(std::ostream &os, BinaryExpr<ExprT1, ExprT2, BinOp, MaskBinOp, VarBinOp> const& v) {
    return os << "(" << v.image() << ", " << v.mask() << ", " << v.variance() << ")";
}

}}}}
#endif
