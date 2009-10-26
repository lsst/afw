// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_FUNCTIONLIBRARY_H
#define LSST_AFW_MATH_FUNCTIONLIBRARY_H
/**
 * @file
 *
 * @brief Define a collection of useful Functions.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <cmath>

#include "lsst/afw/math/Function.h"

namespace lsst {
namespace afw {
namespace math {

#ifndef SWIG
using boost::serialization::make_nvp;
#endif

    /**
     * @brief 1-dimensional integer delta function.
     *
     * f(x) = 1 if x == xo, 0 otherwise.
     *
     * For use as a kernel function be sure to handle the offset for row and column center;
     * see examples/deltaFunctionKernel for an example.
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class IntegerDeltaFunction1: public Function1<ReturnT> {
    public:
        typedef typename Function1<ReturnT>::Ptr Function1Ptr;

        /**
         * @brief Construct an integer delta function with specified xo, yo
         */
        explicit IntegerDeltaFunction1(
            double xo)
        :
            Function1<ReturnT>(0),
            _xo(xo)
        {}
        
        virtual ~IntegerDeltaFunction1() {};
        
        virtual Function1Ptr clone() const {
            return Function1Ptr(new IntegerDeltaFunction1(_xo));
        }
        
        virtual ReturnT operator() (double x, double y) const {
            return static_cast<ReturnT>(x == _xo);
        }

        virtual std::string toString(void) const {
            std::ostringstream os;
            os << "IntegerDeltaFunction1 [" << _xo << "]: ";
            os << Function1<ReturnT>::toString();
            return os.str();
        };

    private:
        double _xo;

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                IntegerDeltaFunction1<ReturnT>, Function1<ReturnT> >(
                    static_cast<IntegerDeltaFunction1<ReturnT>*>(0),
                    static_cast<Function1<ReturnT>*>(0));
#endif
        };
        template <typename R, class Archive>
        friend void boost::serialization::save_construct_data(
            Archive& ar, IntegerDeltaFunction1<R> const* f, unsigned int const version);
    };

    /**
     * @brief 2-dimensional integer delta function.
     *
     * f(x) = 1 if x == xo and y == yo, 0 otherwise.
     *
     * For use as a kernel function be sure to handle the offset for row and column center;
     * see examples/deltaFunctionKernel for an example.
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class IntegerDeltaFunction2: public Function2<ReturnT> {
    public:
        typedef typename Function2<ReturnT>::Ptr Function2Ptr;

        /**
         * @brief Construct an integer delta function with specified xo, yo
         */
        explicit IntegerDeltaFunction2(
            double xo,
            double yo)
        :
            Function2<ReturnT>(0),
            _xo(xo),
            _yo(yo)
        {}
        
        virtual ~IntegerDeltaFunction2() {};
        
        virtual Function2Ptr clone() const {
            return Function2Ptr(new IntegerDeltaFunction2(_xo, _yo));
        }
        
        virtual ReturnT operator() (double x, double y) const {
            return static_cast<ReturnT>((x == _xo) && (y == _yo));
        }

        virtual std::string toString(void) const {
            std::ostringstream os;
            os << "IntegerDeltaFunction2 [" << _xo << ", " << _yo << "]: ";
            os << Function2<ReturnT>::toString();
            return os.str();
        };

    private:
        double _xo;
        double _yo;

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                IntegerDeltaFunction2<ReturnT>, Function2<ReturnT> >(
                    static_cast<IntegerDeltaFunction2<ReturnT>*>(0),
                    static_cast<Function2<ReturnT>*>(0));
#endif
        };
        template <typename R, class Archive>
        friend void boost::serialization::save_construct_data(
            Archive& ar, IntegerDeltaFunction2<R> const* f, unsigned int const version);
    };

    /**
     * @brief 1-dimensional Gaussian
     *
     * f(x) = (1 / (sqrt(2 pi) xSigma)) e^(-x^2 / sigma^2)
     * with coefficient c0 = sigma
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class GaussianFunction1: public Function1<ReturnT> {
    public:
        typedef typename Function1<ReturnT>::Ptr Function1Ptr;

        /**
         * @brief Construct a Gaussian function with specified sigma
         */
        explicit GaussianFunction1(
            double sigma)    ///< sigma
        :
            Function1<ReturnT>(1),
            _multFac(1.0 / std::sqrt(2.0 * M_PI))
        {
            this->_params[0] = sigma;
        }
        virtual ~GaussianFunction1() {};
        
        virtual Function1Ptr clone() const {
            return Function1Ptr(new GaussianFunction1(this->_params[0]));
        }
        
        virtual ReturnT operator() (double x) const {
            return static_cast<ReturnT> (
                (_multFac / this->_params[0]) *
                std::exp(- (x * x) / (2.0 * this->_params[0] * this->_params[0])));
        }
        
        virtual std::string toString(void) const {
            std::ostringstream os;
            os << "GaussianFunction1 [" << _multFac << "]: ";
            os << Function1<ReturnT>::toString();
            return os.str();
        };

    private:
        const double _multFac; ///< precomputed scale factor

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                GaussianFunction1<ReturnT>, Function1<ReturnT> >(
                    static_cast<GaussianFunction1<ReturnT>*>(0),
                    static_cast<Function1<ReturnT>*>(0));
#endif
        };
    };
    
    /**
     * @brief 2-dimensional Gaussian
     *
     * f(x,y) = (1 / (2 pi xSigma ySigma)) e^(-x^2 / xSigma^2) e^(-y^2 / ySigma^2) /
     * with coefficients c0 = xSigma and c1 = ySigma
     *
     * @todo Allow setting angle of ellipticity.
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class GaussianFunction2: public Function2<ReturnT> {
    public:
        typedef typename Function2<ReturnT>::Ptr Function2Ptr;

        /**
         * @brief Construct a Gaussian function with specified x and y sigma
         */
        explicit GaussianFunction2(
            double xSigma,  ///< sigma in x
            double ySigma)  ///< sigma in y
        : 
            Function2<ReturnT>(2),
            _multFac(1.0 / (2.0 * M_PI))
        {
            this->_params[0] = xSigma;
            this->_params[1] = ySigma;
        }
        
        virtual ~GaussianFunction2() {};
        
        virtual Function2Ptr clone() const {
            return Function2Ptr(new GaussianFunction2(this->_params[0], this->_params[1]));
        }
        
        virtual ReturnT operator() (double x, double y) const {
            return static_cast<ReturnT> (
                (_multFac / (this->_params[0] * this->_params[1])) *
                std::exp(- ((x * x) / (2.0 * this->_params[0] * this->_params[0]))
                         - ((y * y) / (2.0 * this->_params[1] * this->_params[1]))));
        }
        
        virtual std::string toString(void) const {
            std::ostringstream os;
            os << "GaussianFunction2 [" << _multFac << "]: ";
            os << Function2<ReturnT>::toString();
            return os.str();
        };

    private:
        const double _multFac; ///< precomputed scale factor

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                GaussianFunction2<ReturnT>, Function2<ReturnT> >(
                    static_cast<GaussianFunction2<ReturnT>*>(0),
                    static_cast<Function2<ReturnT>*>(0));
#endif
        };
    };
    
    /**
     * @brief double Guassian (sum of two Gaussians)
     *
     * Intended for use as a PSF model: the main Gaussian represents the core
     * and the second Gaussian represents the wings.
     *
     * f(x,y) = (1 / (2 pi (sigma1^2 + b sigma2^2))) (e^(-r^2 / sigma1^2) + b e^(-r^2 / sigma2^2))
     * where r^2 = x^2 + y^2
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class DoubleGaussianFunction2: public Function2<ReturnT> {
    public:
        typedef typename Function2<ReturnT>::Ptr Function2Ptr;

        /**
         * @brief Construct a Gaussian function with specified x and y sigma
         */
        explicit DoubleGaussianFunction2(
            double sigma1,      ///< sigma of main Gaussian
            double sigma2 = 0,  ///< sigma of second Gaussian
            double ampl2 = 0)   ///< amplitude of second Gaussian as a fraction of main Gaussian at peak
        : 
            Function2<ReturnT>(3),
            _multFac(1.0 / (2.0 * M_PI))
        {
            this->_params[0] = sigma1;
            this->_params[1] = sigma2;
            this->_params[2] = ampl2;
        }
        
        virtual ~DoubleGaussianFunction2() {};
        
        virtual Function2Ptr clone() const {
            return Function2Ptr(
                new DoubleGaussianFunction2(this->_params[0], this->_params[1], this->_params[2]));
        }
        
        virtual ReturnT operator() (double x, double y) const {
            double radSq = (x * x) + (y * y);
            double sigma1Sq = this->_params[0] * this->_params[0];
            double sigma2Sq = this->_params[1] * this->_params[1];
            double b = this->_params[2];
            return static_cast<ReturnT> (
                (_multFac / (sigma1Sq + (b * sigma2Sq))) *
                (std::exp(-radSq / (2.0 * sigma1Sq))
                + (b * std::exp(-radSq / (2.0 * sigma2Sq)))));
        }
        
        virtual std::string toString(void) const {
            std::ostringstream os;
            os << "DoubleGaussianFunction2 [" << _multFac << "]: ";
            os << Function2<ReturnT>::toString();
            return os.str();
        };

    private:
        const double _multFac; ///< precomputed scale factor

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                DoubleGaussianFunction2<ReturnT>, Function2<ReturnT> >(
                    static_cast<DoubleGaussianFunction2<ReturnT>*>(0),
                    static_cast<Function2<ReturnT>*>(0));
#endif
        };
    };
    
    /**
     * @brief 1-dimensional polynomial function.
     *
     * f(x) = c0 + c1 x + c2 x^2 + ... cn-1 x^(n-1)
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class PolynomialFunction1: public Function1<ReturnT> {
    public:
        typedef typename Function1<ReturnT>::Ptr Function1Ptr;

        /**
         * @brief Construct a polynomial function of the specified order.
         *
         * The parameters are initialized to zero.
         */
        explicit PolynomialFunction1(
            unsigned int order)     ///< order of polynomial (0 for constant)
        :
            Function1<ReturnT>(order+1) {
        }
        
        /**
         * @brief Construct a polynomial function with the specified parameters.
         *
         * The order of the polynomial is set to the length of the params vector.
         *
         * @throw lsst::pex::exceptions::InvalidParameter if params is empty
         */
        explicit PolynomialFunction1(
            std::vector<double> params)  ///< polynomial coefficients (const, x, x^2...)
        :
            Function1<ReturnT>(params)
        {
            if (params.size() < 1) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                                  "PolynomialFunction1 called with empty vector");
            }
        }
        
        virtual ~PolynomialFunction1() {};
       
        virtual Function1Ptr clone() const {
            return Function1Ptr(new PolynomialFunction1(this->_params));
        }
        
        virtual ReturnT operator() (double x) const {
            int maxInd = static_cast<int>(this->_params.size()) - 1;
            double retVal = this->_params[maxInd];
            for (int ii = maxInd-1; ii >= 0; --ii) {
                retVal = (retVal * x) + this->_params[ii];
            }
            return static_cast<ReturnT>(retVal);
        }

        virtual std::string toString(void) const {
            std::ostringstream os;
            os << "PolynomialFunction1 []: ";
            os << Function1<ReturnT>::toString();
            return os.str();
        };


    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                PolynomialFunction1<ReturnT>, Function1<ReturnT> >(
                    static_cast<PolynomialFunction1<ReturnT>*>(0),
                    static_cast<Function1<ReturnT>*>(0));
#endif
        };
    };

    /**
     * @brief 2-dimensional polynomial function with cross terms
     *
     * f(x,y) = c0                                          (0th order)
     *          + c1 x + c2 y                               (1st order)
     *          + c2 x^2 + c3 x y + c4 y^2                  (2nd order)
     *          + c5 x^3 + c6 x^2 y + c7 x y^2 + c8 y^3     (3rd order)
     *          + ...
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class PolynomialFunction2: public Function2<ReturnT> {
    public:
        typedef typename Function2<ReturnT>::Ptr Function2Ptr;

        /**
         * @brief Construct a polynomial function of specified order.
         *
         * The polynomial will have (order + 1) * (order + 2) / 2 coefficients
         *
         * The parameters are initialized to zero.
         */
        explicit PolynomialFunction2(
            unsigned int order) ///< order of polynomial (0 for constant)
        :
            Function2<ReturnT>((order + 1) * (order + 2) / 2),
            _order(order)
        {}

        /**
         * @brief Construct a polynomial function with specified parameters.
         *
         * The order of the polynomial is determined from the length of the params vector:
         *   order = (sqrt(1 + 8 * length) - 3) / 2
         * and if this is not an integer then the length is unsuitable
         *
         * @throw lsst::pex::exceptions::InvalidParameterException if params length is unsuitable
         * @throw lsst::pex::exceptions::Exception if an internal sanity check fails
         */
        explicit PolynomialFunction2(
            std::vector<double> params)  ///< polynomial coefficients (const, x, y, x^2, xy, y^2...);
                                    ///< length must be one of 1, 3, 6, 10, 15...
        :
            Function2<ReturnT>(params),
            _order(static_cast<unsigned int>(
                0.5 + ((-3.0 + (std::sqrt(1.0 + (8.0 * static_cast<double>(params.size()))))) / 2.0)
            ))
        {
            unsigned int nParams = params.size();
            if (nParams < 1) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                                  "PolynomialFunction2 created with empty vector");
            }
            if (nParams != ((_order + 1) * (_order + 2)) / 2) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                                  "PolynomialFunction2 created with vector of unusable length");
            }
        }
        
        virtual ~PolynomialFunction2() {};
       
        virtual Function2Ptr clone() const {
            return Function2Ptr(new PolynomialFunction2(this->_params));
        }
        
        virtual ReturnT operator() (double x, double y) const {
            /* Solve as follows:
            - f(x,y) = Cy0 + Cy1 y + Cy2 y^2 + Cy3 y^3 + ...
            where:
              Cy0 = P0 + P1 x + P3 x^2 + P6 x^3 + ...
              Cy1 = P2 + P4 x + P7 x2 + ...
              Cy2 = P5 + P8 x + ...
              Cy3 = P9 + ...
            First compute the y coefficients: 1-d polynomials in x solved in the usual way.
            Then compute the return value: a 1-d polynomial in y solved in the usual way.
            */
            const int maxYCoeffInd = this->_order;
            std::vector<double> yCoeffs(maxYCoeffInd + 1);
            int paramInd = static_cast<int>(this->_params.size()) - 1;
            // initialize the y coefficients
            for (int yCoeffInd = maxYCoeffInd; yCoeffInd >= 0; --yCoeffInd, --paramInd) {
                yCoeffs[yCoeffInd] = this->_params[paramInd];
            }
            // finish computing the y coefficients
            for (int startYCoeffInd = maxYCoeffInd - 1, yCoeffInd = startYCoeffInd;
                paramInd >= 0; --paramInd) {
                yCoeffs[yCoeffInd] = (yCoeffs[yCoeffInd] * x) + this->_params[paramInd];
                if (yCoeffInd == 0) {
                    --startYCoeffInd;
                    yCoeffInd = startYCoeffInd;
                } else {
                    --yCoeffInd;
                }
            }
            // compute y polynomial
            double retVal = yCoeffs[maxYCoeffInd];
            for (int yCoeffInd = maxYCoeffInd - 1; yCoeffInd >= 0; --yCoeffInd) {
                retVal = (retVal * y) + yCoeffs[yCoeffInd];
            }
            return static_cast<ReturnT>(retVal);
        }

        virtual std::string toString(void) const {
            std::ostringstream os;
            os << "PolynomialFunction2 [" << _order << "]: ";
            os << Function2<ReturnT>::toString();
            return os.str();
        };

    private:
        unsigned int _order; ///< order of polynomial

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
            ar & make_nvp("fn2",
                          boost::serialization::base_object<
                          Function2<ReturnT> >(*this));
            ar & make_nvp("order", _order);
        };
    };
    
    /**
     * @brief 1-dimensional weighted sum of Chebyshev polynomials of the first kind.
     *
     * f(x) = c0 + c1 * T1(x') + c2 * T2(x') + ...
     * where:
     *   x' ranges over [-1, 1] as x ranges over [xMin, xMax]
     *   Tn(x) is the nth Chebyshev function of the first kind:
     *     Tn(x) = cos(n arccos(x))
     *
     * The function argument must be in the range [xMin, xMax].
     *
     * Note: solved using the Clenshaw algorithm. This avoids cosines,
     * but is recursive and so (presumably) cannot be inlined.
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class Chebyshev1Function1: public Function1<ReturnT> {
    public:
        typedef typename Function1<ReturnT>::Ptr Function1Ptr;

        /**
         * @brief Construct a Chebyshev polynomial of specified order and range.
         *
         * The parameters are initialized to zero.
         */
        explicit Chebyshev1Function1(
            unsigned int order, ///< order of polynomial (0 for constant)
            double xMin = -1,    ///< minimum allowed x
            double xMax = 1)     ///< maximum allowed x
        :
            Function1<ReturnT>(order + 1)
        {
            _initialize(xMin, xMax);
        }

        /**
         * @brief Construct a Chebyshev polynomial with specified parameters and range.
         *
         * The order of the polynomial is set to the length of the params vector.
         *
         * @throw lsst::pex::exceptions::InvalidParameterException if params is empty
         */
        explicit Chebyshev1Function1(
            std::vector<double> params,  ///< polynomial coefficients
            double xMin = -1,    ///< minimum allowed x
            double xMax = 1)     ///< maximum allowed x
        :
            Function1<ReturnT>(params)
        {
            if (params.size() < 1) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                                  "Chebyshev1Function1 called with empty vector");
            }
            _initialize(xMin, xMax);
        }
        
        virtual ~Chebyshev1Function1() {};
       
        virtual Function1Ptr clone() const {
            return Function1Ptr(new Chebyshev1Function1(this->_params, _minX, _maxX));
        }
        
        virtual ReturnT operator() (double x) const {
            double xPrime = (x + _offset) * _scale;
            return static_cast<ReturnT>(_clenshaw(xPrime, 0));
        }

        virtual std::string toString(void) const {
            std::ostringstream os;
            os << "Chebyshev1Function1 [";
            os << _minX << ", " << _maxX << ", ";
            os << _scale << ", " << _offset << ", ";
            os << _maxInd << "]: ";
            os << Function1<ReturnT>::toString();
            return os.str();
        };

    private:
        double _minX;    ///< minimum allowed x
        double _maxX;    ///< maximum allowed x
        double _scale;   ///< x' = (x + _offset) * _scale
        double _offset;  ///< x' = (x + _offset) * _scale
        unsigned int _maxInd;   ///< maximum index for Clenshaw function
        
        /**
         * @brief Clenshaw recursive function for solving the Chebyshev polynomial
         */
        double _clenshaw(double x, unsigned int ind) const {
            if (ind == _maxInd) {
                return this->_params[ind];
            } else if (ind == 0) {
                return (x * _clenshaw(x, 1)) + this->_params[0] - _clenshaw(x, 2);
            } else if (ind == _maxInd - 1) {
                return (2 * x * _clenshaw(x, ind+1)) + this->_params[ind];
            } else if (ind < _maxInd) {
                return (2 * x * _clenshaw(x, ind+1)) + this->_params[ind] - _clenshaw(x, ind+2);
            } else {
                // this case only occurs if _maxInd < 3
                return 0;
            }
        }
        
        /**
         * @brief initialize private constants
         */
        void _initialize(double xMin, double xMax) {
            _minX = xMin;
            _maxX = xMax;
            _scale = 2 / (xMax - xMin);
            _offset = -(xMin + xMax) / 2.0;
            _maxInd = this->getNParameters() - 1;
        }

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                Chebyshev1Function1<ReturnT>, Function1<ReturnT> >(
                    static_cast<Chebyshev1Function1<ReturnT>*>(0),
                    static_cast<Function1<ReturnT>*>(0));
#endif
        };
        template <typename R, class Archive>
        friend void boost::serialization::save_construct_data(
            Archive& ar, Chebyshev1Function1<R> const* f, unsigned int const version);
    };


    /**
     * @brief 1-dimensional Lanczos function
     *
     * f(x) = sinc(pi x') sinc(pi x' / n)
     * where x' = x - xOffset
     * and coefficient c0 = xOffset
     *
     * Warning: the Lanczos function is sometimes forced to 0 if |x'| > n
     * but this implementation does not perform that truncation so as to improve Lanczos kernels.
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class LanczosFunction1: public Function1<ReturnT> {
    public:
        typedef typename Function1<ReturnT>::Ptr Function1Ptr;

        /**
         * @brief Construct a Lanczos function of specified order and x,y offset.
         */
        explicit LanczosFunction1(
            unsigned int n,         ///< order of Lanczos function
            double xOffset = 0.0)    ///< x offset
        :
            Function1<ReturnT>(1),
            _invN(1.0 / static_cast<double>(n))
        {
            this->_params[0] = xOffset;
        }

        virtual ~LanczosFunction1() {};
       
        virtual Function1Ptr clone() const {
            unsigned int n = static_cast<unsigned int>(0.5 + (1.0 / _invN));
            return Function1Ptr(new LanczosFunction1(n, this->_params[0]));
        }
        
        virtual ReturnT operator() (double x) const {
            double xArg1 = (x - this->_params[0]) * M_PI;
            double xArg2 = xArg1 * _invN;
            if (std::fabs(xArg1) > 1.0e-5) {
                return static_cast<ReturnT>(std::sin(xArg1) * std::sin(xArg2) / (xArg1 * xArg2));
            } else {
                return static_cast<ReturnT>(1);
            }
        }

        virtual std::string toString(void) const {
            std::ostringstream os;
            os << "LanczosFunction1 [" << _invN << "]: ";;
            os << Function1<ReturnT>::toString();
            return os.str();
        };

    private:
        double _invN;   ///< 1/n

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                LanczosFunction1<ReturnT>, Function1<ReturnT> >(
                    static_cast<LanczosFunction1<ReturnT>*>(0),
                    static_cast<Function1<ReturnT>*>(0));
#endif
        };
        template <typename R, class Archive>
        friend void boost::serialization::save_construct_data(
            Archive& ar, LanczosFunction1<R> const* f, unsigned int const version);
    };

    /**
     * @brief 2-dimensional separable Lanczos function
     *
     * f(x, y) = sinc(pi x') sinc(pi x' / n) sinc(pi y') sinc(pi y' / n)
     * where x' = x - xOffset and y' = y - yOffset
     * and coefficients c0 = xOffset, c1 = yOffset
     *
     * Warning: the Lanczos function is sometimes forced to 0 if |x'| > n or |y'| > n
     * but this implementation does not perform that truncation so as to improve Lanczos kernels.
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class LanczosFunction2: public Function2<ReturnT> {
    public:
        typedef typename Function2<ReturnT>::Ptr Function2Ptr;

        /**
         * @brief Construct a Lanczos function of specified order and x,y offset.
         */
        explicit LanczosFunction2(
            unsigned int n,         ///< order of Lanczos function
            double xOffset = 0.0,    ///< x offset
            double yOffset = 0.0)    ///< y offset
        :
            Function2<ReturnT>(2),
            _invN(1.0 / static_cast<double>(n))
        {
            this->_params[0] = xOffset;
            this->_params[1] = yOffset;
        }

        virtual ~LanczosFunction2() {};
       
        virtual Function2Ptr clone() const {
            unsigned int n = static_cast<unsigned int>(0.5 + (1.0 / _invN));
            return Function2Ptr(new LanczosFunction2(n, this->_params[0], this->_params[1]));
        }
        
        virtual ReturnT operator() (double x, double y) const {
            double xArg1 = (x - this->_params[0]) * M_PI;
            double xArg2 = xArg1 * _invN;
            double xFunc = 1;
            if (std::fabs(xArg1) > 1.0e-5) {
                xFunc = std::sin(xArg1) * std::sin(xArg2) / (xArg1 * xArg2);
            }
            double yArg1 = (y - this->_params[1]) * M_PI;
            double yArg2 = yArg1 * _invN;
            double yFunc = 1;
            if (std::fabs(yArg1) > 1.0e-5) {
                yFunc = std::sin(yArg1) * std::sin(yArg2) / (yArg1 * yArg2);
            }
            return static_cast<ReturnT>(xFunc * yFunc);
        }

        virtual std::string toString(void) const {
            std::ostringstream os;
            os << "LanczosFunction2 [" << _invN << "]: ";;
            os << Function2<ReturnT>::toString();
            return os.str();
        };

    private:
        double _invN;   ///< 1/n

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                LanczosFunction2<ReturnT>, Function2<ReturnT> >(
                    static_cast<LanczosFunction2<ReturnT>*>(0),
                    static_cast<Function2<ReturnT>*>(0));
#endif
        };
        template <typename R, class Archive>
        friend void boost::serialization::save_construct_data(
            Archive& ar, LanczosFunction2<R> const* f, unsigned int const version);
    };

}}}   // lsst::afw::math

namespace boost {
namespace serialization {

template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::IntegerDeltaFunction1<ReturnT> const* f,
                                unsigned int const version) {
    ar << make_nvp("xo", f->_xo);
};

template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::IntegerDeltaFunction2<ReturnT> const* f,
                                unsigned int const version) {
    ar << make_nvp("xo", f->_xo) << make_nvp("yo", f->_yo);
};

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::IntegerDeltaFunction2<ReturnT>* f,
                                unsigned int const version) {
    double xo;
    double yo;
    ar >> make_nvp("xo", xo) >> make_nvp("yo", yo);
    ::new(f) lsst::afw::math::IntegerDeltaFunction2<ReturnT>(xo, yo);
};

template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::GaussianFunction1<ReturnT> const* f,
                                unsigned int const version) {
    ar << make_nvp("sigma", f->getParameters()[0]);
};

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::GaussianFunction1<ReturnT>* f,
                                unsigned int const version) {
    double sigma;
    ar >> make_nvp("sigma", sigma);
    ::new(f) lsst::afw::math::GaussianFunction1<ReturnT>(sigma);
};
    
template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::GaussianFunction2<ReturnT> const* f,
                                unsigned int const version) {
    ar << make_nvp("xSigma", f->getParameters()[0]);
    ar << make_nvp("ySigma", f->getParameters()[1]);
};

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::GaussianFunction2<ReturnT>* f,
                                unsigned int const version) {
    double xSigma;
    double ySigma;
    ar >> make_nvp("xSigma", xSigma);
    ar >> make_nvp("ySigma", ySigma);
    ::new(f) lsst::afw::math::GaussianFunction2<ReturnT>(xSigma, ySigma);
};

template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::DoubleGaussianFunction2<ReturnT> const* f,
                                unsigned int const version) {
    ar << make_nvp("sigma1", f->getParameters()[0]);
    ar << make_nvp("sigma2", f->getParameters()[1]);
    ar << make_nvp("ampl2", f->getParameters()[2]);
};

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::DoubleGaussianFunction2<ReturnT>* f,
                                unsigned int const version) {
    double sigma1;
    double sigma2;
    double ampl2;
    ar >> make_nvp("sigma1", sigma1);
    ar >> make_nvp("sigma2", sigma2);
    ar >> make_nvp("ampl2", ampl2);
    ::new(f) lsst::afw::math::DoubleGaussianFunction2<ReturnT>(sigma1, sigma2, ampl2);
};
    
template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::PolynomialFunction1<ReturnT> const* f,
                                unsigned int const version) {
    ar << make_nvp("params", f->getParameters());
};

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::PolynomialFunction1<ReturnT>* f,
                                unsigned int const version) {
    std::vector<double> params;
    ar >> make_nvp("params", params);
    ::new(f) lsst::afw::math::PolynomialFunction1<ReturnT>(params);
};

template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::PolynomialFunction2<ReturnT> const* f,
                                unsigned int const version) {
    ar << make_nvp("params", f->getParameters());
};

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::PolynomialFunction2<ReturnT>* f,
                                unsigned int const version) {
    std::vector<double> params;
    ar >> make_nvp("params", params);
    ::new(f) lsst::afw::math::PolynomialFunction2<ReturnT>(params);
};
    
template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::Chebyshev1Function1<ReturnT> const* f,
                                unsigned int const version) {
    ar << make_nvp("params", f->getParameters());
    ar << make_nvp("minX", f->_minX);
    ar << make_nvp("maxX", f->_maxX);
};

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::Chebyshev1Function1<ReturnT>* f,
                                unsigned int const version) {
    std::vector<double> params;
    double minX;
    double maxX;
    ar >> make_nvp("params", params);
    ar >> make_nvp("minX", minX);
    ar >> make_nvp("maxX", maxX);
    ::new(f) lsst::afw::math::Chebyshev1Function1<ReturnT>(params, minX, maxX);
};

template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::LanczosFunction1<ReturnT> const* f,
                                unsigned int const version) {
    unsigned int n = static_cast<unsigned int>(0.5 + (1.0 / f->_invN));
    ar << make_nvp("n", n);
    ar << make_nvp("xOffset", f->getParameters()[0]);
};

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::LanczosFunction1<ReturnT>* f,
                                unsigned int const version) {
    unsigned int n;
    double xOffset;
    ar >> make_nvp("n", n);
    ar >> make_nvp("xOffset", xOffset);
    ::new(f) lsst::afw::math::LanczosFunction1<ReturnT>(n, xOffset);
};
    
template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::LanczosFunction2<ReturnT> const* f,
                                unsigned int const version) {
    unsigned int n = static_cast<unsigned int>(0.5 + (1.0 / f->_invN));
    ar << make_nvp("n", n);
    ar << make_nvp("xOffset", f->getParameters()[0]);
    ar << make_nvp("yOffset", f->getParameters()[1]);
};

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::LanczosFunction2<ReturnT>* f,
                                unsigned int const version) {
    unsigned int n;
    double xOffset;
    double yOffset;
    ar >> make_nvp("n", n);
    ar >> make_nvp("xOffset", xOffset);
    ar >> make_nvp("yOffset", yOffset);
    ::new(f) lsst::afw::math::LanczosFunction2<ReturnT>(n, xOffset, yOffset);
};

}}

#endif // #ifndef LSST_AFW_MATH_FUNCTIONLIBRARY_H
