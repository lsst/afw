// -*- LSST-C++ -*-
#ifndef LSST_FW_FunctionLibrary_H
#define LSST_FW_FunctionLibrary_H
/**
 * \file
 *
 * \brief Define a collection of useful Functions.
 *
 * To do:
 * - Add 2-d Chebyshev polynomial
 * - Add a function factory?
 *
 * \author Russell Owen
 *
 * \ingroup fw
 */
#include <cmath>

#include "Function.h"

namespace lsst {
namespace fw {
namespace function {

    /**
     * \brief 2-dimensional integer delta function.
     *
     * f(x) = 1 if x == xo and y == yo, 0 otherwise.
     *
     * For use as a kernel function be sure to handle the offset for row and column center;
     * see examples/deltaFunctionKernel for an example.
     *
     * \ingroup fw
     */
    template<typename ReturnT>
    class IntegerDeltaFunction2: public Function2<ReturnT> {
    public:
        /**
         * \brief Construct an integer delta function with specified xo, yo
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
        
        virtual ReturnT operator() (double x, double y) const {
            return static_cast<ReturnT>((x == _xo) && (y == _yo));
        }
    private:
        double _xo;
        double _yo;
    };


    /**
     * \brief 1-dimensional Gaussian
     *
     * f(x) = e^(-x^2 / sigma^2) / (sqrt(2 pi) xSigma)
     * with coefficient c0 = sigma
     *
     * \ingroup fw
     */
    template<typename ReturnT>
    class GaussianFunction1: public Function1<ReturnT> {
    public:
        /**
         * \brief Construct a Gaussian function with specified sigma
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
        
        virtual ReturnT operator() (double x) const {
            return (_multFac / this->_params[0]) *
                std::exp(- (x * x) / (this->_params[0] * this->_params[0]));
        }
        
    private:
        const double _multFac; ///< precomputed scale factor
    };
    
    
    /**
     * \brief 2-dimensional Gaussian
     *
     * f(x,y) = e^(-x^2 / xSigma^2) e^(-y^2 / ySigma^2) / (2 pi xSigma ySigma)
     * with coefficients c0 = xSigma and c1 = ySigma
     *
     * To do:
     * - Allow setting angle of ellipticity
     * - Perhaps recast as a separable pair of 1-d Guassians
     *
     * \ingroup fw
     */
    template<typename ReturnT>
    class GaussianFunction2: public Function2<ReturnT> {
    public:
        /**
         * \brief Construct a Gaussian function with specified x and y sigma
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
        
        virtual ReturnT operator() (double x, double y) const {
            return (_multFac / (this->_params[0] * this->_params[1])) *
                std::exp(- ((x * x) / (this->_params[0] * this->_params[0]))
                         - ((y * y) / (this->_params[1] * this->_params[1]))
                );
        }
        
    private:
        const double _multFac; ///< precomputed scale factor
    };
    
    
    /**
     * \brief 1-dimensional polynomial function.
     *
     * f(x) = c0 + c1 x + c2 x^2 + ... cn-1 x^(n-1)
     *
     * \ingroup fw
     */
    template<typename ReturnT>
    class PolynomialFunction1: public Function1<ReturnT> {
    public:
        /**
         * \brief Construct a polynomial function of the specified order.
         *
         * The parameters are initialized to zero.
         */
        explicit PolynomialFunction1(
            unsigned int order)     ///< order of polynomial (0 for constant)
        :
            Function1<ReturnT>(order+1) {
        }
        
        /**
         * \brief Construct a polynomial function with the specified parameters.
         *
         * The order of the polynomial is set to the length of the params vector.
         *
         * \throw std::invalid_argument if params is empty
         */
        explicit PolynomialFunction1(
            std::vector<double> params)  ///< polynomial coefficients (const, x, x^2...)
        :
            Function1<ReturnT>(params)
        {
            if (params.size() < 1) {
                throw std::invalid_argument("PolynomialFunction1 called with empty vector");
            }
        }
        
        virtual ~PolynomialFunction1() {};
        
        virtual ReturnT operator() (double x) const {
            int maxInd = static_cast<int>(this->_params.size()) - 1;
            double retVal = this->_params[maxInd];
            for (int ii = maxInd-1; ii >= 0; --ii) {
                retVal = (retVal * x) + this->_params[ii];
            }
            return static_cast<ReturnT>(retVal);
        }
    };


    /**
     * \brief 2-dimensional polynomial function.
     *
     * f(x,y) = c0                                          (0th order)
     *          + c1 x + c2 y                               (1st order)
     *          + c2 x^2 + c3 x y + c4 y^2                  (2nd order)
     *          + c5 x^3 + c6 x^2 y + c7 x y^2 + c8 y^3     (3rd order)
     *          + ...
     *
     * \ingroup fw
     */
    template<typename ReturnT>
    class PolynomialFunction2: public Function2<ReturnT> {
    public:
        /**
         * \brief Construct a polynomial function of specified order.
         *
         * The parameters are initialized to zero.
         */
        explicit PolynomialFunction2(
            unsigned int order) ///< order of polynomial (0 for constant)
        :
            Function2<ReturnT>((order+1)*(order+2)/2),
            _order(order)
        {}

        /**
         * \brief Construct a polynomial function with specified parameters.
         *
         * The order of the polynomial is set to the length of the params vector.
         *
         * \throw std::invalid_argument if params length is unsuitable
         */
        explicit PolynomialFunction2(
            std::vector<double> params)  ///< polynomial coefficients (const, x, y, x^2, xy, y^2...);
                                    ///< length must be one of 1, 3, 6, 10, 15...
        :
            Function2<ReturnT>(params),
            _order((-3 + std::sqrt(1 + (8 * params.size()))) / 2)
        {
            unsigned int nParams = params.size();
            if (nParams < 1) {
                throw std::invalid_argument("PolynomialFunction2 created with empty vector");
            }
            if (nParams != ((_order + 1) * (_order + 2)) / 2) {
                throw std::invalid_argument("PolynomialFunction2 created with vector of unusable length");
            }
        }
        
        virtual ~PolynomialFunction2() {};
        
        virtual ReturnT operator() (double x, double y) const {
            // there must be a more efficient way to solve this
            int paramInd = static_cast<int>(this->_params.size()) - 1;
            int currOrder = static_cast<int>(_order);
            double retVal = this->_params[0];
            while (paramInd > 0) {
                for (int xPower = 0, yPower = currOrder; xPower <= currOrder; ++xPower, --yPower) {
                    retVal += std::pow(x, xPower) * std::pow(y, yPower) * this->_params[paramInd];
                    --paramInd;
                }
            }
            return static_cast<ReturnT>(retVal);
        }
    private:
        unsigned int _order; ///< order of polynomial
    };

    
    /**
     * \brief 1-dimensional weighted sum of Chebyshev polynomials of the first kind.
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
     * \ingroup fw
     */
    template<typename ReturnT>
    class Chebyshev1Function1: public Function1<ReturnT> {
    public:
        /**
         * \brief Construct a Chebyshev polynomial of specified order and range.
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
         * \brief Construct a Chebyshev polynomial with specified parameters and range.
         *
         * The order of the polynomial is set to the length of the params vector.
         *
         * \throw std::invalid_argument if params is empty
         */
        explicit Chebyshev1Function1(
            std::vector<double> params,  ///< polynomial coefficients
            double xMin = -1,    ///< minimum allowed x
            double xMax = 1)     ///< maximum allowed x
        :
            Function1<ReturnT>(params)
        {
            if (params.size() < 1) {
                throw std::invalid_argument("Chebyshev1Function1 called with empty vector");
            }
            _initialize(xMin, xMax);
        }
        
        virtual ~Chebyshev1Function1() {};
        
        virtual ReturnT operator() (double x) const {
            double xPrime = (x * _scale) + _offset;
            return static_cast<ReturnT>(_clenshaw(xPrime, 0));
        }
    private:
        double _minX;    ///< minimum allowed x
        double _maxX;    ///< maximum allowed x
        double _scale;   ///< x' = (x * _scale) + _offset
        double _offset;  ///< x' = (x * _scale) + _offset
        unsigned int _maxInd;   ///< maximum index for Clenshaw function
        
        /**
         * \brief Clenshaw recursive function for solving the Chebyshev polynomial
         */
        double _clenshaw(double x, unsigned int ind) const {
            if (ind == 0) {
                return (x * _clenshaw(x, 1)) + this->_params[0] - _clenshaw(x, 2);
            } else if (ind == _maxInd) {
                return this->_params[ind];
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
         * \brief initialize private constants
         */
        void _initialize(double xMin, double xMax) {
            _minX = xMin;
            _maxX = xMax;
            _scale = 2 / (xMax - xMin);
            _offset = -1 - (xMin * _scale);
            _maxInd = this->getNParameters() - 1;
        }
    };


    /**
     * \brief 1-dimensional Lanczos function
     *
     * f(x) = sinc(pi x') sinc(pi x' / n)
     * where x' = x - xOffset
     * and coefficient c0 = xOffset
     *
     * Warning: the Lanczos function is sometimes forced to 0 if |x'| > n
     * but this implementation does not perform that truncation so as to improve Lanczos kernels.
     *
     * \ingroup fw
     */
    template<typename ReturnT>
    class LanczosFunction1: public Function1<ReturnT> {
    public:
        /**
         * \brief Construct a Lanczos function of specified order and x,y offset.
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
        
        virtual ReturnT operator() (double x) const {
            double xArg1 = (x - this->_params[0]) * M_PI;
            double xArg2 = xArg1 * _invN;
            if (abs(xArg1) > 1.0e-5) {
                return static_cast<ReturnT>(std::sin(xArg1) * std::sin(xArg2) / (xArg1 * xArg2));
            } else {
                return static_cast<ReturnT>(1);
            }
        }
    private:
        double _invN;   ///< 1/n
    };


    /**
     * \brief 2-dimensional radial Lanczos function
     *
     * f(rad) = sinc(pi rad) sinc(pi rad / n)
     * where rad = sqrt((x - xOffset)^2 + (y - yOffset)^2)
     * and coefficients c0 = xOffset, c1 = yOffset
     *
     * Warning: the Lanczos function is sometimes forced to 0 if radius > n
     * but this implementation does not perform that truncation so as to improve Lanczos kernels.
     *
     * \ingroup fw
     */
    template<typename ReturnT>
    class LanczosFunction2: public Function2<ReturnT> {
    public:
        /**
         * \brief Construct a Lanczos function of specified order and x,y offset.
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
        
        virtual ReturnT operator() (double x, double y) const {
            double rad = std::sqrt(((x - this->_params[0]) * (x - this->_params[0]))
                + ((y - this->_params[1]) * (y - this->_params[1])));
            double arg1 = rad * M_PI;
            double arg2 = arg1 * _invN;
            if (abs(rad) > 1.0e-5) {
                return static_cast<ReturnT>(std::sin(arg1) * std::sin(arg2) / (arg1 * arg2));
            } else {
                return static_cast<ReturnT>(1);
            }
        }
    private:
        double _invN;   ///< 1/n
    };


    /**
     * \brief 2-dimensional separable Lanczos function
     *
     * f(x, y) = sinc(pi x') sinc(pi x' / n) sinc(pi y') sinc(pi y' / n)
     * where x' = x - xOffset and y' = y - yOffset
     * and coefficients c0 = xOffset, c1 = yOffset
     *
     * Warning: the Lanczos function is sometimes forced to 0 if |x'| > n or |y'| > n
     * but this implementation does not perform that truncation so as to improve Lanczos kernels.
     *
     * \ingroup fw
     */
    template<typename ReturnT>
    class LanczosSeparableFunction2: public Function2<ReturnT> {
    public:
        /**
         * \brief Construct a Lanczos function of specified order and x,y offset.
         */
        explicit LanczosSeparableFunction2(
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

        virtual ~LanczosSeparableFunction2() {};
        
        virtual ReturnT operator() (double x, double y) const {
            double xArg1 = (x - this->_params[0]) * M_PI;
            double xArg2 = xArg1 * _invN;
            double xFunc = 1;
            if (abs(xArg1) > 1.0e-5) {
                xFunc = std::sin(xArg1) * std::sin(xArg2) / (xArg1 * xArg2);
            }
            double yArg1 = (y - this->_params[1]) * M_PI;
            double yArg2 = yArg1 * _invN;
            double yFunc = 1;
            if (abs(yArg1) > 1.0e-5) {
                yFunc = std::sin(yArg1) * std::sin(yArg2) / (yArg1 * yArg2);
            }
            return static_cast<ReturnT>(xFunc * yFunc);
        }
    private:
        double _invN;   ///< 1/n
    };

}   // namespace functions
}   // namespace fw
}   // namespace lsst

#endif // #ifndef LSST_FW_FunctionLibrary_H
