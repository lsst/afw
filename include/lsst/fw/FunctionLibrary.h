// -*- LSST-C++ -*-
#ifndef LLST_FW_FunctionLibrary_H
#define LLST_FW_FunctionLibrary_H
/**
 * \file
 * \ingroup fw
 *
 * Library of useful functions.
 *
 * To do:
 * - Add a function factory
 * - Add a Lanczos function
 * - Separate implementation from declaration
 *
 * \author Russell Owen
 */
#include <cmath>

#include "Function.h"

namespace lsst {
namespace fw {

    /**
     * 1-dimensional Gaussian with peak value "ampl"
     */
    template<class T>
    class GaussianFunction1: public Function1<T>
    {
    public:
        GaussianFunction1(T ampl, T sigma) : Function1<T>(2) {
            this->_params[0] = ampl;    ///< amplitude (peak value)
            this->_params[1] = sigma;   ///< sigma (width)
        }
        virtual ~GaussianFunction1() {};
        
        virtual T operator() (T x) const {
            return this->_params[0] * std::exp(- (x * x) / (this->_params[1] * this->_params[1]));
        }
    };
    
    
    /**
     * 2-dimensional Gaussian with peak value "ampl"
     *
     * To do:
     * - Allow setting angle of ellipticity
     * - Perhaps recast as a separable pair of 1-d guassians (but then what to do about amplitude?)
     */
    template<class T>
    class GaussianFunction2: public Function2<T>
    {
    public:
        GaussianFunction2(T ampl, T xSigma, T ySigma) : Function2<T>(3) {
            this->_params[0] = ampl;    ///< amplitude (peak value)
            this->_params[1] = xSigma;  ///< x sigma (width)
            this->_params[2] = ySigma;  ///< y sigma (width)
        }
        virtual ~GaussianFunction2() {};
        
        virtual T operator() (T x, T y) const {
            return this->_params[0] * std::exp(- (x * x) / (this->_params[1] * this->_params[1]))
                                    * std::exp(- (y * y) / (this->_params[2] * this->_params[2]));
        }
    };
    
    
    /**
     * 1-dimensional polynomial function.
     */
    template<class T>
    class PolynomialFunction1: public Function1<T>
    {
    public:
        PolynomialFunction1(unsigned order) : Function1<T>(order) {};
        PolynomialFunction1(std::vector<T> params) : Function1<T>(params) {};
        virtual ~PolynomialFunction1() {};
        
        virtual T operator() (T x) const {
            T retVal = 0;
            for (long ii = this->_params.size() - 1; ii >= 0; ii--) {
                // sum from high order terms down to reduce roundoff error
                retVal += this->_params[ii] * std::pow(x, static_cast<T> (ii));
            }
            return retVal;
        }
    };
    
    
    /**
     * 1-dimensional weighted sum of Chebyshev polynomials of the first kind.
     * The function argument must be in the range [minArg, maxArg].
     */
    template<class T>
    class Chebyshev1Function1: public Function1<T>
    {
    public:
        Chebyshev1Function1(
            unsigned order,
            T minArg = -1.0,
            T maxArg = 1.0
        ) :
            Function1<T>(order)
        {
            _initialize(minArg, maxArg);
        }
        Chebyshev1Function1(
            std::vector<T> params,
            T minArg = -1.0,
            T maxArg = 1.0
        ) :
            Function1<T>(params)
        {
            _initialize(minArg, maxArg);
        }
        virtual ~Chebyshev1Function1() {};
        
        virtual T operator() (T x) const {
            T retVal = 0;
            T arg = (x * _scale) + _offset;
            for (long ii = 0; ii < this->_params.size(); ii++) {
                retVal += this->_params[ii] * std::cos(static_cast<T>(ii) * std::acos(arg));
            }
            return retVal;
        }
    private:
        T _minArg;
        T _maxArg;
        T _scale;
        T _offset;
        
        void _initialize(T minArg, T maxArg) {
            _minArg = minArg;
            _maxArg = maxArg;
            _scale = 2 / (maxArg - minArg);
            _offset = -1 - (minArg * _scale);
        }
    };
    
}   // namespace fw
}   // namespace lsst

#endif // #ifndef LLST_FW_FunctionLibrary_H
