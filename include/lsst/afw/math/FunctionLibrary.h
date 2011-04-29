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
#include <algorithm>
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

        virtual std::string toString(std::string const& prefix="") const {
            std::ostringstream os;
            os << "IntegerDeltaFunction1 [" << _xo << "]: ";
            os << Function1<ReturnT>::toString(prefix);
            return os.str();
        };

    private:
        double _xo;

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive&, unsigned int const) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                IntegerDeltaFunction1<ReturnT>, Function1<ReturnT> >(
                    static_cast<IntegerDeltaFunction1<ReturnT>*>(0),
                    static_cast<Function1<ReturnT>*>(0));
#endif
        }
        template <typename R, class Archive>
        friend void boost::serialization::save_construct_data(
            Archive& ar, IntegerDeltaFunction1<R> const* f, unsigned int const);
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
        
        virtual ~IntegerDeltaFunction2() {}
        
        virtual Function2Ptr clone() const {
            return Function2Ptr(new IntegerDeltaFunction2(_xo, _yo));
        }
        
        virtual ReturnT operator() (double x, double y) const {
            return static_cast<ReturnT>((x == _xo) && (y == _yo));
        }

        virtual std::string toString(std::string const& prefix) const {
            std::ostringstream os;
            os << "IntegerDeltaFunction2 [" << _xo << ", " << _yo << "]: ";
            os << Function2<ReturnT>::toString(prefix);
            return os.str();
        }

    private:
        double _xo;
        double _yo;

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive&, unsigned int const) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                IntegerDeltaFunction2<ReturnT>, Function2<ReturnT> >(
                    static_cast<IntegerDeltaFunction2<ReturnT>*>(0),
                    static_cast<Function2<ReturnT>*>(0));
#endif
        }
        template <typename R, class Archive>
        friend void boost::serialization::save_construct_data(
            Archive& ar, IntegerDeltaFunction2<R> const* f, unsigned int const);
    };

    /**
     * @brief 1-dimensional Gaussian
     *
     * f(x) = A e^(-x^2 / 2 sigma^2)
     * where:
     * * A = 1 / (sqrt(2 pi) xSigma)
     * coefficient c0 = sigma
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
        virtual ~GaussianFunction1() {}
        
        virtual Function1Ptr clone() const {
            return Function1Ptr(new GaussianFunction1(this->_params[0]));
        }
        
        virtual ReturnT operator() (double x) const {
            return static_cast<ReturnT> (
                (_multFac / this->_params[0]) *
                std::exp(- (x * x) / (2.0 * this->_params[0] * this->_params[0])));
        }
        
        virtual std::string toString(std::string const& prefix) const {
            std::ostringstream os;
            os << "GaussianFunction1 [" << _multFac << "]: ";
            os << Function1<ReturnT>::toString(prefix);
            return os.str();
        }

    private:
        const double _multFac; ///< precomputed scale factor

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive&, unsigned int const) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                GaussianFunction1<ReturnT>, Function1<ReturnT> >(
                    static_cast<GaussianFunction1<ReturnT>*>(0),
                    static_cast<Function1<ReturnT>*>(0));
#endif
        }
    };
    
    /**
     * @brief 2-dimensional Gaussian
     *
     * f(x,y) = A e^((-pos1^2 / 2 sigma1^2) - (pos2^2 / 2 sigma2^2))
     * where:
     * * A = 1 / (2 pi sigma1 sigma2)
     * * pos1 =  cos(angle) x + sin(angle) y
     * * pos2 = -sin(angle) x + cos(angle) y
     * coefficients c0 = sigma1, c1 = sigma2, c2 = angle
     *
     * @note if sigma1 > sigma2 then angle is the angle of the major axis
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class GaussianFunction2: public Function2<ReturnT> {
    public:
        typedef typename Function2<ReturnT>::Ptr Function2Ptr;

        /**
         * @brief Construct a 2-dimensional Gaussian function
         */
        explicit GaussianFunction2(
            double sigma1,      ///< sigma along the pos1 axis
            double sigma2,      ///< sigma along the pos2 axis
            double angle = 0.0) ///< angle of pos1 axis, in rad (along x=0, y=pi/2)
        : 
            Function2<ReturnT>(3),
            _multFac(1.0 / (2.0 * M_PI))
        {
            this->_params[0] = sigma1;
            this->_params[1] = sigma2;
            this->_params[2] = angle;
            _updateCache();
        }
        
        virtual ~GaussianFunction2() {}
        
        virtual Function2Ptr clone() const {
            return Function2Ptr(new GaussianFunction2(this->_params[0], this->_params[1], this->_params[2]));
        }
        
        virtual ReturnT operator() (double x, double y) const {
            if (_angle != this->_params[2]) {
                _updateCache();
            }
            double pos1 = ( _cosAngle * x) + (_sinAngle * y);
            double pos2 = (-_sinAngle * x) + (_cosAngle * y);
            return static_cast<ReturnT> (
                (_multFac / (this->_params[0] * this->_params[1])) *
                std::exp(- ((pos1 * pos1) / (2.0 * this->_params[0] * this->_params[0]))
                         - ((pos2 * pos2) / (2.0 * this->_params[1] * this->_params[1]))));
        }
        
        virtual std::string toString(std::string const& prefix) const {
            std::ostringstream os;
            os << "GaussianFunction2: ";
            os << Function2<ReturnT>::toString(prefix);
            return os.str();
        }
    
    private:
        /**
        * @brief Update cached values
        *
        * sin(angle) and cos(angle) are cached to speed computation
        * and angle is cached so one can check if an update is required
        *
        * The current design is to have operator() update the cache if needed.
        * An alternate design is to update the cache when the parameters are set,
        * not test in operator().
        * The main advantage to updating in operator() is safety and simplicity.
        * The test is performed in one place, and it is the place where it matters the most.
        * In contrast, there are multiple member functions to set parameters, and all must be overloaded
        * to update the cache; miss one and the function silently misbehaves.
        * There are trade-offs, of course. Testing the cache in operator() slows down operator() slightly.
        * The overhead is small, but the function is typically evaulated more often
        * than its parameters are changed.
        */
        void _updateCache() const {
            _angle = this->_params[2];
            _sinAngle = std::sin(_angle);
            _cosAngle = std::cos(_angle);
        }
        const double _multFac;  ///< precomputed scale factor
        mutable double _angle;    ///< cached angle
        mutable double _sinAngle; ///< cached sin(angle)
        mutable double _cosAngle; ///< cached cos(angle)

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive&, unsigned int const) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                GaussianFunction2<ReturnT>, Function2<ReturnT> >(
                    static_cast<GaussianFunction2<ReturnT>*>(0),
                    static_cast<Function2<ReturnT>*>(0));
#endif
        }
    };
    
    /**
     * @brief double Guassian (sum of two Gaussians)
     *
     * Intended for use as a PSF model: the main Gaussian represents the core
     * and the second Gaussian represents the wings.
     *
     * f(x,y) = A (e^(-r^2 / 2 sigma1^2) + ampl2 e^(-r^2 / 2 sigma2^2))
     * where:
     * * A = 1 / (2 pi (sigma1^2 + ampl2 sigma2^2))
     * * r^2 = x^2 + y^2
     * coefficients c[0] = sigma1, c[1] = sigma2, c[2] = ampl2
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
        
        virtual ~DoubleGaussianFunction2() {}
        
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
        
        virtual std::string toString(std::string const& prefix) const {
            std::ostringstream os;
            os << "DoubleGaussianFunction2 [" << _multFac << "]: ";
            os << Function2<ReturnT>::toString(prefix);
            return os.str();
        }

    private:
        const double _multFac; ///< precomputed scale factor

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive&, unsigned int const) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                DoubleGaussianFunction2<ReturnT>, Function2<ReturnT> >(
                    static_cast<DoubleGaussianFunction2<ReturnT>*>(0),
                    static_cast<Function2<ReturnT>*>(0));
#endif
        }
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
        
        virtual ~PolynomialFunction1() {}
       
        virtual Function1Ptr clone() const {
            return Function1Ptr(new PolynomialFunction1(this->_params));
        }

        virtual bool isLinearCombination() const { return true; };
        
        virtual ReturnT operator() (double x) const {
            int maxInd = static_cast<int>(this->_params.size()) - 1;
            double retVal = this->_params[maxInd];
            for (int ii = maxInd-1; ii >= 0; --ii) {
                retVal = (retVal * x) + this->_params[ii];
            }
            return static_cast<ReturnT>(retVal);
        }

        virtual std::string toString(std::string const& prefix) const {
            std::ostringstream os;
            os << "PolynomialFunction1 []: ";
            os << Function1<ReturnT>::toString(prefix);
            return os.str();
        }


    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive&, unsigned int const) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                PolynomialFunction1<ReturnT>, Function1<ReturnT> >(
                    static_cast<PolynomialFunction1<ReturnT>*>(0),
                    static_cast<Function1<ReturnT>*>(0));
#endif
        }
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
    class PolynomialFunction2: public BasePolynomialFunction2<ReturnT> {
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
            BasePolynomialFunction2<ReturnT>(order),
            _yCoeffs(this->_order + 1)
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
            BasePolynomialFunction2<ReturnT>(params),
            _yCoeffs(this->_order + 1)
        {}
        
        virtual ~PolynomialFunction2() {}
       
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
              ...

            First compute Cy0, Cy1...Cyn by solving 1-d polynomials in x in the usual way.
            Then compute f(x,y) by solving the 1-d polynomial in y in the usual way.
            */
            const int maxYCoeffInd = this->_order;
            int paramInd = static_cast<int>(this->_params.size()) - 1;
            // initialize the y coefficients
            for (int yCoeffInd = maxYCoeffInd; yCoeffInd >= 0; --yCoeffInd, --paramInd) {
                _yCoeffs[yCoeffInd] = this->_params[paramInd];
            }
            // finish computing the y coefficients
            for (int startYCoeffInd = maxYCoeffInd - 1, yCoeffInd = startYCoeffInd;
                paramInd >= 0; --paramInd) {
                _yCoeffs[yCoeffInd] = (_yCoeffs[yCoeffInd] * x) + this->_params[paramInd];
                if (yCoeffInd == 0) {
                    --startYCoeffInd;
                    yCoeffInd = startYCoeffInd;
                } else {
                    --yCoeffInd;
                }
            }
            // compute y polynomial
            double retVal = _yCoeffs[maxYCoeffInd];
            for (int yCoeffInd = maxYCoeffInd - 1; yCoeffInd >= 0; --yCoeffInd) {
                retVal = (retVal * y) + _yCoeffs[yCoeffInd];
            }
            return static_cast<ReturnT>(retVal);
        }

        virtual std::vector<double> getDFuncDParameters(double x, double y) const;

        virtual std::string toString(std::string const& prefix) const {
            std::ostringstream os;
            os << "PolynomialFunction2 [" << this->_order << "]: ";
            os << Function2<ReturnT>::toString(prefix);
            return os.str();
        }

    private:
        mutable std::vector<double> _yCoeffs; ///< working vector

        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const) {
            ar & make_nvp("fn2",
                          boost::serialization::base_object<
                          BasePolynomialFunction2<ReturnT> >(*this));
            ar & make_nvp("order", this->_order);
        }
    };
    
    /**
     * @brief 1-dimensional weighted sum of Chebyshev polynomials of the first kind.
     *
     * f(x) = c0 + c1 * T1(x') + c2 * T2(x') + ...
     * where:
     *   Tn(x) is the nth Chebyshev function of the first kind:
     *     Tn(x) = cos(n arccos(x))
     *   x' is x offset and scaled to range [-1, 1] as x ranges over [minX, maxX]
     *
     * The function argument must be in the range [minX, maxX].
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
            double minX = -1,    ///< minimum allowed x
            double maxX = 1)     ///< maximum allowed x
        :
            Function1<ReturnT>(order + 1)
        {
            _initialize(minX, maxX);
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
            double minX = -1,    ///< minimum allowed x
            double maxX = 1)     ///< maximum allowed x
        :
            Function1<ReturnT>(params)
        {
            if (params.size() < 1) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                                  "Chebyshev1Function1 called with empty vector");
            }
            _initialize(minX, maxX);
        }
        
        virtual ~Chebyshev1Function1() {}
       
        virtual Function1Ptr clone() const {
            return Function1Ptr(new Chebyshev1Function1(this->_params, _minX, _maxX));
        }

        int getMinX() const { return _minX; };
        int getMaxX() const { return _maxX; };

        virtual bool isLinearCombination() const { return true; };
        
        virtual ReturnT operator() (double x) const {
            double xPrime = (x + _offset) * _scale;
            return static_cast<ReturnT>(_clenshaw(xPrime, 0));
        }

        virtual std::string toString(std::string const& prefix) const {
            std::ostringstream os;
            os << "Chebyshev1Function1 [" << _minX << ", " << _maxX << "]: ";
            os << Function1<ReturnT>::toString(prefix);
            return os.str();
        }

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
        void _initialize(double minX, double maxX) {
            _minX = minX;
            _maxX = maxX;
            _scale = 2 / (maxX - minX);
            _offset = -(minX + maxX) / 2.0;
            _maxInd = this->getNParameters() - 1;
        }

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive&, unsigned int const) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                Chebyshev1Function1<ReturnT>, Function1<ReturnT> >(
                    static_cast<Chebyshev1Function1<ReturnT>*>(0),
                    static_cast<Function1<ReturnT>*>(0));
#endif
        }
        template <typename R, class Archive>
        friend void boost::serialization::save_construct_data(
            Archive& ar, Chebyshev1Function1<R> const* f, unsigned int const);
    };

    /**
     * @brief 2-dimensional weighted sum of Chebyshev polynomials of the first kind.
     *
     * f(x,y) = c0
     *        + c1 * T1(x') + c2 * T1(y')
     *        + c3 * T2(x') + c4 * T1(x') * T1(y') + c5 * T2(y')
     *        + ...
     * where:
     *   Tn(x) is the nth Chebyshev function of the first kind:
     *     Tn(x) = cos(n arccos(x))
     *   x' is x offset and scaled to range [-1, 1] as x ranges over [minX, maxX]
     *   y' is y offset and scaled to range [-1, 1] as y ranges over [minY, maxY]
     *
     * Return value is incorrect if function arguments are not in the range [minX, maxX], [minY, maxY].
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class Chebyshev1Function2: public BasePolynomialFunction2<ReturnT> {
    public:
        typedef typename Function2<ReturnT>::Ptr Function2Ptr;

        /**
         * @brief Construct a Chebyshev polynomial of specified order and range.
         *
         * The parameters are initialized to zero.
         */
        explicit Chebyshev1Function2(
            unsigned int order, ///< order of polynomial (0 for constant)
            double minX = -1,   ///< minimum allowed x
            double minY = -1,   ///< minimum allowed y
            double maxX = 1,    ///< maximum allowed x
            double maxY = 1)    ///< maximum allowed y
        :
            BasePolynomialFunction2<ReturnT>(order),
            _xCheby(this->_order + 1),
            _yCoeffs(this->_order + 1)
        {
            _initialize(minX, maxX, minY, maxY);
        }

        /**
         * @brief Construct a Chebyshev polynomial with specified parameters and range.
         *
         * The order of the polynomial is set to the length of the params vector.
         *
         * @throw lsst::pex::exceptions::InvalidParameterException if params is empty
         */
        explicit Chebyshev1Function2(
            std::vector<double> params,
                ///< polynomial coefficients (const, T1(x), T1(y), T2(x), T1(x) T1(y), T2(y)...)
                ///< length must be one of 1, 3, 6, 10, 15...
            double minX = -1,    ///< minimum allowed x
            double minY = -1,    ///< minimum allowed y
            double maxX = 1,     ///< maximum allowed x
            double maxY = 1)     ///< maximum allowed y
        :
            BasePolynomialFunction2<ReturnT>(params),
            _xCheby(this->_order + 1),
            _yCoeffs(this->_order + 1)
        {
            _initialize(minX, maxX, minY, maxY);
        }
        
        virtual ~Chebyshev1Function2() {}
       
        virtual Function2Ptr clone() const {
            return Function2Ptr(new Chebyshev1Function2(this->_params, _minX, _maxX));
        }
        
        int getMinX() const { return _minX; };
        int getMinY() const { return _minY; };
        int getMaxX() const { return _maxX; };
        int getMaxY() const { return _maxY; };
        
        /**
         * @brief Return a truncated copy of lower (or equal) order
         *
         * @throw lsst::pex::exceptions::InvalidParameter if truncated order > original order
         */
        virtual Chebyshev1Function2 truncate(
                int truncOrder ///< order of truncated polynomial
        ) {
            if (truncOrder > this->_order) {
                std::ostringstream os;
                os << "truncated order=" << truncOrder << " must be <= original order=" << this->_order;
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, os.str());
            }
            int truncNParams = this->nParametersFromOrder(truncOrder);
            std::vector<double> truncParams(this->_params.begin(), this->_params.begin() + truncNParams);
            return Chebyshev1Function2(truncParams, _minX, _minY, _maxX, _maxY);

        }

        virtual ReturnT operator() (double x, double y) const {
            /* Solve as follows:
            - f(x,y) = Cy0 T0(y') + Cy1 T1(y') + Cy2 T2(y') + Cy3 T3(y') + ...
            where:
              Cy0 = P0 T0(x') + P1 T1(x') + P3 T2(x') + P6 T3(x') + ...
              Cy1 = P2 T0(x') + P4 T1(x') + P7 T2(x') + ...
              Cy2 = P5 T0(x') + P8 T1(x') + ...
              Cy3 = P9 T0(x') + ...
              ...
            
            First compute Tn(x') for each n
            Then use that to compute Cy0, Cy1, ...Cyn
            Then solve the y Chebyshev polynomial using the Clenshaw algorithm
            */
            double const xPrime = (x + _offsetX) * _scaleX;
            double const yPrime = (y + _offsetY) * _scaleY;

            // Compute _xCheby[i] = Ti(x') using the standard recurrence relationship;
            // note that _initialize already set _xCheby[0] = 1.
            if (this->_order > 0) {
                _xCheby[1] = xPrime;
            }
            for (int xInd = 2; xInd <= this->_order; ++xInd) {
                _xCheby[xInd] = (2 * xPrime * _xCheby[xInd-1]) - _xCheby[xInd-2];
            }
            
            // Initialize _yCoeffs to right-hand terms of equation shown in documentation block
            int paramInd = static_cast<int>(this->_params.size()) - 1;
            for (int yCoeffInd = this->_order, xChebyInd = 0; yCoeffInd >= 0;
                --yCoeffInd, ++xChebyInd, --paramInd) {
                _yCoeffs[yCoeffInd] = this->_params[paramInd] * _xCheby[xChebyInd];
            }
            // Add the remaining terms to _yCoeffs (starting from _order-1 because _yCoeffs[_order] is done)
            for (int startYCoeffInd = this->_order - 1, yCoeffInd = startYCoeffInd, xChebyInd = 0;
                paramInd >= 0; --paramInd) {
                _yCoeffs[yCoeffInd] += this->_params[paramInd] * _xCheby[xChebyInd];
                if (yCoeffInd == 0) {
                    --startYCoeffInd;
                    yCoeffInd = startYCoeffInd;
                    xChebyInd = 0;
                } else {
                    --yCoeffInd;
                    ++xChebyInd;
                }
            }
            
            // Compute result using Clenshaw algorithm for the polynomial in y
            return static_cast<ReturnT>(_clenshaw(yPrime, 0));
        }

        virtual std::string toString(std::string const& prefix) const {
            std::ostringstream os;
            os << "Chebyshev1Function2 [";
            os << this->_order << ", " << _minX << ", " << _minY << ", " << _maxX << ", "<< _maxY << "]";
            os << Function2<ReturnT>::toString(prefix);
            return os.str();
        }

    private:
        mutable std::vector<double> _xCheby; ///< working vector: value of T_n(x')
        mutable std::vector<double> _yCoeffs; ///< working vector: transformed coeffs of Y polynomial
        double _minX;    ///< minimum allowed x
        double _minY;    ///< minimum allowed y
        double _maxX;    ///< maximum allowed x
        double _maxY;    ///< maximum allowed y
        double _scaleX;   ///< x' = (x + _offsetX) * _scaleX
        double _scaleY;   ///< y' = (y + _offsetY) * _scaleY
        double _offsetX;  ///< x' = (x + _offsetX) * _scaleX
        double _offsetY;  ///< y' = (y + _offsetY) * _scaleY
        
        /**
         * @brief Clenshaw recursive function for solving the Chebyshev polynomial
         */
        double _clenshaw(double y, unsigned int ind) const {
            if (ind == this->_order) {
                return _yCoeffs[ind];
            } else if (ind == 0) {
                return (y * _clenshaw(y, 1)) + _yCoeffs[0] - _clenshaw(y, 2);
            } else if (ind == this->_order - 1) {
                return (2 * y * _clenshaw(y, ind+1)) + _yCoeffs[ind];
            } else if (ind < this->_order) {
                return (2 * y * _clenshaw(y, ind+1)) + _yCoeffs[ind] - _clenshaw(y, ind+2);
            } else {
                // this case only occurs if _order < 3
                return 0;
            }
        }
        
        /**
         * @brief initialize private constants
         */
        void _initialize(double minX, double maxX, double minY, double maxY) {
            _minX = minX;
            _minY = minY;
            _maxX = maxX;
            _maxY = maxY;
            _scaleX = 2 / (maxX - minX);
            _scaleY = 2 / (maxY - minY);
            _offsetX = -(minX + maxX) / 2.0;
            _offsetY = -(minY + maxY) / 2.0;
            _xCheby[0] = 1.0;
        }

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const) {
            ar & make_nvp("fn2",
                          boost::serialization::base_object<
                          BasePolynomialFunction2<ReturnT> >(*this));
            ar & make_nvp("order", this->_order);
        }
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
            _n(n),
            _invN(1.0/static_cast<double>(n))
        {
            this->_params[0] = xOffset;
        }

        virtual ~LanczosFunction1() {}
       
        virtual Function1Ptr clone() const {
            return Function1Ptr(new LanczosFunction1(_n, this->_params[0]));
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

        virtual std::string toString(std::string const& prefix) const {
            std::ostringstream os;
            os << "LanczosFunction1 [" << _invN << "]: ";;
            os << Function1<ReturnT>::toString(prefix);
            return os.str();
        }

    private:
        int _n;
        double _invN;                   // == 1/n

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive&, unsigned int const) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                LanczosFunction1<ReturnT>, Function1<ReturnT> >(
                    static_cast<LanczosFunction1<ReturnT>*>(0),
                    static_cast<Function1<ReturnT>*>(0));
#endif
        }
        template <typename R, class Archive>
        friend void boost::serialization::save_construct_data(
            Archive& ar, LanczosFunction1<R> const* f, unsigned int const);
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

        virtual ~LanczosFunction2() {}
       
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

        virtual std::string toString(std::string const& prefix) const {
            std::ostringstream os;
            os << "LanczosFunction2 [" << _invN << "]: ";;
            os << Function2<ReturnT>::toString(prefix);
            return os.str();
        }

    private:
        double _invN;   ///< 1/n

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive&, unsigned int const) {
#ifndef SWIG
            boost::serialization::void_cast_register<
                LanczosFunction2<ReturnT>, Function2<ReturnT> >(
                    static_cast<LanczosFunction2<ReturnT>*>(0),
                    static_cast<Function2<ReturnT>*>(0));
#endif
        }
        template <typename R, class Archive>
        friend void boost::serialization::save_construct_data(
            Archive& ar, LanczosFunction2<R> const* f, unsigned int const);
    };

}}}   // lsst::afw::math

namespace boost {
namespace serialization {

template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::IntegerDeltaFunction1<ReturnT> const* f,
                                unsigned int const) {
    ar << make_nvp("xo", f->_xo);
}

template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::IntegerDeltaFunction2<ReturnT> const* f,
                                unsigned int const) {
    ar << make_nvp("xo", f->_xo) << make_nvp("yo", f->_yo);
}

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::IntegerDeltaFunction2<ReturnT>* f,
                                unsigned int const) {
    double xo;
    double yo;
    ar >> make_nvp("xo", xo) >> make_nvp("yo", yo);
    ::new(f) lsst::afw::math::IntegerDeltaFunction2<ReturnT>(xo, yo);
}

template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::GaussianFunction1<ReturnT> const* f,
                                unsigned int const) {
    ar << make_nvp("sigma", f->getParameters()[0]);
}

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::GaussianFunction1<ReturnT>* f,
                                unsigned int const) {
    double sigma;
    ar >> make_nvp("sigma", sigma);
    ::new(f) lsst::afw::math::GaussianFunction1<ReturnT>(sigma);
}
    
template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::GaussianFunction2<ReturnT> const* f,
                                unsigned int const) {
    ar << make_nvp("sigma1", f->getParameters()[0]);
    ar << make_nvp("sigma2", f->getParameters()[1]);
    ar << make_nvp("angle",  f->getParameters()[2]);
}

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::GaussianFunction2<ReturnT>* f,
                                unsigned int const) {
    double sigma1;
    double sigma2;
    double angle;
    ar >> make_nvp("sigma1", sigma1);
    ar >> make_nvp("sigma2", sigma2);
    ar >> make_nvp("angle",  angle);
    ::new(f) lsst::afw::math::GaussianFunction2<ReturnT>(sigma1, sigma2, angle);
}

template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::DoubleGaussianFunction2<ReturnT> const* f,
                                unsigned int const) {
    ar << make_nvp("sigma1", f->getParameters()[0]);
    ar << make_nvp("sigma2", f->getParameters()[1]);
    ar << make_nvp("ampl2", f->getParameters()[2]);
}

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::DoubleGaussianFunction2<ReturnT>* f,
                                unsigned int const) {
    double sigma1;
    double sigma2;
    double ampl2;
    ar >> make_nvp("sigma1", sigma1);
    ar >> make_nvp("sigma2", sigma2);
    ar >> make_nvp("ampl2", ampl2);
    ::new(f) lsst::afw::math::DoubleGaussianFunction2<ReturnT>(sigma1, sigma2, ampl2);
}
    
template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::PolynomialFunction1<ReturnT> const* f,
                                unsigned int const) {
    ar << make_nvp("params", f->getParameters());
}

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::PolynomialFunction1<ReturnT>* f,
                                unsigned int const) {
    std::vector<double> params;
    ar >> make_nvp("params", params);
    ::new(f) lsst::afw::math::PolynomialFunction1<ReturnT>(params);
}

template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::PolynomialFunction2<ReturnT> const* f,
                                unsigned int const) {
    ar << make_nvp("params", f->getParameters());
}

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::PolynomialFunction2<ReturnT>* f,
                                unsigned int const) {
    std::vector<double> params;
    ar >> make_nvp("params", params);
    ::new(f) lsst::afw::math::PolynomialFunction2<ReturnT>(params);
}
    
template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::Chebyshev1Function1<ReturnT> const* f,
                                unsigned int const) {
    ar << make_nvp("params", f->getParameters());
    ar << make_nvp("minX", f->_minX);
    ar << make_nvp("maxX", f->_maxX);
}

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::Chebyshev1Function1<ReturnT>* f,
                                unsigned int const) {
    std::vector<double> params;
    double minX;
    double maxX;
    ar >> make_nvp("params", params);
    ar >> make_nvp("minX", minX);
    ar >> make_nvp("maxX", maxX);
    ::new(f) lsst::afw::math::Chebyshev1Function1<ReturnT>(params, minX, maxX);
}
    
template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::Chebyshev1Function2<ReturnT> const* f,
                                unsigned int const) {
    ar << make_nvp("params", f->getParameters());
    ar << make_nvp("minX", f->_minX);
    ar << make_nvp("minY", f->_minY);
    ar << make_nvp("maxX", f->_maxX);
    ar << make_nvp("maxY", f->_maxY);
}

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::Chebyshev1Function2<ReturnT>* f,
                                unsigned int const) {
    std::vector<double> params;
    double minX;
    double minY;
    double maxX;
    double maxY;
    ar >> make_nvp("params", params);
    ar >> make_nvp("minX", minX);
    ar >> make_nvp("minY", minY);
    ar >> make_nvp("maxX", maxX);
    ar >> make_nvp("maxY", maxY);
    ::new(f) lsst::afw::math::Chebyshev1Function2<ReturnT>(params, minX, minY, maxX, maxY);
}

template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::LanczosFunction1<ReturnT> const* f,
                                unsigned int const) {
    unsigned int n = static_cast<unsigned int>(0.5 + (1.0 / f->_invN));
    ar << make_nvp("n", n);
    ar << make_nvp("xOffset", f->getParameters()[0]);
}

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::LanczosFunction1<ReturnT>* f,
                                unsigned int const) {
    unsigned int n;
    double xOffset;
    ar >> make_nvp("n", n);
    ar >> make_nvp("xOffset", xOffset);
    ::new(f) lsst::afw::math::LanczosFunction1<ReturnT>(n, xOffset);
}
    
template <typename ReturnT, class Archive>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::LanczosFunction2<ReturnT> const* f,
                                unsigned int const) {
    unsigned int n = static_cast<unsigned int>(0.5 + (1.0 / f->_invN));
    ar << make_nvp("n", n);
    ar << make_nvp("xOffset", f->getParameters()[0]);
    ar << make_nvp("yOffset", f->getParameters()[1]);
}

template <typename ReturnT, class Archive>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::LanczosFunction2<ReturnT>* f,
                                unsigned int const) {
    unsigned int n;
    double xOffset;
    double yOffset;
    ar >> make_nvp("n", n);
    ar >> make_nvp("xOffset", xOffset);
    ar >> make_nvp("yOffset", yOffset);
    ::new(f) lsst::afw::math::LanczosFunction2<ReturnT>(n, xOffset, yOffset);
}

}}

#endif // #ifndef LSST_AFW_MATH_FUNCTIONLIBRARY_H
