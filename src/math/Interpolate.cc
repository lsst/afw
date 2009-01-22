/**
 * \file
 * \brief Interpolation Classes, Linear, Spline, SplineNatural, SplineNotAKnot
 */
#include <iostream>
#include <cmath>
#include <limits>
#include <vector>
#include <cassert>
#include <iterator>

#include "boost/shared_ptr.hpp"
#include "lsst/afw/math/Interpolate.h"

using namespace std;
namespace math = lsst::afw::math;

namespace {
    double const NaN = std::numeric_limits<double>::quiet_NaN();
}


// =======================================================================================
/**
 * \brief Constructor for Generic interpolation
 *
 * The constructor makes (allocates) private copies of the input x/y vectors and sets private member values
 * for x-vector bounds, and gridspacings.
 *
 */
template<typename xT, typename yT>
math::Interpolate<xT,yT>::Interpolate(vector<xT> const& x, vector<yT> const& y, math::InterpControl const& ictrl)
    : _n(x.size() + 2),
      _x(*new vector<xT>), _y(*new vector<yT>),
      _xgridspace( static_cast<double>(x[1] - x[0]) ),
      _invxgrid( 1.0/_xgridspace ),
      _xlo( x[0] - static_cast<xT>(_xgridspace) ),
      _xhi( x[x.size() - 1] + static_cast<xT>(_xgridspace)),
      _ictrl( ictrl )  {
    
    assert ( x.size() == y.size() );
    
    _x.resize(_n);
    _y.resize(_n);
    std::copy(x.begin(), x.end(), _x.begin() + 1);
    std::copy(y.begin(), y.end(), _y.begin() + 1);
    _x[0] = _xlo;
    _x[_n - 1] = _xhi;
    
}
    

// ======================  LINEAR ===========================================

/**
 * \brief A private method to intialize for linear interpolation
 *
 * This pre-computes the first derivatives over the intervals
 */
template<typename xT, typename yT>
math::LinearInterpolate<xT,yT>::LinearInterpolate(vector<xT> const& x, vector<yT> const& y,
                                                  math::InterpControl const& ictrl)
    : Interpolate<xT,yT>::Interpolate(x,y,ictrl), _dydx(*new vector<yT>) {
    
    _dydx.resize(_n - 1);
    for (int i = 1; i < _n - 2; ++i) {
        _dydx[i] = (_y[i + 1] - _y[i]) * static_cast<yT>(_invxgrid);
    }
    // carry extra points off the end to extrapolate.
    _dydx[0] = _dydx[1];
    _y[0] = _y[1] - static_cast<yT>(_dydx[1]*_xgridspace);
    
    _dydx[_n - 2] = _dydx[_n - 3];
    _y[_n - 1] = _y[_n - 2] + static_cast<yT>(_dydx[_n - 2]*_xgridspace);
}

// ==== LINEAR no-safe ===
/**
 * \brief Private method to return a linearly-interpolated value for a point *without* bounds checking.
 */
template<typename xT, typename yT>
inline yT math::LinearInterpolate<xT,yT>::interpolate(xT const xinterp) const {
    int const index = static_cast<int>((xinterp - _xlo) * _invxgrid);
    return _y[index] + static_cast<yT>(_dydx[index]*(xinterp - _x[index]));
}
/**
 * \brief Private method to return a linearly-interpolated value for a point *without* bounds checking.
 */
template<typename xT, typename yT>
inline yT math::LinearInterpolate<xT,yT>::interpolateDyDx(xT const xinterp) const {
    // assume dydx represents the mid-value of the interval
    xT const xinterp_tmp = static_cast<xT>(xinterp - 0.5*_xgridspace);
    int const index = static_cast<int>((xinterp_tmp - _xlo) * _invxgrid);
    yT const a = static_cast<yT>((_x[index + 1] - xinterp_tmp)*_invxgrid);
    yT const b = static_cast<yT>((xinterp_tmp - _x[index])*_invxgrid );
    return a*_dydx[index] + b*_dydx[index + 1];
}
/**
 * \brief Private method to return a linearly-interpolated value for a point *without* bounds checking.
 */
template<typename xT, typename yT>
inline yT math::LinearInterpolate<xT,yT>::interpolateD2yDx2(xT const xinterp) const {
    return 0;
}

// ==== LINEAR safe ====
/**
 * \brief Private method to return a linearly-interpolated value for a point *with* bounds checking.
 */
template<typename xT, typename yT>
yT math::LinearInterpolate<xT,yT>::interpolate_safe(xT const xinterp) const {
    
    int index = static_cast<int>((xinterp - _xlo) * _invxgrid);
    if ( index < 0 ) {
        index = 0;
    } else if ( index > _n - 1 ) {
        index = _n - 1;
    }
    return _y[index] + static_cast<yT>(_dydx[index]*(xinterp - _x[index]));
}

/**
 * \brief Private method to return a linearly-interpolated value for a point *without* bounds checking.
 */
template<typename xT, typename yT>
yT math::LinearInterpolate<xT,yT>::interpolateDyDx_safe(xT const xinterp) const {
    // assume dydx represents the mid-value of the interval
    xT const xinterp_tmp = static_cast<xT>(xinterp - 0.5*_xgridspace);
    int index = static_cast<int>((xinterp_tmp - _xlo) * _invxgrid);
    if ( index < 0 ) {
        index = 0;
    } else if ( index > _n - 1 ) {
        index = _n - 1;
    }
    yT const a = static_cast<yT>((_x[index + 1] - xinterp_tmp)*_invxgrid);
    yT const b = static_cast<yT>((xinterp_tmp - _x[index])*_invxgrid );
    return a*_dydx[index] + b*_dydx[index + 1];
}
/**
 * \brief Private method to return a linearly-interpolated value for a point *without* bounds checking.
 */
template<typename xT, typename yT>
yT math::LinearInterpolate<xT,yT>::interpolateD2yDx2_safe(xT const xinterp) const {
    return 0;
}

// =========================   END LINEAR ==================================





// ==========================  SPLINE ======================================

/**
 * \brief Initialization for Cubic Spline interpolation - Press et al. 2007
 *
 * This mainly just pre-computes the second derivatives over the intervals
 */
template<typename xT, typename yT>
math::SplineInterpolate<xT,yT>::SplineInterpolate(vector<xT> const& x, vector<yT> const& y, InterpControl const& ictrl)
    : Interpolate<xT,yT>::Interpolate(x, y, ictrl), _d2ydx2(*new vector<yT>) {
    
    //_dydx0 = _dydxN = std::numeric_limits<double>::quiet_NaN();
    _dydx0 = _ictrl.getDydx0();
    _dydxN = _ictrl.getDydxN();
    
    _d2ydx2.resize(_n);
    vector<yT> u(_n);
    
    // =============================================================
    // the lower boundary condition
    if ( std::isnan(_dydx0) ) {
        _d2ydx2[1] = u[1] = static_cast<yT>(0.0);
    } else {
        _d2ydx2[1] = static_cast<yT>(-0.5);
        u[1] = static_cast<yT>((3.0*_invxgrid) * ((_y[2] - _y[1])*(_invxgrid) - _dydx0));
    }

    // =============================================================
    // decomposition for the tri-diagonal matrix
    for (int i = 2; i < _n - 2; ++i) {
        yT const invp   = static_cast<yT>(1.0 / (0.5 * _d2ydx2[i - 1] + 2.0));
        _d2ydx2[i] = static_cast<yT>(-0.5)*invp;
        u[i] = static_cast<yT>(_invxgrid * ( _y[i + 1] - 2.0 * _y[i] + _y[i - 1] ));
        u[i] = static_cast<yT>(0.5*(6.0*u[i]*_invxgrid - u[i - 1])*invp);
    }

    // =============================================================
    // the upper boundary condition
    yT qn, un;
    if ( std::isnan(_dydxN) ) {
        qn = un = static_cast<yT>(0.0);
    } else {
        qn = static_cast<yT>(0.5);
        un = static_cast<yT>( (3.0*_invxgrid)*(_dydxN - (_y[_n - 2] - _y[_n - 3])*_invxgrid) );
    }
    _d2ydx2[_n - 2] = static_cast<yT>( (un - qn*u[_n - 3])/(qn*_d2ydx2[_n - 3] + 1.0) );

    // =============================================================
    // the back-substitution loop for tri-diag algorithm
    for (int k = _n - 3; k >= 1; k--) {
        _d2ydx2[k] = _d2ydx2[k] * _d2ydx2[k + 1] + u[k];
    }

    
    // =============================================================
    // solve for the extrapolated _y[0] and _y[_n-1] values;
    // -- recall that we shifted the vectors by one to put anchor points in
    //    the first and last vector positions.
    _y[0]      = interpolate_safe(_x[0]);
    _d2ydx2[0] = interpolateD2yDx2_safe(_x[0]);
    
    _y[_n - 1]      = interpolate_safe(_x[_n - 1]);
    _d2ydx2[_n - 1] = interpolateD2yDx2_safe(_x[_n - 1]);
    
}


// ==== SPLINE no-safe ====

/**
 * \brief Public method to return spline-interpolated values over a vector<> *without* bounds checking
 */
template<typename xT, typename yT>
inline yT math::SplineInterpolate<xT,yT>::interpolate(xT const xinterp) const {
    int index = static_cast<int>((xinterp - _xlo) * _invxgrid);
    double const a = (_x[index + 1] - xinterp) * _invxgrid;
    double const b = ( xinterp - _x[index] ) * _invxgrid;
    double const yinterp = a*_y[index] + b*_y[index + 1] +
        ( (a*a*a - a)*_d2ydx2[index] + (b*b*b - b)*_d2ydx2[index + 1] ) * (_xgridspace*_xgridspace) / 6.0;
    return static_cast<yT>(yinterp);
}

/**
 * \brief Public method to return spline-interpolated first derivatives over a vector<> *without* bounds checking
 */
template<typename xT, typename yT>
inline yT math::SplineInterpolate<xT,yT>::interpolateDyDx(xT const xinterp) const {
    int index = static_cast<int>((xinterp - _xlo) * _invxgrid);
    double const a = (_x[index + 1] - xinterp) * _invxgrid;
    double const b = ( xinterp - _x[index] ) * _invxgrid;
    double const yinterp = (_y[index+1] - _y[index]) * _invxgrid -
        ((3.0*a*a - 1.0)/6.0)*_xgridspace*_d2ydx2[index] +
        ((3.0*b*b - 1.0)/6.0)*_xgridspace*_d2ydx2[index+1];
    return static_cast<yT>(yinterp);
}

/**
 * \brief Public method to return spline-interpolated second derivatives over a vector<> *without* bounds checking
 */
template<typename xT, typename yT>
inline yT math::SplineInterpolate<xT,yT>::interpolateD2yDx2(xT const xinterp) const {
    int index = static_cast<int>((xinterp - _xlo) * _invxgrid);
    double const a = (_x[index + 1] - xinterp) * _invxgrid;
    double const b = ( xinterp - _x[index] ) * _invxgrid;
    double const yinterp = a*_d2ydx2[index] + b*_d2ydx2[index + 1];
    return static_cast<yT>(yinterp);
}


// ==== SPLINE safe ====

/**
 * \brief Public method to return spline-interpolated values over a vector<> *with* bounds checking
 */
template<typename xT, typename yT>
yT math::SplineInterpolate<xT,yT>::interpolate_safe(xT const xinterp) const {
    
    // Caveat = only good for an even grid spacing.
    int index = static_cast<int>((xinterp - _xlo) * _invxgrid);
    if ( index < 1 ) {
        index = 1;
    } else if ( index > _n - 3 ) {
        index = _n - 3;
    }
    
    double const a = (_x[index + 1] - xinterp) * _invxgrid;
    double const b = ( xinterp - _x[index] ) * _invxgrid;
    double const yinterp = a*_y[index] + b*_y[index + 1] +
        ( (a*a*a - a)*_d2ydx2[index] + (b*b*b - b)*_d2ydx2[index + 1] ) * (_xgridspace*_xgridspace) / 6.0;
    return static_cast<yT>(yinterp);
}

/**
 * \brief Public method to return spline-interpolated first derivatives over a vector<> *with* bounds checking
 */
template<typename xT, typename yT>
yT math::SplineInterpolate<xT,yT>::interpolateDyDx_safe(xT const xinterp) const {
    
    // Caveat = only good for an even grid spacing.
    int index = static_cast<int>((xinterp - _xlo) * _invxgrid);
    if ( index < 0 ) {
        index = 1;
    } else if ( index > _n - 3 ) {
        index = _n - 3;
    }
    
    double const a = (_x[index + 1] - xinterp) * _invxgrid;
    double const b = ( xinterp - _x[index] ) * _invxgrid;
    double const yinterp = (_y[index+1] - _y[index]) * _invxgrid -
        ((3.0*a*a - 1.0)/6.0)*_xgridspace*_d2ydx2[index] +
        ((3.0*b*b - 1.0)/6.0)*_xgridspace*_d2ydx2[index+1];

    return static_cast<yT>(yinterp);
}

/**
 * \brief Public method to return spline-interpolated second derivatives over a vector<> *with* bounds checking
 */
template<typename xT, typename yT>
yT math::SplineInterpolate<xT,yT>::interpolateD2yDx2_safe(xT const xinterp) const {
    
    // Caveat = only good for an even grid spacing.
    int index = static_cast<int>((xinterp - _xlo) * _invxgrid);
    if ( index < 1 ) {
        index = 1;
    } else if ( index > _n - 3 ) {
        index = _n - 3;
    }
    
    double const a = (_x[index + 1] - xinterp) * _invxgrid;
    double const b = ( xinterp - _x[index] ) * _invxgrid;
    double const yinterp = a*_d2ydx2[index] + b*_d2ydx2[index + 1];
    return static_cast<yT>(yinterp);
}

// =========================================  END SPLINE ====================================





/************************************************************************************************************/
//
// Explicit instantiations
//
template class math::Interpolate<double,double>;
template class math::Interpolate<float,float>;
template class math::Interpolate<int,double>;
template class math::Interpolate<int,float>;
template class math::Interpolate<int,int>;

template class math::LinearInterpolate<double,double>;
template class math::LinearInterpolate<float,float>;
template class math::LinearInterpolate<int,double>;
template class math::LinearInterpolate<int,float>;
template class math::LinearInterpolate<int,int>;

template class math::SplineInterpolate<double,double>;
template class math::SplineInterpolate<float,float>;
template class math::SplineInterpolate<int,double>;
template class math::SplineInterpolate<int,float>;
template class math::SplineInterpolate<int,int>;

