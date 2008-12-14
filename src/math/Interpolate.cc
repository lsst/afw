/**
 * \file
 * \brief Interpolation Classes, Linear, Spline, SplineNatural, SplineNotAKnot
 */
#include <iostream>
#include <cmath>
#include <limits>
#include <vector>
#include <cassert>

#include "lsst/afw/math/Interpolate.h"

using namespace std;
namespace math = lsst::afw::math;
namespace interpolate = lsst::afw::math::interpolate;

namespace {
    double const NaN = std::numeric_limits<double>::quiet_NaN();
}


// =======================================================================================
/**
 * Constructor for Linear interpolation
 *
 */
template<typename xT, typename yT>
interpolate::Interpolator<xT,yT>::Interpolator(vector<xT> const& x, vector<yT> const& y,
                                               interpolate::InterpControl const& ictrl
                                              ) : _x(x), _y(y), _ictrl(ictrl) {
    
    
    switch( _ictrl.getStyle() ) {
      case ( LINEAR ):
          _init_Linear(_x, _y);
          break;
      case ( NATURAL_SPLINE ):
          _dydx0 = std::numeric_limits<yT>::quiet_NaN();
          _dydxN = std::numeric_limits<yT>::quiet_NaN();
          _init_Spline(_x, _y);
          break;
      case ( NOTAKNOT_SPLINE ):
          _dydx0 = _ictrl.getDydx0();
          _dydxN = _ictrl.getDydxN();
          _init_Spline(_x, _y);
          break;
      case ( CUBIC_SPLINE ):
          _dydx0 = _ictrl.getDydx0();
          _dydxN = _ictrl.getDydxN();
          _init_Spline(_x, _y);
          break;
    }
    
}

template<typename xT, typename yT>
std::vector<yT> interpolate::Interpolator<xT,yT>::interp(std::vector<xT> const& xinterp) const {

    std::vector<yT> yinterp(xinterp.size());
    int const style = static_cast<interpolate::Style>(_ictrl.getStyle());
    if ( style & ( LINEAR ) ) {
        for (int i = 0; i < static_cast<int>(xinterp.size()); ++i) {        
            yinterp[i] = _interp_Linear(xinterp[i]);
        }
    } else if ( style & (NATURAL_SPLINE | NOTAKNOT_SPLINE | CUBIC_SPLINE) ) {
        for (int i = 0; i < static_cast<int>(xinterp.size()); ++i) {        
            yinterp[i] = _interp_Spline(xinterp[i]);
        }
    }
    return yinterp;
    
}

template<typename xT, typename yT>
yT interpolate::Interpolator<xT,yT>::interp(xT const& xinterp) const {

    yT yinterp;
    int const style = static_cast<interpolate::Style>(_ictrl.getStyle());
    if ( style & ( LINEAR ) ) {
        yinterp = _interp_Linear(xinterp);
    } else if ( style & ( NATURAL_SPLINE | NOTAKNOT_SPLINE | CUBIC_SPLINE) ) {
        yinterp = _interp_Spline(xinterp);
    }
    return yinterp;
}


template<typename xT, typename yT>
void interpolate::Interpolator<xT,yT>::_init_Linear(std::vector<xT> const& x, std::vector<yT> const& y) {

    assert( x.size() == y.size() );
    _n = x.size();
    _dydx.resize(_n-1);
    _xlo = _x[0];
    _xhi = _x[_n-1];
    _xgridspace = (_xhi - _xlo)/(_n - 1);
    _invxgrid = 1.0 / _xgridspace;
    for (int i = 0; i < _n-1; ++i) {
        //_dydx[i] = (_y[i+1] - _y[i]) / (_x[i+1] - _x[i]);
        _dydx[i] = (_y[i+1] - _y[i]) * _invxgrid;
    }
    
}


template<typename xT, typename yT>
std::vector<yT> interpolate::Interpolator<xT,yT>::_interp_Linear(std::vector<xT> const& xinterp) const {
    std::vector<yT> yinterp(xinterp.size());
    for (int i = 0; i < static_cast<int>(xinterp.size()); ++i) {        
        yinterp[i] = interpolate::Interpolator<xT,yT>::_interp_Linear(xinterp[i]);
    }
    return yinterp;
}

template<typename xT, typename yT>
yT interpolate::Interpolator<xT,yT>::_interp_Linear(xT const& xinterp) const {
    
    // Caveat = only good for an even grid spacing.
    int index = static_cast<int>(std::floor((xinterp - _xlo) * _invxgrid));
    int dindex = index;
    if ( xinterp < _xlo ) {
        index = 0;     dindex = 0;
    } else if ( xinterp >= _xhi ) {
        index = _n-1;  dindex = _n-2;
    }
    return _y[index] + static_cast<yT>(_dydx[dindex]*(xinterp - _x[index]));
}



// =======================================================================================
/**
 * Constructor for Natural Spline interpolation - Press et al. 2007
 *
 */
template<typename xT, typename yT>
void interpolate::Interpolator<xT,yT>::_init_Spline(std::vector<xT> const& x, std::vector<yT> const& y) {

    int const nx = x.size();
    int const ny = y.size();
    assert( nx == ny );
    _n = nx;
    _d2ydx2.resize(_n);
    _xlo = _x[0];
    _xhi = _x[nx - 1];
    _xgridspace = (_xhi - _xlo) / (_n - 1);
    _invxgrid = 1.0 / _xgridspace;
    
    vector<double> u(_n);
    
    // =============================================================
    // the lower boundary condition
    if ( std::isnan(_dydx0) ) {
        _d2ydx2[0] = u[0] = 0.0;
    } else {
        _d2ydx2[0] = -0.5;
        u[0] = (3.0/(x[1] - x[0])) * ((y[1] - y[0])/(x[1] - x[0]) - _dydx0);
    }

    // =============================================================
    // decomposition for the tri-diagonal matrix
    for (int i = 1; i < _n - 1; ++i) {
        
//         double sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
//         double p   = sig * _d2ydx2[i - 1] + 2.0;
//         _d2ydx2[i] = (sig - 1.0)/p;
//         u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
//         u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
        
        double const invp   = 1.0 / (0.5 * _d2ydx2[i - 1] + 2.0);
        _d2ydx2[i] = -0.5*invp;
        u[i] = _invxgrid * ( y[i + 1] - 2.0 * y[i] + y[i - 1] );
        u[i] = 0.5*(6.0*u[i]*_invxgrid - u[i - 1])*invp;       
    }

    // =============================================================
    // the upper boundary condition
    double qn, un;
    if ( std::isnan(_dydxN) ) {
        qn = un = 0.0;
    } else {
        qn = 0.5;
        un = (3.0/(x[_n - 1] - x[_n - 2])) * (_dydxN - (y[_n - 1] - y[_n - 2])/(x[_n - 1] - x[_n - 2]));
    }
    _d2ydx2[_n - 1] = (un - qn*u[_n - 2]) / (qn*_d2ydx2[_n - 2] + 1.0);

    // =============================================================
    // the back-substitution loop for tri-diag algorithm
    for (int k = _n - 2; k >= 0; k--) {
        _d2ydx2[k] = _d2ydx2[k] * _d2ydx2[k + 1] + u[k];
    }
    
}


template<typename xT, typename yT>
std::vector<yT> interpolate::Interpolator<xT,yT>::_interp_Spline(std::vector<xT> const& xinterp) const {
    std::vector<yT> yinterp(xinterp.size());
    for (int i = 0; i < static_cast<int>(xinterp.size()); ++i) {
        yinterp[i] = interpolate::Interpolator<xT,yT>::_interp_Spline(xinterp[i]);
    }
    return yinterp;
}

template<typename xT, typename yT>
yT interpolate::Interpolator<xT,yT>::_interp_Spline(xT const& xinterp) const {
    
    // Caveat = only good for an even grid spacing.
    int index = static_cast<int>(std::floor((xinterp - _xlo) * _invxgrid));
    if ( index < 0 ) {
        index = 0;
    } else if ( index > _n - 2 ) {
        index = _n - 2;
    }
    
    double const a = (_x[index + 1] - xinterp) * _invxgrid;
    double const b = ( xinterp - _x[index] ) * _invxgrid;
    double const yinterp = a*_y[index] + b*_y[index + 1] +
        ( (a*a*a - a)*_d2ydx2[index] + (b*b*b - b)*_d2ydx2[index + 1] ) * (_xgridspace*_xgridspace) / 6.0;
    return static_cast<yT>(yinterp);
}





/************************************************************************************************************/
//
// Explicit instantiations
//
template class interpolate::Interpolator<double,double>;
template class interpolate::Interpolator<float,float>;
template class interpolate::Interpolator<int,double>;
template class interpolate::Interpolator<int,float>;
template class interpolate::Interpolator<int,int>;

