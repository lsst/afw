#if !defined(LSST_AFW_MATH_INTERPOLATE_H)
#define LSST_AFW_MATH_INTERPOLATE_H
/**
 * \file
 * \brief Interpolation Header
 */

#include "boost/shared_ptr.hpp"

namespace lsst { namespace afw { namespace math {


    namespace {
        double const NaN = std::numeric_limits<double>::quiet_NaN();
    }

    /// \brief Select style of interpolation to use
    enum Style {
        LINEAR = 0x01,                  ///< use linear interpolation
        NATURAL_SPLINE = 0x02,          ///< use a natural spline    [ y''(0) = y''(n-1) = 0 ]
        NOTAKNOT_SPLINE = 0x04,         ///< use a not-a-knot spline [ y'''(0,n-1) = y'''(1,n-2) ]
        CUBIC_SPLINE = 0x08,            ///< a generic cubic spline, with user-set y'(0), y'(n-1)
    };

    /** \brief Pass parameters in to the interpolation routine
     *
     *  This class is not currently implemented with the Interpolate class.
     */
    class InterpControl {
    public:
        InterpControl( Style const style=math::NATURAL_SPLINE,
                       double const dydx0=NaN, double const dydxN=NaN
                     ) : _style(style), _dydx0(dydx0), _dydxN(dydxN) {
        };
        void setDydx0(double const dydx0) { _dydx0 = dydx0; }
        void setDydxN(double const dydxN) { _dydxN = dydxN; }
        double getDydx0() { return _dydx0; }
        double getDydxN() { return _dydxN; }
        Style getStyle() const { return _style; }
    private:
        Style _style;                   // interpolation style from "enum Style" above
        double _dydx0;                  // user-set first deriv at x_i=0 (for spline boundary conditions)
        double _dydxN;                  // user-set first deriv at x_i=N (for spline boundary conditions)
    };


    /** \brief A class to handle interpolation between x,y points in vector<> inputs
     *
     * An interpolator object is declared and initialized for a pair of
     * vector<>s describing x,y coordinates to be interpolated
     * over. Interpolated points are then obtained by calling an 'interp' method
     * for the interpolator object.
     *
     * \code
           vector<double> x;                                          // put x-coords in this
           vector<double> y;                                          // put f(x) in this
           double xinterp;                    // the x coord we'd like an interpolated value for
           
           math::LinearInterpolate<double,double> Linterpobj(x, y);    // make a LinearInterpolate object
           double yinterp = Linterpobj.interpolate(xinterp1);              // a linear interpolated value

           math::SplineInterpolate<double,double> Sinterpobj(x, y);    // make a SplineInterpolate object
           double yinterp = Sinterpobj.interpolate(xinterp1);              // a spline interpolated value
     * \endcode
     *
     * Notes: The routines assume evenly spaced grid points.  This is not, in general, a requirement for the
     *  algorithm, but was used here for speed.
     *
     */
    
    template<typename xT, typename yT>
    class Interpolate {
    public:
        Interpolate(std::vector<xT> const& x, std::vector<yT> const& y, InterpControl const& ictrl = InterpControl());
        virtual ~Interpolate() { delete &_x; delete &_y; };
        virtual yT interpolate(xT const xinterp) const = 0;  // linearly interpolate this object at x=xinterp
        virtual yT interpolate_safe(xT const xinterp) const = 0;  // linearly interpolate this object at x=xinterp

        //InterpControl const& ictrl = InterpControl());
        
    private:
    protected:
        int const _n;                         // the number of points in the _x,_y vectors
        std::vector<xT>& _x;                  // _n x-coordinates
        std::vector<yT>& _y;                  // _n y-coordinates        
        double const _xgridspace;             // the grid spacing
        double const _invxgrid;               // the inverse grid spacing (1/_xgridspacing)
        xT const _xlo;                        // the lowest value in _x
        xT const _xhi;                        // the highest value in _x
        InterpControl _ictrl;
    };


    template<typename xT, typename yT>
    class LinearInterpolate : public Interpolate<xT,yT> {
    public:
        
        // pre-calculate dydx values
        LinearInterpolate(std::vector<xT> const& x, std::vector<yT> const& y, InterpControl const& ictrl = InterpControl());
        ~LinearInterpolate() { delete &_dydx; };

        // fast methods with *no* bounds checking
        yT interpolate(xT const xinterp) const;  // linearly interpolate this object at x=xinterp
        yT interpolateDyDx(xT const xinterp) const; // linearly interpolate this obejct at x=xinterp
        yT interpolateD2yDx2(xT const xinterp) const; // lineary interpolate this obejct at x=xinterp

        // slow methods with bounds checking
        yT interpolate_safe(xT const xinterp) const;  // linearly interpolate this object at x=xinterp
        yT interpolateDyDx_safe(xT const xinterp) const; // linearly interpolate this obejct at x=xinterp
        yT interpolateD2yDx2_safe(xT const xinterp) const; // linearly interpolate this obejct at x=xinterp
        
    private:
        // see Meyer's Item 43.
        using Interpolate<xT,yT>::_n;
        using Interpolate<xT,yT>::_x;
        using Interpolate<xT,yT>::_y;
        using Interpolate<xT,yT>::_xgridspace;
        using Interpolate<xT,yT>::_invxgrid;
        using Interpolate<xT,yT>::_xlo;           
        using Interpolate<xT,yT>::_xhi;
        using Interpolate<xT,yT>::_ictrl;
        std::vector<yT>& _dydx;
    };
    
    template<typename xT, typename yT>
    class SplineInterpolate : public Interpolate<xT,yT> {
    public:

        // pre-calculate d2ydx2 values
        SplineInterpolate(std::vector<xT> const& x, std::vector<yT> const& y, InterpControl const& ictrl = InterpControl());
        ~SplineInterpolate() { delete &_d2ydx2; };

        // fast methods with *no* bounds checking        
        yT interpolate(xT const xinterp) const; // spline interpolate this obejct at x=xinterp
        yT interpolateDyDx(xT const xinterp) const; // spline interpolate this obejct at x=xinterp
        yT interpolateD2yDx2(xT const xinterp) const; // spline interpolate this obejct at x=xinterp

        // slow methods with bounds checking        
        yT interpolate_safe(xT const xinterp) const; // spline interpolate this obejct at x=xinterp
        yT interpolateDyDx_safe(xT const xinterp) const; // spline interpolate this obejct at x=xinterp
        yT interpolateD2yDx2_safe(xT const xinterp) const; // spline interpolate this obejct at x=xinterp
        
    private:
        using Interpolate<xT,yT>::_n;
        using Interpolate<xT,yT>::_x;
        using Interpolate<xT,yT>::_y;
        using Interpolate<xT,yT>::_xgridspace;
        using Interpolate<xT,yT>::_invxgrid;
        using Interpolate<xT,yT>::_xlo;           
        using Interpolate<xT,yT>::_xhi;
        using Interpolate<xT,yT>::_ictrl;
        std::vector<yT>& _d2ydx2;    // _n second-derivatives used for spline interp
        double _dydx0;                  // the first derivative at point 0 (user-set for spline)
        double _dydxN;                  // the first derivative at point N (user-set for spline)
    };

        
}}}
                     
#endif // LSST_AFW_MATH_INTERPOLATE_H
