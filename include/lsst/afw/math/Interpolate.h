#if !defined(LSST_AFW_MATH_INTERPOLATE_H)
#define LSST_AFW_MATH_INTERPOLATE_H
/**
 * \file
 * \brief Interpolation Header
 */
namespace lsst { namespace afw { namespace math { namespace interpolate {


    namespace {
        double const NaN = std::numeric_limits<double>::quiet_NaN();
    }


    enum Style {
        LINEAR = 0x01,
        NATURAL_SPLINE = 0x02,
        NOTAKNOT_SPLINE = 0x04,
        CUBIC_SPLINE = 0x08,
    };

    class InterpControl {
    public:
        InterpControl( Style const style=interpolate::NATURAL_SPLINE,
                       double const dydx0=NaN, double const dydxN=NaN
                     ) : _style(style), _dydx0(dydx0), _dydxN(dydxN) {
        };
        void setDydx0(double const dydx0) { _dydx0 = dydx0; }
        void setDydxN(double const dydxN) { _dydxN = dydxN; }
        double getDydx0() { return _dydx0; }
        double getDydxN() { return _dydxN; }
        Style getStyle() const { return _style; }
    private:
        Style _style;
        double _dydx0;
        double _dydxN;
    };

    
    template<typename xT, typename yT>
    class Interpolator {
    public:
        Interpolator(std::vector<xT> const& x, std::vector<yT> const& y,
                     InterpControl const& ictrl = InterpControl());
        
        std::vector<yT> interp(std::vector<xT> const& xinterp) const;
        yT interp(xT const& xinterp) const;
    private:
        int _n;
        std::vector<xT> _x;
        std::vector<yT> _y;
        std::vector<double> _dydx;
        std::vector<double> _d2ydx2;
        double _xgridspace;
        double _invxgrid;
        xT _xlo;
        xT _xhi;
        double _dydx0;
        double _dydxN;
        InterpControl _ictrl;
        void _init_Linear(std::vector<xT> const& x, std::vector<yT> const& y);
        void _init_Spline(std::vector<xT> const& x, std::vector<yT> const& y);


        std::vector<yT> _interp_Linear(std::vector<xT> const& xinterp) const;
        yT _interp_Linear(xT const& xinterp) const;
        
        std::vector<yT> _interp_Spline(std::vector<xT> const& xinterp) const;
        yT _interp_Spline(xT const& xinterp) const;
    };

    
}}}}
                     
#endif // LSST_AFW_MATH_INTERPOLATE_H
