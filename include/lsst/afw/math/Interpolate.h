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


//     enum Style {
//         LINEAR = 0x01;
//         NATURAL_SPLINE = 0x02;
//         NOTAKNOT_SPLINE = 0x04;
//     };

//     class InterpControl {
//     public:
//         explicit InterpControl( 
//                               ) : {
//         }
//     private:
//     };

//     template<typename xT, typename yT>
//     class Interpolator {
//     public:
//         explicit Interpolator(std::vector<xT> const& x, std::vector<yT> const& y,
//                               interpolate::InterpControl const& ictrl);
        
//         std::vector<yT> interp(std::vector<xT> const& xinterp) const;
//         yT interp(xT const& xinterp) const;
//     private:
//         int _n;
//         std::vector<xT> _x;
//         std::vector<yT> _y;
//         std::vector<double> _dydx;
//         double _xgridspace;
//         double _invxgrid;
//         xT _xlo;
//         xT _xhi;
//         yT _interp(xT const& xinterp) const;
//     };

    
    template<typename xT, typename yT>
    class Linear {
    public:
        explicit Linear(std::vector<xT> const& x, std::vector<yT> const& y);
        std::vector<yT> interp(std::vector<xT> const& xinterp) const;
        yT interp(xT const& xinterp) const;
    private:
        int _n;
        std::vector<xT> _x;
        std::vector<yT> _y;
        std::vector<double> _dydx;
        double _xgridspace;
        double _invxgrid;
        xT _xlo;
        xT _xhi;
        yT _interp(xT const& xinterp) const;
    };

       

    template<typename xT, typename yT>
    Linear<xT,yT> init_Linear(std::vector<xT> const& x, std::vector<yT> const& y) { ///< ImageT (or MaskedImage) whose properties we want
        return Linear<xT,yT>(x, y);
    };

    
    template<typename xT, typename yT>
    class NaturalSpline {
    public:
        explicit NaturalSpline(std::vector<xT> const& x, std::vector<yT> const& y, yT const dydx0, yT const dydxN);
        std::vector<yT> interp(std::vector<xT> const& xinterp) const;
        yT interp(xT const& xinterp) const;
    private:
        int _n;
        std::vector<xT> _x;
        std::vector<yT> _y;
        std::vector<double> _d2ydx2;
        double _xgridspace;
        double _invxgrid;
        xT _xlo;
        xT _xhi;
        double _dydx0;
        double _dydxN;
        yT _interp(xT const& xinterp) const;
    };

       

    template<typename xT, typename yT>
    NaturalSpline<xT,yT> init_NaturalSpline(std::vector<xT> const& x, std::vector<yT> const& y, yT const dydx0=NaN, yT const dydxN=NaN) { ///< ImageT (or MaskedImage) whose properties we want
        return NaturalSpline<xT,yT>(x, y, dydx0, dydxN);
    };


}}}}
                     
#endif // LSST_AFW_MATH_INTERPOLATE_H
