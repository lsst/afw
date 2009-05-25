#if !defined(LSST_AFW_MATH_QUADRATURE_H)
#define LSST_AFW_MATH_QUADRATURE_H 1
/**
 * @file    Quadrature.h
 * @brief   Compute romberg integral and double integral.
 * @ingroup afw/math
 * @author  Steve Bickerton
 * @date    May 25, 2009
 */
namespace lsst { namespace afw { namespace math {


// =============================================================
/* @class IntegrandBase
 * @ingroup afw
 *
 * @brief Create a base class for function objects to be integrated by Quadrature.
 *
 * The Quadrature::romb2D() function wraps the function to be integrated in a
 * function object which calls Quadrature::romb().  romb2D() then calls romb()
 * with the wrapped function object.  In essence, romb() is calling itself
 * recusively to compute the double integral.  To operate this way, the
 * integrand must have a common format, as defined in this IntegrandBase base
 * class.
 *
 */
class IntegrandBase {
public:
    void setY(double const y) { _y = y; }
    virtual double operator() (double const x) = 0;
protected:
    double _y;
private:
};

// =============================================================
/* @brief Function to compute 1D and 2D Romberg integration.
 */
template<typename FunctionT>                        
double romb(FunctionT &func, double const a, double const b, double const eps = 1.0e-6);

double romb2D(IntegrandBase &func, double const x1, double const x2,
              double const y1, double const y2, double const eps = 1.0e-6);            


}}}
#endif
