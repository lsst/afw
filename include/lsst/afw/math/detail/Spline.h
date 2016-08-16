#if !defined(LSST_AFW_MATH_DETAIL_SPLINE)
#define LSST_AFW_MATH_DETAIL_SPLINE 1
#include <cmath>
#include <vector>

namespace lsst { namespace afw { namespace math { namespace detail {

/*****************************************************************************/
/*
 * Splines
 */
class Spline {
public:
    virtual ~Spline() {}

    void interpolate(std::vector<double> const& x, ///< points to interpolate at
                     std::vector<double> &y        ///< interpolated values at x
                    ) const;
    void derivative(std::vector<double> const& x, ///< points to evaluate derivative at
                    std::vector<double> &dydx     ///< derivatives at x
                   ) const;

    std::vector<double> roots(double const value, ///< desired value
                              double const x0,    ///< specify desired range is [x0,x1)
                              double const x1     ///< specify desired range is [x0,x1)
                             ) const;

protected:
    Spline() {}
    void _allocateSpline(int const nknot);

    std::vector<double> _knots;                // positions of knots
    std::vector<std::vector<double> > _coeffs; // and associated coefficients
};

class TautSpline : public Spline {
public:
    enum Symmetry { Unknown, Odd, Even };

    TautSpline(std::vector<double> const& x,
               std::vector<double> const& y,
               double const gamma=0,
               Symmetry type=Unknown
          );
private:
    void calculateTautSpline(std::vector<double> const& x,
                             std::vector<double> const& y,
                             double const gamma0
                            );
    void calculateTautSplineEvenOdd(std::vector<double> const& x,
                                    std::vector<double> const& y,
                                    double const gamma0,
                                    bool even
                                );
};

class SmoothedSpline : public Spline {
public:
    SmoothedSpline(std::vector<double> const& x,  ///< points where function's specified; monotonic increasing
                   std::vector<double> const& y,  ///< values of function at x
                   std::vector<double> const& dy, ///< error in function at x
                   double s,                      ///< desired chisq
                   double *chisq=NULL,            ///< final chisq (if non-NULL)
                   std::vector<double> *errs=NULL ///< error estimates, (if non-NULL).  You'll need to delete it
          );
};
}}}}
#endif
