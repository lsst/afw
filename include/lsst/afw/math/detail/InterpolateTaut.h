#if !defined(LSST_AFW_MATH_DETAIL_INTERPOLATE_TAUT)
#define LSST_AFW_MATH_DETAIL_INTERPOLATE_TAUT
#include <cmath>
#include <vector>
#include "lsst/afw/math/Interpolate.h"

namespace lsst {
namespace afw {
namespace math {
namespace detail {

#if !defined(SWIG)
class InterpolateSdssSpline : public Interpolate {
    friend class InterpolateSdssSpline;
public:
    virtual ~InterpolateSdssSpline() {}
    virtual double interpolate(double const x) const;
    virtual void interpolate(std::vector<double> const& x, std::vector<double> &y) const;
    virtual double derivative(double const x) const;
    virtual void derivative(std::vector<double>  const& x, std::vector<double> &dydx) const;
    virtual std::vector<double> roots(double const value, double const x0, double const x1) const;
protected:
    /// \brief ctor
    InterpolateSdssSpline(std::vector<double> const &x, ///< the ordinates of points
                          std::vector<double> const &y ///< the values at x[]
                         ) : Interpolate(x, y) {}

    void _allocateSpline(int const nknot);
    std::vector<double> _knots;                // positions of knots
    std::vector<std::vector<double> > _coeffs; // and associated coefficients
};
#endif

class InterpolateControlTautSpline;
    
/**
 * \brief Taut Splines
 *
 * Adapted from
 *      A Practical Guide to Splines
 * by
 *      C. de Boor (N.Y. : Springer-Verlag, 1978)
 * (His routine tautsp converted to C by Robert Lupton and then to C++ by an older and grayer Robert Lupton)
 */
#if defined(SWIG)
class InterpolateTautSpline {
public:
    enum Symmetry { UNKNOWN, ODD, EVEN };
private:
    InterpolateTautSpline();
};
#else
class InterpolateTautSpline : public InterpolateSdssSpline {
    friend PTR(Interpolate) math::makeInterpolate(std::vector<double> const &x, std::vector<double> const &y,
                                                  InterpolateControl const& ictrl);
public:
    enum Symmetry { UNKNOWN, ODD, EVEN }; ///< Specify the desired symmetry for the spline

    virtual ~InterpolateTautSpline() {}
private:
    InterpolateTautSpline(std::vector<double> const &x, std::vector<double> const &y,
                          InterpolateControlTautSpline const& ictrl);

    void _calculateTautSpline(std::vector<double> const& x,
                             std::vector<double> const& y,
                             double const gamma0);
    void _calculateTautSplineEvenOdd(std::vector<double> const& x,
                                     std::vector<double> const& y,
                                     double const gamma0,
                                     bool even);
    PTR(InterpolateControlTautSpline) _ictrl;
};
#endif
    
/*****************************************************************************/
/**
 * \brief Control the creation of an InterpolateTautSpline
 *
 */
class InterpolateControlTautSpline : public InterpolateControl {
    friend class InterpolateTautSpline;
public:
    InterpolateControlTautSpline(float gamma=0.0,
                                 InterpolateTautSpline::Symmetry symmetry=InterpolateTautSpline::UNKNOWN);
private:
    float _gamma;                     ///< How taut should I be?
    InterpolateTautSpline::Symmetry _symmetry; ///< Desired symmetry
};
    
}}}}
#endif
