/** \file
 *
 * Support routines for
 */
#include "lsst/afw/math/FunctionLibrary.h"

namespace afwMath = lsst::afw::math;

/**
 * Return the coefficients of the Function's parameters, evaluated at (x, y)
 * I.e. given c0, c1, c2, c3 ... return c0, c1 x, c2 y, c3 x^2 ...
 */
template<typename ReturnT>
std::vector<double> afwMath::PolynomialFunction2<ReturnT>::getDFuncDParameters(double x, double y) const {
    int const nOrder = _order;
    std::vector<double> coeffs((nOrder + 1)*(nOrder + 2)/2);

    //
    // Go through params order by order, evaluating c x^r y^s;  we do this by first evaluating
    // c x^r for a complete order, then going through again multiplying by y^s
    //
    int i0 = 0;                         // starting index for this order's coefficients
    for (int order = 0; order <= nOrder; ++order) {
        coeffs[i0] = 1;
        double zn = y;                  // y^s
        for (int i = 1; i <= order; ++i) {
            coeffs[i0 + i] = zn;
            zn *= y;
        }

        zn = x;                         // x^r
        for (int i = order - 1; i >= 0; --i) {
            coeffs[i0 + i] *= zn;
            zn *= x;
        }


        i0 += order + 1;
    }

    assert (i0 == coeffs.size());

    return coeffs;
}

/************************************************************************************************************/
#define INSTANTIATE(TYPE) \
    template std::vector<double> \
    afwMath::PolynomialFunction2<TYPE>::getDFuncDParameters(double x, double y) const

INSTANTIATE(double);
INSTANTIATE(float);
