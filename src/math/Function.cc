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
 
/** \file
 *
 * Support routines for
 */
#include "lsst/afw/math/FunctionLibrary.h"

namespace afwMath = lsst::afw::math;

/**
 * Return the coefficients of the Function's parameters, evaluated at (x, y)
 * I.e. given c0, c1, c2, c3 ... return 1, x, y, x^2 ...
 */
template<typename ReturnT>
std::vector<double> afwMath::PolynomialFunction2<ReturnT>::getDFuncDParameters(double x, double y) const {
    int const nOrder = _order;
    std::vector<double> coeffs((nOrder + 1)*(nOrder + 2)/2);

    //
    // Go through params order by order, evaluating x^r y^s;  we do this by first evaluating
    // y^s for a complete order, then going through again multiplying by x^r
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

    assert (i0 == static_cast<int>(coeffs.size()));

    return coeffs;
}

/************************************************************************************************************/
/**
 * Compute a cache for Lanczos kernels
 */
template<typename ReturnT>
void afwMath::LanczosFunction1<ReturnT>::computeCache(int const cacheSize) {
    if (cacheSize <= 0) {
        _kernelCache.erase(_kernelCache.begin(), _kernelCache.end());
        return;
    }

    if (cacheSize < static_cast<int>(_kernelCache.size())) {
        _kernelCache.erase(_kernelCache.begin() + cacheSize, _kernelCache.end());
    } else {
        _kernelCache.reserve(cacheSize);
        for (int i = _kernelCache.size(); i != cacheSize; ++i) {
            _kernelCache[i].resize(_n);
        }
    }
    //
    // Actually fill the cache
    //
#if 0
    std::<double> x;
    for (int i = 0; i != cacheSize; ++i) {
        this->setParameter(0, i/static_cast<double>(cacheSize - 1));
        fillVector(this->_kernelX, _kernelCache[i]);
    }
#endif
}

template<typename ReturnT>
std::vector<std::vector<ReturnT> > afwMath::LanczosFunction1<ReturnT>::_kernelCache;

/**
 * Fill the vector values with the value of the function at the points x
 *
 * \note No array bound checking is performed;  the vectors are assumed to be the same size
 */
template<typename ReturnT>
double afwMath::LanczosFunction1<ReturnT>::fillVector(std::vector<double> const& x,
                                                      std::vector<ReturnT> &values) const
{
    double sum = 0.0;
    int const cacheSize = _kernelCache.size();
    if (cacheSize == 0) {
        for (unsigned int i = 0; i != x.size(); ++i) {
            ReturnT val = operator()(x[i]);
            values[i] = val;
            sum += val;
        }
    } else {
        std::cout << "Using cache" << std::endl;

        std::vector<ReturnT> &cachedValues = _kernelCache.at(this->getParameter(0)*cacheSize);
        for (unsigned int i = 0; i != x.size(); ++i) {
            ReturnT val = cachedValues[i];
            values[i] = val;
            sum += val;
        }
    }

    return sum;
}

/************************************************************************************************************/
#define INSTANTIATE(TYPE) \
    template std::vector<double> \
    afwMath::PolynomialFunction2<TYPE>::getDFuncDParameters(double x, double y) const; \
    template std::vector<std::vector<TYPE> > afwMath::LanczosFunction1<TYPE>::_kernelCache; \
    template void afwMath::LanczosFunction1<TYPE>::computeCache(int const n); \
    template double afwMath::LanczosFunction1<TYPE>::fillVector(std::vector<double> const& x, \
                                                                std::vector<TYPE> &values) const

INSTANTIATE(double);
INSTANTIATE(float);
