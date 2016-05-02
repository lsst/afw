// -*- LSST-C++ -*-

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
 
#ifndef LSST_AFW_MATH_MINIMIZE_H
#define LSST_AFW_MATH_MINIMIZE_H
/**
 * @file
 *
 * @brief Adaptor for minuit
 *
 * Class that Minuit knows how to minimize, that contains an lsst::afw::math::Function
 *
 * @author Andrew Becker and Russell Owen
 *
 * @ingroup afw
 */
#include <memory>
#include "Minuit2/FCNBase.h"

#include "lsst/daf/base/Citizen.h"
#include "lsst/afw/math/Function.h"

namespace lsst {
namespace afw {
namespace math {

    /**
     * @brief Results from minimizing a function
     */ 
    struct FitResults {
    public:
        bool isValid;   ///< true if the fit converged; false otherwise
        double chiSq;   ///< chi squared; may be nan or infinite, but only if isValid false
        std::vector<double> parameterList; ///< fit parameters
        std::vector<std::pair<double,double> > parameterErrorList; ///< negative,positive (1 sigma?) error for each parameter
    };
        
    template<typename ReturnT>
    FitResults minimize(
        lsst::afw::math::Function1<ReturnT> const &function,
        std::vector<double> const &initialParameterList,
        std::vector<double> const &stepSizeList,
        std::vector<double> const &measurementList,
        std::vector<double> const &varianceList,
        std::vector<double> const &xPositionList,
        double errorDef
    );

    template<typename ReturnT>
    FitResults minimize(
        lsst::afw::math::Function2<ReturnT> const &function,
        std::vector<double> const &initialParameterList,
        std::vector<double> const &stepSizeList,
        std::vector<double> const &measurementList,
        std::vector<double> const &varianceList,
        std::vector<double> const &xPositionList,
        std::vector<double> const &yPositionList,
        double errorDef
    );
    
}}}   // lsst::afw::math

#endif // !defined(LSST_AFW_MATH_MINIMIZE_H)
