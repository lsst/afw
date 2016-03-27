// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
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
#include "boost/shared_ptr.hpp"
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
