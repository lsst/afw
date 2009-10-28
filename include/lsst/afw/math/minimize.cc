// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Definition of member functions for minuit adaptors
 *
 * This file is meant to be included by lsst/afw/math/minimize.h
 *
 * @author Andrew Becker and Russell Owen
 *
 * @ingroup afw
 */

#include <string> // for upar.add

#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnMigrad.h"
#include "Minuit/MnMinos.h"
#include "Minuit/MnPrint.h"

#include "lsst/pex/logging/Trace.h"

// Constructors
template<typename ReturnT>
lsst::afw::math::MinimizerFunctionBase1<ReturnT>::MinimizerFunctionBase1()
:
    lsst::daf::data::LsstBase(typeid(this)),
    _functionPtr(),
    _measurementList(),
    _varianceList(),
    _xPositionList(),
    _errorDef(1.0)
{}

template<typename ReturnT>
lsst::afw::math::MinimizerFunctionBase1<ReturnT>::MinimizerFunctionBase1(
    lsst::afw::math::Function1<ReturnT> const &function,
    std::vector<double> const &measurementList,
    std::vector<double> const &varianceList,
    std::vector<double> const &xPositionList, 
    double errorDef)
:
    lsst::daf::data::LsstBase(typeid(this)),
    _functionPtr(function.clone()),
    _measurementList(measurementList),
    _varianceList(varianceList),
    _xPositionList(xPositionList),
    _errorDef(errorDef)
{}

template<typename ReturnT>
lsst::afw::math::MinimizerFunctionBase2<ReturnT>::MinimizerFunctionBase2()
:
    lsst::daf::data::LsstBase(typeid(this)),
    _functionPtr(),
    _measurementList(),
    _varianceList(),
    _xPositionList(),
    _yPositionList(),
    _errorDef(1.0)
{}

template<typename ReturnT>
lsst::afw::math::MinimizerFunctionBase2<ReturnT>::MinimizerFunctionBase2(
    lsst::afw::math::Function2<ReturnT> const &function,
    std::vector<double> const &measurementList,
    std::vector<double> const &varianceList,
    std::vector<double> const &xPositionList,
    std::vector<double> const &yPositionList,
    double errorDef)
:
    lsst::daf::data::LsstBase(typeid(this)),
    _functionPtr(function.clone()),
    _measurementList(measurementList),
    _varianceList(varianceList),
    _xPositionList(xPositionList),
    _yPositionList(yPositionList),
    _errorDef(errorDef)
{}



// Only method we need to set up; basically this is a chi^2 routine
template<typename ReturnT>
double lsst::afw::math::MinimizerFunctionBase1<ReturnT>::operator() (const std::vector<double>& par) const {
    // Initialize the function with the fit parameters
    this->_functionPtr->setParameters(par);
    
    double chi2 = 0.0;
    for (unsigned int i = 0; i < this->_measurementList.size(); i++) {
        double resid = (*(this->_functionPtr))(this->_xPositionList[i]) - this->_measurementList[i];
        chi2 += resid * resid / this->_varianceList[i];
    }
    
    return chi2;
}


template<typename ReturnT>
double lsst::afw::math::MinimizerFunctionBase2<ReturnT>::operator() (const std::vector<double>& par) const {
    // Initialize the function with the fit parameters
    this->_functionPtr->setParameters(par);
    
    double chi2 = 0.0;
    for (unsigned int i = 0; i < this->_measurementList.size(); i++) {
        double resid = (*(this->_functionPtr))(this->_xPositionList[i], this->_yPositionList[i]) - this->_measurementList[i];
        chi2 += resid * resid / this->_varianceList[i];
    }
    
    return chi2;
}

/**
 * @brief Find the minimum of a function(x)
 *
 * @return true if minimum is valid, false otherwise
 *
 * Uses the Minuit fitting package with a standard definition of chiSq
 * (see lsst::afw::math::MinimizerFunctionBase1).
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if any input vector is the wrong length 
 *
 * To do:
 * - Document stepSizeList better
 * - Document errorDef
 * - Compute stepSize automatically? (if so, find a different way to fix parameters)
 */
template<typename ReturnT>
lsst::afw::math::FitResults lsst::afw::math::minimize(
    lsst::afw::math::Function1<ReturnT> const &function, ///< function(x) to be minimized
    std::vector<double> const &initialParameterList,    ///< initial guess for parameters
    std::vector<double> const &stepSizeList, ///< step size for each parameter; use 0.0 to fix a parameter
    std::vector<double> const &measurementList, ///< measured values
    std::vector<double> const &varianceList,    ///< variance for each measurement
    std::vector<double> const &xPositionList,   ///< x position of each measurement
    double errorDef ///< what is this?
) {
    unsigned int const nParameters = function.getNParameters();
    if (initialParameterList.size() != nParameters) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          "initialParameterList is the wrong length");
    }
    if (stepSizeList.size() != nParameters) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          "stepSizeList is the wrong length");
    }
    unsigned int const nMeasurements = measurementList.size();
    if (varianceList.size() != nMeasurements) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          "varianceList is the wrong length");
    }
    if (xPositionList.size() != nMeasurements) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          "xPositionList is the wrong length");
    }

    MinimizerFunctionBase1<ReturnT> minimizerFunc(
        function,
        measurementList,
        varianceList,
        xPositionList,
        errorDef
    );

    MnUserParameters fitPar;
    std::vector<std::string> paramNames;
    for (unsigned int i = 0; i < nParameters; ++i) {
        paramNames.push_back((boost::format("p%d") % i).str());
        fitPar.add(paramNames[i].c_str(), initialParameterList[i], stepSizeList[i]);
    }

    MnMigrad migrad(minimizerFunc, fitPar);
    FunctionMinimum min = migrad();
    MnMinos minos(minimizerFunc, min);
    
    FitResults fitResults;
    fitResults.chiSq = min.fval();
    fitResults.isValid = min.isValid() && std::isfinite(fitResults.chiSq);
    if (!fitResults.isValid) {
        lsst::pex::logging::Trace("lsst::afw::math::minimize", 1, "WARNING : Fit failed to converge");
    }
    
    for (unsigned int i = 0; i < nParameters; ++i) {
        fitResults.parameterList.push_back(min.userState().value(paramNames[i].c_str()));
        if (fitResults.isValid) {
            fitResults.parameterErrorList.push_back(minos(i));
        } else {
            double e = min.userState().error(paramNames[i].c_str());
            std::pair<double,double> ep(-1 * e, e);
            fitResults.parameterErrorList.push_back(ep);
        }
    }
    return fitResults;
}


/**
 * @brief Find the minimum of a function(x, y)
 *
 * Uses the Minuit fitting package with a standard definition of chiSq.
 * (see lsst::afw::math::MinimizerFunctionBase2).
 *
 * @return true if minimum is valid, false otherwise
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if any input vector is the wrong length 
 *
 * To do:
 * - Document stepSizeList better
 * - Document errorDef
 * - Compute stepSize automatically? (if so, find a different way to fix parameters)
 */
template<typename ReturnT>
lsst::afw::math::FitResults lsst::afw::math::minimize(
    lsst::afw::math::Function2<ReturnT> const &function,  ///< function(x,y) to be minimized
    std::vector<double> const &initialParameterList,    ///< initial guess for parameters
    std::vector<double> const &stepSizeList,        ///< step size for each parameter; use 0.0 to fix a parameter
    std::vector<double> const &measurementList, ///< measured values
    std::vector<double> const &varianceList,    ///< variance for each measurement
    std::vector<double> const &xPositionList,   ///< x position of each measurement
    std::vector<double> const &yPositionList,   ///< y position of each measurement
    double errorDef ///< what is this?
) {
    unsigned int const nParameters = function.getNParameters();
    if (initialParameterList.size() != nParameters) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, "initialParameterList is the wrong length");
    }
    if (stepSizeList.size() != nParameters) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, "stepSizeList is the wrong length");
    }
    unsigned int const nMeasurements = measurementList.size();
    if (varianceList.size() != nMeasurements) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, "varianceList is the wrong length");
    }
    if (xPositionList.size() != nMeasurements) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, "xPositionList is the wrong length");
    }
    if (yPositionList.size() != nMeasurements) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, "yPositionList is the wrong length");
    }

    MinimizerFunctionBase2<ReturnT> minimizerFunc(
        function,
        measurementList,
        varianceList,
        xPositionList,
        yPositionList,
        errorDef
    );

    MnUserParameters fitPar;
    std::vector<std::string> paramNames;
    for (unsigned int i = 0; i < nParameters; ++i) {
        paramNames.push_back((boost::format("p%d") % i).str());
        fitPar.add(paramNames[i].c_str(), initialParameterList[i], stepSizeList[i]);
    }

    MnMigrad migrad(minimizerFunc, fitPar);
    FunctionMinimum min = migrad();
    MnMinos minos(minimizerFunc, min);
    
    FitResults fitResults;
    fitResults.chiSq = min.fval();
    fitResults.isValid = min.isValid() && std::isfinite(fitResults.chiSq);
    if (!fitResults.isValid) {
        lsst::pex::logging::Trace("lsst::afw::math::minimize", 1, "WARNING : Fit failed to converge");
    }
    for (unsigned int i = 0; i < nParameters; ++i) {
        fitResults.parameterList.push_back(min.userState().value(paramNames[i].c_str()));
        if (fitResults.isValid) {
            fitResults.parameterErrorList.push_back(minos(i));
        } else {
            double e = min.userState().error(paramNames[i].c_str());
            std::pair<double,double> ep(-1 * e, e);
            fitResults.parameterErrorList.push_back(ep);
        }
    }
    return fitResults;
}
