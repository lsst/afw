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

/*
 * Definition of member functions for minuit adaptors
 *
 * This file is meant to be included by lsst/afw/math/minimize.h
 */

#include <string>

#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MnPrint.h"

#include "lsst/log/Log.h"
#include "lsst/afw/math/minimize.h"

namespace afwMath = lsst::afw::math;

namespace {
    /*
     * Minuit wrapper for a function(x)
     */
    template<typename ReturnT>
    class MinimizerFunctionBase1 : public ROOT::Minuit2::FCNBase, public lsst::daf::base::Citizen {
    public:
        explicit MinimizerFunctionBase1(
            afwMath::Function1<ReturnT> const &function,
            std::vector<double> const &measurementList,
            std::vector<double> const &varianceList,
            std::vector<double> const &xPositionList,
            double errorDef
        );
        virtual ~MinimizerFunctionBase1() {};
        // Required by ROOT::Minuit2::FCNBase
        virtual double Up() const { return _errorDef; }
        virtual double operator() (const std::vector<double>&) const;

#if 0                                   // not used
        inline std::vector<double> getMeasurements() const {return _measurementList;}
        inline std::vector<double> getVariances() const {return _varianceList;}
        inline std::vector<double> getPositions() const {return _xPositionList;}
        inline void setErrorDef(double def) {_errorDef=def;}
#endif
    private:
        std::shared_ptr<afwMath::Function1<ReturnT> > _functionPtr;
        std::vector<double> _measurementList;
        std::vector<double> _varianceList;
        std::vector<double> _xPositionList;
        double _errorDef;
    };

    /*
     * Minuit wrapper for a function(x, y)
     */
    template<typename ReturnT>
    class MinimizerFunctionBase2 : public ROOT::Minuit2::FCNBase, public lsst::daf::base::Citizen {
    public:
        explicit MinimizerFunctionBase2(
            afwMath::Function2<ReturnT> const &function,
            std::vector<double> const &measurementList,
            std::vector<double> const &varianceList,
            std::vector<double> const &xPositionList,
            std::vector<double> const &yPositionList,
            double errorDef
        );
        virtual ~MinimizerFunctionBase2() {};
        // Required by ROOT::Minuit2::FCNBase
        virtual double Up() const { return _errorDef; }
        virtual double operator() (const std::vector<double>& par) const;

#if 0                                   // not used
        inline std::vector<double> getMeasurements() const {return _measurementList;}
        inline std::vector<double> getVariances() const {return _varianceList;}
        inline std::vector<double> getPosition1() const {return _xPositionList;}
        inline std::vector<double> getPosition2() const {return _yPositionList;}
        inline void setErrorDef(double def) {_errorDef=def;}
#endif
    private:
        std::shared_ptr<afwMath::Function2<ReturnT> > _functionPtr;
        std::vector<double> _measurementList;
        std::vector<double> _varianceList;
        std::vector<double> _xPositionList;
        std::vector<double> _yPositionList;
        double _errorDef;
    };
}
/// @cond
template<typename ReturnT>
MinimizerFunctionBase1<ReturnT>::MinimizerFunctionBase1(
    lsst::afw::math::Function1<ReturnT> const &function,
    std::vector<double> const &measurementList,
    std::vector<double> const &varianceList,
    std::vector<double> const &xPositionList,
    double errorDef)
:
    lsst::daf::base::Citizen(typeid(this)),
    _functionPtr(function.clone()),
    _measurementList(measurementList),
    _varianceList(varianceList),
    _xPositionList(xPositionList),
    _errorDef(errorDef)
{}

template<typename ReturnT>
MinimizerFunctionBase2<ReturnT>::MinimizerFunctionBase2(
    lsst::afw::math::Function2<ReturnT> const &function,
    std::vector<double> const &measurementList,
    std::vector<double> const &varianceList,
    std::vector<double> const &xPositionList,
    std::vector<double> const &yPositionList,
    double errorDef)
:
    lsst::daf::base::Citizen(typeid(this)),
    _functionPtr(function.clone()),
    _measurementList(measurementList),
    _varianceList(varianceList),
    _xPositionList(xPositionList),
    _yPositionList(yPositionList),
    _errorDef(errorDef)
{}



// Only method we need to set up; basically this is a chi^2 routine
template<typename ReturnT>
double MinimizerFunctionBase1<ReturnT>::operator() (const std::vector<double>& par) const {
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
double MinimizerFunctionBase2<ReturnT>::operator() (const std::vector<double>& par) const {
    // Initialize the function with the fit parameters
    this->_functionPtr->setParameters(par);

    double chi2 = 0.0;
    for (unsigned int i = 0; i < this->_measurementList.size(); i++) {
        double resid = (*(this->_functionPtr))(this->_xPositionList[i],
                                               this->_yPositionList[i]) - this->_measurementList[i];
        chi2 += resid * resid / this->_varianceList[i];
    }

    return chi2;
}
/// @endcond

template<typename ReturnT>
afwMath::FitResults afwMath::minimize(
    lsst::afw::math::Function1<ReturnT> const &function,
    std::vector<double> const &initialParameterList,
    std::vector<double> const &stepSizeList,
    std::vector<double> const &measurementList,
    std::vector<double> const &varianceList,
    std::vector<double> const &xPositionList,
    double errorDef
) {
    unsigned int const nParameters = function.getNParameters();
    if (initialParameterList.size() != nParameters) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "initialParameterList is the wrong length");
    }
    if (stepSizeList.size() != nParameters) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "stepSizeList is the wrong length");
    }
    unsigned int const nMeasurements = measurementList.size();
    if (varianceList.size() != nMeasurements) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "varianceList is the wrong length");
    }
    if (xPositionList.size() != nMeasurements) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "xPositionList is the wrong length");
    }

    MinimizerFunctionBase1<ReturnT> minimizerFunc(
        function,
        measurementList,
        varianceList,
        xPositionList,
        errorDef
    );

    ROOT::Minuit2::MnUserParameters fitPar;
    std::vector<std::string> paramNames;
    for (unsigned int i = 0; i < nParameters; ++i) {
        paramNames.push_back((boost::format("p%d") % i).str());
        fitPar.Add(paramNames[i].c_str(), initialParameterList[i], stepSizeList[i]);
    }

    ROOT::Minuit2::MnMigrad migrad(minimizerFunc, fitPar);
    ROOT::Minuit2::FunctionMinimum min = migrad();
    ROOT::Minuit2::MnMinos minos(minimizerFunc, min);

    FitResults fitResults;
    fitResults.chiSq = min.Fval();
    fitResults.isValid = min.IsValid() && std::isfinite(fitResults.chiSq);
    if (!fitResults.isValid) {
        LOGL_WARN("afw.math.minimize", "Fit failed to converge");
    }

    for (unsigned int i = 0; i < nParameters; ++i) {
        fitResults.parameterList.push_back(min.UserState().Value(paramNames[i].c_str()));
        if (fitResults.isValid) {
            fitResults.parameterErrorList.push_back(minos(i));
        } else {
            double e = min.UserState().Error(paramNames[i].c_str());
            std::pair<double,double> ep(-1 * e, e);
            fitResults.parameterErrorList.push_back(ep);
        }
    }
    return fitResults;
}


template<typename ReturnT>
afwMath::FitResults afwMath::minimize(
    lsst::afw::math::Function2<ReturnT> const &function,
    std::vector<double> const &initialParameterList,
    std::vector<double> const &stepSizeList,
    std::vector<double> const &measurementList,
    std::vector<double> const &varianceList,
    std::vector<double> const &xPositionList,
    std::vector<double> const &yPositionList,
    double errorDef
) {
    unsigned int const nParameters = function.getNParameters();
    if (initialParameterList.size() != nParameters) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "initialParameterList is the wrong length");
    }
    if (stepSizeList.size() != nParameters) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "stepSizeList is the wrong length");
    }
    unsigned int const nMeasurements = measurementList.size();
    if (varianceList.size() != nMeasurements) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "varianceList is the wrong length");
    }
    if (xPositionList.size() != nMeasurements) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "xPositionList is the wrong length");
    }
    if (yPositionList.size() != nMeasurements) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "yPositionList is the wrong length");
    }

    MinimizerFunctionBase2<ReturnT> minimizerFunc(
        function,
        measurementList,
        varianceList,
        xPositionList,
        yPositionList,
        errorDef
    );

    ROOT::Minuit2::MnUserParameters fitPar;
    std::vector<std::string> paramNames;
    for (unsigned int i = 0; i < nParameters; ++i) {
        paramNames.push_back((boost::format("p%d") % i).str());
        fitPar.Add(paramNames[i].c_str(), initialParameterList[i], stepSizeList[i]);
    }

    ROOT::Minuit2::MnMigrad migrad(minimizerFunc, fitPar);
    ROOT::Minuit2::FunctionMinimum min = migrad();
    ROOT::Minuit2::MnMinos minos(minimizerFunc, min);

    FitResults fitResults;
    fitResults.chiSq = min.Fval();
    fitResults.isValid = min.IsValid() && std::isfinite(fitResults.chiSq);
    if (!fitResults.isValid) {
        LOGL_WARN("afw.math.minimize", "Fit failed to converge");
    }

    for (unsigned int i = 0; i < nParameters; ++i) {
        fitResults.parameterList.push_back(min.UserState().Value(paramNames[i].c_str()));
        if (fitResults.isValid) {
            fitResults.parameterErrorList.push_back(minos(i));
        } else {
            double e = min.UserState().Error(paramNames[i].c_str());
            std::pair<double,double> ep(-1 * e, e);
            fitResults.parameterErrorList.push_back(ep);
        }
    }
    return fitResults;
}

// Explicit instantiation
/// @cond
#define NL /* */
#define minimizeFuncs(ReturnT) \
    template afwMath::FitResults afwMath::minimize( \
        afwMath::Function1<ReturnT> const &, \
        std::vector<double> const &,         \
        std::vector<double> const &, \
        std::vector<double> const &, \
        std::vector<double> const &, \
        std::vector<double> const &, \
        double \
    ); NL \
    template afwMath::FitResults afwMath::minimize( \
        afwMath::Function2<ReturnT> const &, \
        std::vector<double> const &, \
        std::vector<double> const &, \
        std::vector<double> const &, \
        std::vector<double> const &, \
        std::vector<double> const &, \
        std::vector<double> const &, \
        double \
    );

minimizeFuncs(float)
minimizeFuncs(double)
/// @endcond
