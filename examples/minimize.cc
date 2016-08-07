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

#include <iostream>
#include <vector>
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/math/minimize.h"

template<class T>
void printVector(std::vector<T> v) {
    for (unsigned int ii = 0; ii < v.size(); ++ii) {
        std::cout << "  " << v[ii];
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    typedef double FuncReturn;
    const unsigned int order = 3;
    const unsigned int npts = 10;

    lsst::afw::math::Chebyshev1Function1<FuncReturn> chebyFunc(order);

    const unsigned int nParams = chebyFunc.getNParameters();
    std::vector<double> modelParams = chebyFunc.getParameters();
    for (unsigned int ii = 0; ii < nParams; ++ii) {
        modelParams[ii] = static_cast<double>(ii) * 0.1;
    }
    chebyFunc.setParameters(modelParams);
    std::cout << "Input : Chebychev polynomial of the first kind with parameters: " << std::endl;
    printVector(modelParams);

    std::vector<double> measurements(npts);
    std::vector<double> variances(npts);
    std::vector<double> positions(npts);

    double x = -1.0;
    double deltaX = 2.0 / static_cast<double>(npts - 1);
    for (unsigned int i = 0; i < npts; i++, x += deltaX) {
        measurements[i] = chebyFunc(x);
        variances[i] = 0.1;
        positions[i] = x;
    }

    // add a bit of randomization to model parameters for initial parameters
    // and set stepSizes
    std::vector<double> initialParams = modelParams;
    std::vector<double> stepSizes(nParams);
    for (unsigned int ii = 0; ii < nParams; ++ii) {
        initialParams[ii] += 0.5 * double(rand())/RAND_MAX;
        stepSizes[ii] = 0.1;
    }
    std::cout << "Initial guess:" << std::endl;
    printVector(initialParams);
    std::cout << "Step size:" << std::endl;
    printVector(stepSizes);

    double errorDef = 1.0;

    lsst::afw::math::FitResults fitResults = lsst::afw::math::minimize(
        chebyFunc,
        initialParams,
        stepSizes,
        measurements,
        variances,
        positions,
        errorDef
    );

    std::vector<double> fitParams = chebyFunc.getParameters();
    std::cout << "fitResults.isValid =" << fitResults.isValid << std::endl;
    std::cout << "fitResults.parameterList:" << std::endl;
    printVector(fitResults.parameterList);
    std::vector<double> negErrors;
    std::vector<double> posErrors;
    for (unsigned int ii = 0; ii < nParams; ++ii) {
        negErrors.push_back(fitResults.parameterErrorList[ii].first);
        posErrors.push_back(fitResults.parameterErrorList[ii].second);
    }
    std::cout << "Negative parameter errors:" << std::endl;
    printVector(negErrors);
    std::cout << "Positive parameter errors:" << std::endl;
    printVector(posErrors);


//    MnMigrad migrad(myFcn, upar);
//    FunctionMinimum min = migrad();
//
//    MnMinos minos(myFcn, min);
//
//    std::pair<double,double> e0 = minos(0);
//    std::pair<double,double> e1 = minos(1);
//    std::pair<double,double> e2 = minos(2);
//    std::pair<double,double> e3 = minos(3);
//
//    cout << "Best fit:" << endl;
//    std::cout<<"par0: "<<min.userState().value("p0")<<" "<<e0.first<<" "<<e0.second<<std::endl;
//    std::cout<<"par1: "<<min.userState().value("p1")<<" "<<e1.first<<" "<<e1.second<<std::endl;
//    std::cout<<"par2: "<<min.userState().value("p2")<<" "<<e2.first<<" "<<e2.second<<std::endl;
//    std::cout<<"par3: "<<min.userState().value("p3")<<" "<<e3.first<<" "<<e3.second<<std::endl;
//
//    // Try fitting for a number with no variation
//    // Try fitting for a number with no variation
//    // Try fitting for a number with no variation
//    // Try fitting for a number with no variation
//    MnUserParameters upar2;
//    upar2.add("p0", 1, 0.1);
//    std::vector<double> measurements2(npts);
//    std::vector<double> variances2(npts);
//    std::vector<double> positions2(npts);
//    x = -1.;
//    for (unsigned int i = 0; i < npts; i++, x += 0.2) {
//        measurements[i] = 1.;
//        variances[i] = 0.1;
//        positions[i] = x;
//    }
//    def = 1.0;
//
//
//    const unsigned int polyorder = 0;
//    std::shared_ptr<lsst::afw::math::PolynomialFunction1<FuncReturn> > polyFuncPtr(
//        new lsst::afw::math::PolynomialFunction1<FuncReturn>(polyorder)
//        );
//
//    lsst::afw::math::MinimizerFunctionBase1<FuncReturn> myFcn2(
//        measurements, variances, positions, def, polyFuncPtr);
//
//    MnMigrad migrad2(myFcn2, upar2);
//    FunctionMinimum min2 = migrad2();
//    MnMinos minos2(myFcn2, min2);
//
//    std::pair<double,double> e02 = minos2(0);
//
//    cout << "Best fit:" << endl;
//    std::cout<<"par0: "<<min2.userState().value("p0")<<" "<<e02.first<<" "<<e02.second<<std::endl;

    return 0;
}
