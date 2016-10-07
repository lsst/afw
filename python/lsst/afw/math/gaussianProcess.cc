/* 
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */

#include <pybind11/pybind11.h>
//#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/afw/math/GaussianProcess.h"

PYBIND11_DECLARE_HOLDER_TYPE(MyType, std::shared_ptr<MyType>);

namespace py = pybind11;

using namespace lsst::afw::math;

template <typename T>
void declareKdTree(py::module &mod, const std::string & suffix) {
    py::class_<KdTree<T>> clsKdTree(mod, ("KdTree"+suffix).c_str());
    clsKdTree.def(py::init<>());
    clsKdTree.def("Initialize", &KdTree<T>::Initialize);
    clsKdTree.def("removePoint", &KdTree<T>::removePoint);
    //clsKdTree.def("getData", &KdTree<T>::getData);
    clsKdTree.def("getData", [](KdTree<T> &t, int ipt, int idim) -> T {
            return t.getData(ipt, idim);
    });
    //clsKdTree.def("", &KdTree<T>::);
    //clsKdTree.def("", &KdTree<T>::);
    //clsKdTree.def("", &KdTree<T>::);
    //clsKdTree.def("", &KdTree<T>::);
};

template <typename T>
void declareCovariograms(py::module &mod, const std::string & suffix) {
    /* Covariogram */
    py::class_<Covariogram<T>, std::shared_ptr<Covariogram<T>> > clsCovariogram(mod, ("Covariogram"+suffix).c_str());
    clsCovariogram.def(py::init<>());
    clsCovariogram.def("__call__", &Covariogram<T>::operator());

    /* SquaredExpCovariogram */
    py::class_<SquaredExpCovariogram<T>, std::shared_ptr<SquaredExpCovariogram<T>>, Covariogram<T>>
        clsSquaredExpCovariogram(mod, ("SquaredExpCovariogram"+suffix).c_str());
    clsSquaredExpCovariogram.def(py::init<>());
    clsSquaredExpCovariogram.def("__call__", &SquaredExpCovariogram<T>::operator());
    clsSquaredExpCovariogram.def("setEllSquared", &SquaredExpCovariogram<T>::setEllSquared);
    
    /* SquaredExpCovariogram */
    py::class_<NeuralNetCovariogram<T>, std::shared_ptr<NeuralNetCovariogram<T>>, Covariogram<T>>
        clsNeuralNetCovariogram(mod, ("NeuralNetCovariogram"+suffix).c_str());
    clsNeuralNetCovariogram.def(py::init<>());
    clsNeuralNetCovariogram.def("setSigma0", &NeuralNetCovariogram<T>::setSigma0);
    clsNeuralNetCovariogram.def("setSigma1", &NeuralNetCovariogram<T>::setSigma1);
};

template <typename T>
void declareGaussianProcess(py::module &mod, const std::string & suffix) {
    py::class_<GaussianProcess<T>> clsGaussianProcess(mod, ("GaussianProcess"+suffix).c_str());
    /* Constructors */
    clsGaussianProcess.def(py::init<ndarray::Array<T,2,2> const &,
                                    ndarray::Array<T,1,1> const &,
                                    std::shared_ptr< Covariogram<T> > const &>()
    );
    clsGaussianProcess.def(py::init<ndarray::Array<T,2,2> const &,
                                    ndarray::Array<T,1,1> const &,
                                    ndarray::Array<T,1,1> const &,
                                    ndarray::Array<T,1,1> const &,
                                    std::shared_ptr< Covariogram<T> > const &>()
    );
    clsGaussianProcess.def(py::init<ndarray::Array<T,2,2> const &,
                                    ndarray::Array<T,2,2> const &,
                                    std::shared_ptr< Covariogram<T> > const &>()
    );
    clsGaussianProcess.def(py::init<ndarray::Array<T,2,2> const &,
                                    ndarray::Array<T,1,1> const &,
                                    ndarray::Array<T,1,1> const &,
                                    ndarray::Array<T,2,2> const &,
                                    std::shared_ptr< Covariogram<T> > const &>()
    );
    /* Members */
    clsGaussianProcess.def("interpolate", 
                           (T (GaussianProcess<T>::*)(ndarray::Array<T,1,1>,
                                                      ndarray::Array<T,1,1> const &,
                                                      int) const) &GaussianProcess<T>::interpolate
    );
    clsGaussianProcess.def("interpolate",
                           (void (GaussianProcess<T>::*)(ndarray::Array<T,1,1>,
                                                      ndarray::Array<T,1,1>,
                                                      ndarray::Array<T,1,1> const &,
                                                      int) const) &GaussianProcess<T>::interpolate
    );
    clsGaussianProcess.def("selfInterpolate",
                           (T (GaussianProcess<T>::*)(ndarray::Array<T,1,1>,
                                                      int,
                                                      int) const) &GaussianProcess<T>::selfInterpolate
    );
    clsGaussianProcess.def("selfInterpolate",
                           (void (GaussianProcess<T>::*)(ndarray::Array<T,1,1>,
                                                      ndarray::Array<T,1,1>,
                                                      int,
                                                      int) const) &GaussianProcess<T>::selfInterpolate
    );
    clsGaussianProcess.def("setLambda", &GaussianProcess<T>::setLambda);
    clsGaussianProcess.def("setCovariogram", &GaussianProcess<T>::setCovariogram);
    clsGaussianProcess.def("addPoint",
        (void (GaussianProcess<T>::*)(ndarray::Array<T,1,1> const &, T)) &GaussianProcess<T>::addPoint
    );
    clsGaussianProcess.def("addPoint",
        (void (GaussianProcess<T>::*)(ndarray::Array<T,1,1> const &,
                                      ndarray::Array<T,1,1> const &)) &GaussianProcess<T>::addPoint
    );
    clsGaussianProcess.def("batchInterpolate",
        (void (GaussianProcess<T>::*)(ndarray::Array<T,1,1>,
                                      ndarray::Array<T,1,1>,
                                      ndarray::Array<T,2,2> const &) const)
                                          &GaussianProcess<T>::batchInterpolate
    );
    clsGaussianProcess.def("batchInterpolate",
        (void (GaussianProcess<T>::*)(ndarray::Array<T,1,1>,
                                      ndarray::Array<T,2,2> const &) const)
                                          &GaussianProcess<T>::batchInterpolate
    );
    clsGaussianProcess.def("batchInterpolate", 
        (void (GaussianProcess<T>::*)(ndarray::Array<T,2,2>,
                                      ndarray::Array<T,2,2>,
                                      ndarray::Array<T,2,2> const &) const)
                                          &GaussianProcess<T>::batchInterpolate
    );
    clsGaussianProcess.def("batchInterpolate", 
        (void (GaussianProcess<T>::*)(ndarray::Array<T,2,2>,
                                      ndarray::Array<T,2,2> const &) const)
                                          &GaussianProcess<T>::batchInterpolate
    );
    clsGaussianProcess.def("setKrigingParameter", &GaussianProcess<T>::setKrigingParameter);
    clsGaussianProcess.def("removePoint", &GaussianProcess<T>::removePoint);
};

PYBIND11_PLUGIN(_gaussianProcess) {
    py::module mod("_gaussianProcess", "Python wrapper for afw _gaussianProcess library");
    
    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
    };

    declareCovariograms<double>(mod, "D");
    declareGaussianProcess<double>(mod, "D");
    declareKdTree<double>(mod, "D");

    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}