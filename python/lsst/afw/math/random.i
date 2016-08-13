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

%{
#include "lsst/afw/math/Random.h"
%}

%shared_ptr(lsst::afw::math::Random);


%extend lsst::afw::math::Random {

    PyObject * getState() const {
        std::string state = self->getState();
        return PyBytes_FromStringAndSize(state.data(), state.size());
    }

    void setState(PyObject * state) {
        char * buffer = nullptr;
        Py_ssize_t len = 0;
        if (PyBytes_AsStringAndSize(state, &buffer, &len) == 0) {
            std::string state(buffer, len);
            self->setState(state);
        }
    }

}

%ignore lsst::afw::math::Random::getState;
%ignore lsst::afw::math::Random::setState;

%include "lsst/afw/math/Random.h"

%define %randomImage(TYPE)
%template(randomUniformImage)    lsst::afw::math::randomUniformImage<lsst::afw::image::Image<TYPE> >;
%template(randomUniformPosImage) lsst::afw::math::randomUniformPosImage<lsst::afw::image::Image<TYPE> >;
%template(randomUniformIntImage) lsst::afw::math::randomUniformIntImage<lsst::afw::image::Image<TYPE> >;
%template(randomFlatImage)       lsst::afw::math::randomFlatImage<lsst::afw::image::Image<TYPE> >;
%template(randomGaussianImage)   lsst::afw::math::randomGaussianImage<lsst::afw::image::Image<TYPE> >;
%template(randomChisqImage)      lsst::afw::math::randomChisqImage<lsst::afw::image::Image<TYPE> >;
%template(randomPoissonImage)    lsst::afw::math::randomPoissonImage<lsst::afw::image::Image<TYPE> >;
%enddef

%randomImage(double)
%randomImage(float)
