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
#include <fstream>
#include <sstream>

#include "boost/format.hpp"
#if defined(__ICC)
#pragma warning(push)
#pragma warning(disable : 444)
#endif
#include "boost/archive/text_oarchive.hpp"
#if defined(__ICC)
#pragma warning(pop)
#endif

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"

namespace pexExcept = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace math {

generic_kernel_tag generic_kernel_tag_;  ///< Used as default value in argument lists
deltafunction_kernel_tag deltafunction_kernel_tag_;
///< Used as default value in argument lists

//
// Constructors
//
Kernel::Kernel()
        : daf::base::Citizen(typeid(this)),
          _spatialFunctionList(),
          _width(0),
          _height(0),
          _ctrX(0),
          _ctrY(0),
          _nKernelParams(0) {}

Kernel::Kernel(int width, int height, unsigned int nKernelParams, SpatialFunction const &spatialFunction)
        : daf::base::Citizen(typeid(this)),
          _spatialFunctionList(),
          _width(width),
          _height(height),
          _ctrX((width - 1) / 2),
          _ctrY((height - 1) / 2),
          _nKernelParams(nKernelParams) {
    if ((width < 1) || (height < 1)) {
        std::ostringstream os;
        os << "kernel height = " << height << " and/or width = " << width << " < 1";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    if (dynamic_cast<NullSpatialFunction const *>(&spatialFunction)) {
        // spatialFunction is not really present
    } else {
        if (nKernelParams == 0) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterError, "Kernel function has no parameters");
        }
        for (unsigned int ii = 0; ii < nKernelParams; ++ii) {
            SpatialFunctionPtr spatialFunctionCopy = spatialFunction.clone();
            this->_spatialFunctionList.push_back(spatialFunctionCopy);
        }
    }
}

double Kernel::computeImage(image::Image<Pixel> &image, bool doNormalize, double x, double y) const {
    if (image.getDimensions() != this->getDimensions()) {
        std::ostringstream os;
        os << "image dimensions = ( " << image.getWidth() << ", " << image.getHeight() << ") != ("
           << this->getWidth() << ", " << this->getHeight() << ") = kernel dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    image.setXY0(-_ctrX, -_ctrY);
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(x, y);
    }
    return doComputeImage(image, doNormalize);
}

Kernel::Kernel(int width, int height, std::vector<SpatialFunctionPtr> spatialFunctionList)
        : daf::base::Citizen(typeid(this)),
          _width(width),
          _height(height),
          _ctrX(width / 2),
          _ctrY(height / 2),
          _nKernelParams(spatialFunctionList.size()) {
    if ((width < 1) || (height < 1)) {
        std::ostringstream os;
        os << "kernel height = " << height << " and/or width = " << width << " < 1";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    for (unsigned int ii = 0; ii < spatialFunctionList.size(); ++ii) {
        SpatialFunctionPtr spatialFunctionCopy = spatialFunctionList[ii]->clone();
        this->_spatialFunctionList.push_back(spatialFunctionCopy);
    }
}

//
// Public Member Functions
//
void Kernel::setSpatialParameters(const std::vector<std::vector<double> > params) {
    // Check params size before changing anything
    unsigned int nKernelParams = this->getNKernelParameters();
    if (params.size() != nKernelParams) {
        throw LSST_EXCEPT(
                pexExcept::InvalidParameterError,
                (boost::format("params has %d entries instead of %d") % params.size() % nKernelParams).str());
    }
    unsigned int nSpatialParams = this->getNSpatialParameters();
    for (unsigned int ii = 0; ii < nKernelParams; ++ii) {
        if (params[ii].size() != nSpatialParams) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                              (boost::format("params[%d] has %d entries instead of %d") % ii %
                               params[ii].size() % nSpatialParams)
                                      .str());
        }
    }
    // Set parameters
    if (nSpatialParams > 0) {
        for (unsigned int ii = 0; ii < nKernelParams; ++ii) {
            this->_spatialFunctionList[ii]->setParameters(params[ii]);
        }
    }
}

void Kernel::computeKernelParametersFromSpatialModel(std::vector<double> &kernelParams, double x,
                                                     double y) const {
    std::vector<double>::iterator paramIter = kernelParams.begin();
    std::vector<SpatialFunctionPtr>::const_iterator funcIter = _spatialFunctionList.begin();
    for (; funcIter != _spatialFunctionList.end(); ++funcIter, ++paramIter) {
        *paramIter = (*(*funcIter))(x, y);
    }
}

Kernel::SpatialFunctionPtr Kernel::getSpatialFunction(unsigned int index) const {
    if (index >= _spatialFunctionList.size()) {
        if (!this->isSpatiallyVarying()) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterError, "kernel is not spatially varying");
        } else {
            std::ostringstream errStream;
            errStream << "index = " << index << "; must be < , " << _spatialFunctionList.size();
            throw LSST_EXCEPT(pexExcept::InvalidParameterError, errStream.str());
        }
    }
    return _spatialFunctionList[index]->clone();
}

std::vector<Kernel::SpatialFunctionPtr> Kernel::getSpatialFunctionList() const {
    std::vector<SpatialFunctionPtr> spFuncCopyList;
    for (std::vector<SpatialFunctionPtr>::const_iterator spFuncIter = _spatialFunctionList.begin();
         spFuncIter != _spatialFunctionList.end(); ++spFuncIter) {
        spFuncCopyList.push_back((**spFuncIter).clone());
    }
    return spFuncCopyList;
}

std::vector<double> Kernel::getKernelParameters() const { return std::vector<double>(); }

geom::Box2I Kernel::growBBox(geom::Box2I const &bbox) const {
    return geom::Box2I(geom::Point2I(bbox.getMin() - geom::Extent2I(getCtr())),
                       geom::Extent2I(bbox.getDimensions() + getDimensions() - geom::Extent2I(1, 1)));
}

geom::Box2I Kernel::shrinkBBox(geom::Box2I const &bbox) const {
    if ((bbox.getWidth() < getWidth()) || ((bbox.getHeight() < getHeight()))) {
        std::ostringstream os;
        os << "bbox dimensions = " << bbox.getDimensions() << " < (" << getWidth() << ", " << getHeight()
           << ") in one or both dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    return geom::Box2I(geom::Point2I(bbox.getMinX() + getCtrX(), bbox.getMinY() + getCtrY()),
                       geom::Extent2I(bbox.getWidth() + 1 - getWidth(), bbox.getHeight() + 1 - getHeight()));
}

std::string Kernel::toString(std::string const &prefix) const {
    std::ostringstream os;
    os << prefix << "Kernel:" << std::endl;
    os << prefix << "..height, width: " << _height << ", " << _width << std::endl;
    os << prefix << "..ctr (X, Y): " << _ctrX << ", " << _ctrY << std::endl;
    os << prefix << "..nKernelParams: " << _nKernelParams << std::endl;
    os << prefix << "..isSpatiallyVarying: " << (this->isSpatiallyVarying() ? "True" : "False") << std::endl;
    if (this->isSpatiallyVarying()) {
        os << prefix << "..spatialFunctions:" << std::endl;
        for (std::vector<SpatialFunctionPtr>::const_iterator spFuncPtr = _spatialFunctionList.begin();
             spFuncPtr != _spatialFunctionList.end(); ++spFuncPtr) {
            os << prefix << "...." << (*spFuncPtr)->toString() << std::endl;
        }
    }
    return os.str();
}

#if 0  //  This fails to compile with icc
void Kernel::toFile(std::string fileName) const {
    std::ofstream os(fileName.c_str());
    boost::archive::text_oarchive oa(os);
    oa << this;
}
#endif

//
// Protected Member Functions
//

void Kernel::setKernelParameter(unsigned int, double) const {
    throw LSST_EXCEPT(pexExcept::InvalidParameterError, "Kernel has no kernel parameters");
}

void Kernel::setKernelParametersFromSpatialModel(double x, double y) const {
    std::vector<SpatialFunctionPtr>::const_iterator funcIter = _spatialFunctionList.begin();
    for (int ii = 0; funcIter != _spatialFunctionList.end(); ++funcIter, ++ii) {
        this->setKernelParameter(ii, (*(*funcIter))(x, y));
    }
}

std::string Kernel::getPythonModule() const { return "lsst.afw.math"; }
}  // namespace math
}  // namespace afw
}  // namespace lsst