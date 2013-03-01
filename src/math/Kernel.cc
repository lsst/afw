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
 
/**
 * @file
 *
 * @brief Definitions of Kernel member functions.
 *
 * @ingroup afw
 */
#include <fstream>
#include <sstream>

#include "boost/format.hpp"
#if defined(__ICC)
#pragma warning (push)
#pragma warning (disable: 444)
#endif
#include "boost/archive/text_oarchive.hpp"
#if defined(__ICC)
#pragma warning (pop)
#endif

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwGeom = lsst::afw::geom;
namespace afwMath = lsst::afw::math;

afwMath::generic_kernel_tag afwMath::generic_kernel_tag_; ///< Used as default value in argument lists
afwMath::deltafunction_kernel_tag afwMath::deltafunction_kernel_tag_;
    ///< Used as default value in argument lists

//
// Constructors
//
/**
 * @brief Construct a null Kernel of size 0,0.
 *
 * A null constructor is primarily intended for persistence.
 */
namespace {
}
afwMath::Kernel::Kernel()
:
    daf::base::Citizen(typeid(this)),
    _spatialFunctionList(),
    _width(0),
    _height(0),
    _ctrX(0),
    _ctrY(0),
    _nKernelParams(0)
{}

/**
 * @brief Construct a spatially invariant Kernel or a spatially varying Kernel with one spatial function
 * that is duplicated as needed.
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if a spatial function is specified
 * and the kernel has no parameters.
 * @throw lsst::pex::exceptions::InvalidParameterException if a width or height < 1
 */
afwMath::Kernel::Kernel(
    int width,                          ///< number of columns
    int height,                         ///< number of height
    unsigned int nKernelParams,         ///< number of kernel parameters
    SpatialFunction const &spatialFunction) ///< spatial function (or NullSpatialFunction if none specified)
:
    daf::base::Citizen(typeid(this)),
    _spatialFunctionList(),
    _width(width),
    _height(height),
    _ctrX((width-1)/2),
    _ctrY((height-1)/2),
    _nKernelParams(nKernelParams)
{
    if ((width < 1) || (height < 1)) {
        std::ostringstream os;
        os << "kernel height = " << height << " and/or width = " << width << " < 1";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    if (dynamic_cast<const NullSpatialFunction*>(&spatialFunction)) {
        // spatialFunction is not really present
    } else {
        if (nKernelParams == 0) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, "Kernel function has no parameters");
        }
        for (unsigned int ii = 0; ii < nKernelParams; ++ii) {
            SpatialFunctionPtr spatialFunctionCopy = spatialFunction.clone();
            this->_spatialFunctionList.push_back(spatialFunctionCopy);
        }
    }
}

/**
 * @brief Construct a spatially varying Kernel with a list of spatial functions (one per kernel parameter)
 *
 * Note: if the list of spatial functions is empty then the kernel is not spatially varying.
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if a width or height < 1
 */
afwMath::Kernel::Kernel(
    int width,                          ///< number of columns
    int height,                         ///< number of height
    std::vector<SpatialFunctionPtr> spatialFunctionList)
        ///< list of spatial function, one per kernel parameter
:
    daf::base::Citizen(typeid(this)),
   _width(width),
   _height(height),
   _ctrX(width/2),
   _ctrY(height/2),
   _nKernelParams(spatialFunctionList.size())
{
    if ((width < 1) || (height < 1)) {
        std::ostringstream os;
        os << "kernel height = " << height << " and/or width = " << width << " < 1";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    for (unsigned int ii = 0; ii < spatialFunctionList.size(); ++ii) {
        SpatialFunctionPtr spatialFunctionCopy = spatialFunctionList[ii]->clone();
        this->_spatialFunctionList.push_back(spatialFunctionCopy);
    }
}

//
// Public Member Functions
//
/**
 * @brief Set the parameters of all spatial functions
 *
 * Params is indexed as [kernel parameter][spatial parameter]
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if params is the wrong shape
 *  (if this exception is thrown then no parameters are changed)
 */
void afwMath::Kernel::setSpatialParameters(const std::vector<std::vector<double> > params) {
    // Check params size before changing anything
    unsigned int nKernelParams = this->getNKernelParameters();
    if (params.size() != nKernelParams) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
            (boost::format("params has %d entries instead of %d") % params.size() % nKernelParams).str());
    }
    unsigned int nSpatialParams = this->getNSpatialParameters();
    for (unsigned int ii = 0; ii < nKernelParams; ++ii) {
        if (params[ii].size() != nSpatialParams) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                (boost::format("params[%d] has %d entries instead of %d") %
                ii % params[ii].size() % nSpatialParams).str());
        }
    }
    // Set parameters
    if (nSpatialParams > 0) {
        for (unsigned int ii = 0; ii < nKernelParams; ++ii) {
            this->_spatialFunctionList[ii]->setParameters(params[ii]);
        }
    }
}

/**
 * @brief Compute the kernel parameters at a specified point
 *
 * Warning: this is a low-level function that assumes kernelParams is the right length.
 * It will fail in unpredictable ways if that condition is not met.
 */
void afwMath::Kernel::computeKernelParametersFromSpatialModel(
    std::vector<double> &kernelParams, double x, double y) const {
    std::vector<double>::iterator paramIter = kernelParams.begin();
    std::vector<SpatialFunctionPtr>::const_iterator funcIter = _spatialFunctionList.begin();
    for ( ; funcIter != _spatialFunctionList.end(); ++funcIter, ++paramIter) {
        *paramIter = (*(*funcIter))(x,y);
    }
}

/**
 * @brief Return a clone of the specified spatial function (one component of the spatial model)
 *
 * @return a shared pointer to a spatial function. The function is a deep copy, so setting its parameters
 * has no effect on the kernel.
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if kernel not spatially varying
 * @throw lsst::pex::exceptions::InvalidParameterException if index out of range
 */
afwMath::Kernel::SpatialFunctionPtr afwMath::Kernel::getSpatialFunction(
    unsigned int index  ///< index of desired spatial function;
                        ///< must be in range [0, number spatial parameters - 1]
) const {
    if (index >= _spatialFunctionList.size()) {
        if (!this->isSpatiallyVarying()) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, "kernel is not spatially varying");
        } else {
            std::ostringstream errStream;
            errStream << "index = " << index << "; must be < , " << _spatialFunctionList.size();
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, errStream.str());
        }
    }
    return _spatialFunctionList[index]->clone();
}

/**
 * @brief Return a list of clones of the spatial functions.
 *
 * @return a list of shared pointers to spatial functions. The functions are deep copies,
 * so setting their parameters has no effect on the kernel.
 */
std::vector<afwMath::Kernel::SpatialFunctionPtr> afwMath::Kernel::getSpatialFunctionList(
) const {
    std::vector<SpatialFunctionPtr> spFuncCopyList;
    for (std::vector<SpatialFunctionPtr>::const_iterator spFuncIter = _spatialFunctionList.begin();
        spFuncIter != _spatialFunctionList.end(); ++spFuncIter) {
        spFuncCopyList.push_back((**spFuncIter).clone());
    }
    return spFuncCopyList;
}

/**
 * @brief Return the current kernel parameters
 *
 * If the kernel is spatially varying then the parameters are those last computed.
 * See also computeKernelParametersFromSpatialModel.
 * If there are no kernel parameters then returns an empty vector.
 */
std::vector<double> afwMath::Kernel::getKernelParameters() const {
    return std::vector<double>();
}


/**
 * Given a bounding box for pixels one wishes to compute by convolving an image with this kernel,
 * return the bounding box of pixels that must be accessed on the image to be convolved.
 * Thus the box shifted by -kernel.getCtr() and its size is expanded by kernel.getDimensions()-1.
 *
 * @return the bbox expanded by the kernel. 
 */
afwGeom::Box2I afwMath::Kernel::growBBox(afwGeom::Box2I const &bbox) const {
    return afwGeom::Box2I(
        afwGeom::Point2I(
            bbox.getMinX() - getCtrX(),
            bbox.getMinY() - getCtrY()),
        afwGeom::Extent2I(
            bbox.getWidth()  + getWidth() - 1,
            bbox.getHeight() + getHeight() - 1));
}

/**
 * Given a bounding box for an image one wishes to convolve with this kernel,
 * return the bounding box for the region of pixels that can be computed.
 * Thus the box shifted by kernel.getCtr() and its size is reduced by kernel.getDimensions()-1.
 *
 * @return the bbox shrunk by the kernel.
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if the resulting box would have
 * dimension < 1 in either axis
 */
afwGeom::Box2I afwMath::Kernel::shrinkBBox(afwGeom::Box2I const &bbox) const {
    if ((bbox.getWidth() < getWidth()) || ((bbox.getHeight() < getHeight()))) {
        std::ostringstream os;
        os << "bbox dimensions = " << bbox.getDimensions() << " < ("
           << getWidth() << ", " << getHeight() << ") in one or both dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    return afwGeom::Box2I(
        afwGeom::Point2I(
            bbox.getMinX() + getCtrX(),
            bbox.getMinY() + getCtrY()),
        afwGeom::Extent2I(
            bbox.getWidth()  + 1 - getWidth(),
            bbox.getHeight() + 1 - getHeight()));
}


/**
 * @brief Return a string representation of the kernel
 */
std::string afwMath::Kernel::toString(std::string const& prefix) const {
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

#if 0                                   //  This fails to compile with icc
void afwMath::Kernel::toFile(std::string fileName) const {
    std::ofstream os(fileName.c_str());
    boost::archive::text_oarchive oa(os);
    oa << this;
}
#endif

//
// Protected Member Functions
//

/**
 * @brief Set one kernel parameter
 *
 * Classes that have kernel parameters must subclass this function.
 *
 * This function is marked "const", despite modifying unimportant internals,
 * so that computeImage can be const.
 *
 * @throw lsst::pex::exceptions::InvalidParameterException always (unless subclassed)
 */
void afwMath::Kernel::setKernelParameter(unsigned int, double) const {
    throw LSST_EXCEPT(pexExcept::InvalidParameterException, "Kernel has no kernel parameters");
}

/**
 * @brief Set the kernel parameters from the spatial model (if any).
 *
 * This function has no effect if there is no spatial model.
 *
 * This function is marked "const", despite modifying unimportant internals,
 * so that computeImage can be const.
 */
void afwMath::Kernel::setKernelParametersFromSpatialModel(double x, double y) const {
    std::vector<SpatialFunctionPtr>::const_iterator funcIter = _spatialFunctionList.begin();
    for (int ii = 0; funcIter != _spatialFunctionList.end(); ++funcIter, ++ii) {
        this->setKernelParameter(ii, (*(*funcIter))(x,y));
    }
}

std::string afwMath::Kernel::getPythonModule() const { return "lsst.afw.math"; }
