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
 * @brief Definitions of AnalyticKernel member functions.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <sstream>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelPersistenceHelper.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwMath = lsst::afw::math;
namespace afwImage = lsst::afw::image;

/**
 * @brief Construct an empty spatially invariant AnalyticKernel of size 0x0
 */
afwMath::AnalyticKernel::AnalyticKernel()
:
    Kernel(),
    _kernelFunctionPtr()
{}

/**
 * @brief Construct a spatially invariant AnalyticKernel,
 * or a spatially varying AnalyticKernel where the spatial model
 * is described by one function (that is cloned to give one per analytic function parameter).
 */
afwMath::AnalyticKernel::AnalyticKernel(
    int width,  ///< width of kernel
    int height, ///< height of kernel
    KernelFunction const &kernelFunction,   ///< kernel function; a deep copy is made
    Kernel::SpatialFunction const &spatialFunction  ///< spatial function;
        ///< one deep copy is made for each kernel function parameter;
        ///< if omitted or set to Kernel::NullSpatialFunction then the kernel is spatially invariant
) :
    Kernel(width, height, kernelFunction.getNParameters(), spatialFunction),
    _kernelFunctionPtr(kernelFunction.clone())
{}

/**
 * @brief Construct a spatially varying AnalyticKernel, where the spatial model
 * is described by a list of functions (one per analytic function parameter).
 *
 * @throw lsst::pex::exceptions::InvalidParameterException
 *        if the length of spatialFunctionList != # kernel function parameters.
 */
afwMath::AnalyticKernel::AnalyticKernel(
    int width,  ///< width of kernel
    int height, ///< height of kernel
    KernelFunction const &kernelFunction,   ///< kernel function; a deep copy is made
    std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList  ///< list of spatial functions,
        ///< one per kernel function parameter; a deep copy is made of each function
) :
    Kernel(width, height, spatialFunctionList),
    _kernelFunctionPtr(kernelFunction.clone())
{
    if (kernelFunction.getNParameters() != spatialFunctionList.size()) {
        std::ostringstream os;
        os << "kernelFunction.getNParameters() = " << kernelFunction.getNParameters()
            << " != " << spatialFunctionList.size() << " = " << "spatialFunctionList.size()";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
}

PTR(afwMath::Kernel) afwMath::AnalyticKernel::clone() const {
    PTR(afwMath::Kernel) retPtr;
    if (this->isSpatiallyVarying()) {
        retPtr.reset(new afwMath::AnalyticKernel(this->getWidth(), this->getHeight(),
            *(this->_kernelFunctionPtr), this->_spatialFunctionList));
    } else {
        retPtr.reset(new afwMath::AnalyticKernel(this->getWidth(), this->getHeight(),
            *(this->_kernelFunctionPtr)));
    }
    retPtr->setCtrX(this->getCtrX());
    retPtr->setCtrY(this->getCtrY());
    return retPtr;
}

double afwMath::AnalyticKernel::computeImage(
    afwImage::Image<Pixel> &image,
    bool doNormalize,
    double xPos,
    double yPos
) const {
#if 0
    if (image.getDimensions() != this->getDimensions()) {
        std::ostringstream os;
        os << "image dimensions = ( " << image.getWidth() << ", " << image.getHeight()
            << ") != (" << this->getWidth() << ", " << this->getHeight() << ") = kernel dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
#endif
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(xPos, yPos);
    }

    double xOffset = -this->getCtrX();
    double yOffset = -this->getCtrY();

    double imSum = 0;
    for (int y = 0; y != image.getHeight(); ++y) {
        double const fy = y + yOffset;
        afwImage::Image<Pixel>::x_iterator ptr = image.row_begin(y);
        for (int x = 0; x != image.getWidth(); ++x, ++ptr) {
            double const fx = x + xOffset;
            Pixel const pixelVal = (*_kernelFunctionPtr)(fx, fy);
            *ptr = pixelVal;
            imSum += pixelVal;
        }
    }
    if (doNormalize) {
        if (imSum == 0) {
            throw LSST_EXCEPT(pexExcept::OverflowErrorException, "Cannot normalize; kernel sum is 0");
        }
        image /= imSum;
        imSum = 1;
    }

    return imSum;
}

/**
 * @brief Get a deep copy of the kernel function
 */
afwMath::AnalyticKernel::KernelFunctionPtr afwMath::AnalyticKernel::getKernelFunction(
) const {
    return _kernelFunctionPtr->clone();
}

std::string afwMath::AnalyticKernel::toString(std::string const& prefix) const {
    std::ostringstream os;
    os << prefix << "AnalyticKernel:" << std::endl;
    os << prefix << "..function: " << (_kernelFunctionPtr ? _kernelFunctionPtr->toString() : "None")
        << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
}

std::vector<double> afwMath::AnalyticKernel::getKernelParameters() const {
    return _kernelFunctionPtr->getParameters();
}

//
// Protected Member Functions
//
void afwMath::AnalyticKernel::setKernelParameter(unsigned int ind, double value) const {
    _kernelFunctionPtr->setParameter(ind, value);
}

// ------ Persistence ---------------------------------------------------------------------------------------

namespace lsst { namespace afw { namespace math {

namespace {

struct AnalyticKernelPersistenceHelper : public Kernel::PersistenceHelper {
    table::Key<int> kernelFunction;

    explicit AnalyticKernelPersistenceHelper(int nSpatialFunctions) :
        Kernel::PersistenceHelper(nSpatialFunctions),
        kernelFunction(
            schema.addField<int>(
                "kernelfunction", "archive ID for analytic function used to produce kernel images"
            )
        )
    {}

    explicit AnalyticKernelPersistenceHelper(table::Schema const & schema_) :
        Kernel::PersistenceHelper(schema_),
        kernelFunction(schema["kernelfunction"])
    {}
};

} // anonymous

class AnalyticKernel::Factory : public afw::table::io::PersistableFactory {
public:

    virtual PTR(afw::table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        AnalyticKernelPersistenceHelper const keys(catalogs.front().getSchema());
        afw::table::BaseRecord const & record = catalogs.front().front();
        PTR(AnalyticKernel::KernelFunction) kernelFunction =
            archive.get<AnalyticKernel::KernelFunction>(record.get(keys.kernelFunction));
        PTR(AnalyticKernel) result;
        if (keys.spatialFunctions.isValid()) {
            result = boost::make_shared<AnalyticKernel>(
                record.get(keys.dimensions.getX()), record.get(keys.dimensions.getY()), *kernelFunction,
                keys.readSpatialFunctions(archive, record)
            );
        } else {
            result = boost::make_shared<AnalyticKernel>(
                record.get(keys.dimensions.getX()), record.get(keys.dimensions.getY()), *kernelFunction
            );
        }
        result->setCtr(record.get(keys.center));
        return result;
    }

    explicit Factory(std::string const & name) : afw::table::io::PersistableFactory(name) {}
};

namespace {

std::string getAnalyticKernelPersistenceName() { return "AnalyticKernel"; }

AnalyticKernel::Factory registration(getAnalyticKernelPersistenceName());

} // anonymous

std::string AnalyticKernel::getPersistenceName() const { return getAnalyticKernelPersistenceName(); }

void AnalyticKernel::write(OutputArchiveHandle & handle) const {
    AnalyticKernelPersistenceHelper const keys(_spatialFunctionList.size());
    PTR(afw::table::BaseRecord) record = keys.write(handle, *this);
    record->set(keys.kernelFunction, handle.put(_kernelFunctionPtr.get()));
}

}}} // namespace lsst::afw::math
