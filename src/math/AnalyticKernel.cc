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
#include <sstream>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelPersistenceHelper.h"

namespace pexExcept = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace math {

AnalyticKernel::AnalyticKernel() : Kernel(), _kernelFunctionPtr() {}

AnalyticKernel::AnalyticKernel(int width, int height, KernelFunction const &kernelFunction,
                               Kernel::SpatialFunction const &spatialFunction)
        : Kernel(width, height, kernelFunction.getNParameters(), spatialFunction),
          _kernelFunctionPtr(kernelFunction.clone()) {}

AnalyticKernel::AnalyticKernel(int width, int height, KernelFunction const &kernelFunction,
                               std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList)
        : Kernel(width, height, spatialFunctionList), _kernelFunctionPtr(kernelFunction.clone()) {
    if (kernelFunction.getNParameters() != spatialFunctionList.size()) {
        std::ostringstream os;
        os << "kernelFunction.getNParameters() = " << kernelFunction.getNParameters()
           << " != " << spatialFunctionList.size() << " = "
           << "spatialFunctionList.size()";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
}

std::shared_ptr<Kernel> AnalyticKernel::clone() const {
    std::shared_ptr<Kernel> retPtr;
    if (this->isSpatiallyVarying()) {
        retPtr.reset(new AnalyticKernel(this->getWidth(), this->getHeight(), *(this->_kernelFunctionPtr),
                                        this->_spatialFunctionList));
    } else {
        retPtr.reset(new AnalyticKernel(this->getWidth(), this->getHeight(), *(this->_kernelFunctionPtr)));
    }
    retPtr->setCtr(this->getCtr());
    return retPtr;
}

std::shared_ptr<Kernel> AnalyticKernel::resized(int width, int height) const {
    std::shared_ptr<Kernel> retPtr;
    if (isSpatiallyVarying()) {
        retPtr = std::make_shared<AnalyticKernel>(width, height, *_kernelFunctionPtr, _spatialFunctionList);
    } else {
        retPtr = std::make_shared<AnalyticKernel>(width, height, *_kernelFunctionPtr);
    }
    return retPtr;
}

double AnalyticKernel::computeImage(image::Image<Pixel> &image, bool doNormalize, double x, double y) const {
    lsst::geom::Extent2I llBorder = (image.getDimensions() - getDimensions()) / 2;
    image.setXY0(lsst::geom::Point2I(-lsst::geom::Extent2I(getCtr() + llBorder)));
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(x, y);
    }
    return doComputeImage(image, doNormalize);
}

AnalyticKernel::KernelFunctionPtr AnalyticKernel::getKernelFunction() const {
    return _kernelFunctionPtr->clone();
}

std::string AnalyticKernel::toString(std::string const &prefix) const {
    std::ostringstream os;
    os << prefix << "AnalyticKernel:" << std::endl;
    os << prefix << "..function: " << (_kernelFunctionPtr ? _kernelFunctionPtr->toString() : "None")
       << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
}

std::vector<double> AnalyticKernel::getKernelParameters() const {
    return _kernelFunctionPtr->getParameters();
}

//
// Protected Member Functions
//
double AnalyticKernel::doComputeImage(image::Image<Pixel> &image, bool doNormalize) const {
    double imSum = 0;
    for (int y = 0; y != image.getHeight(); ++y) {
        double const fy = image.indexToPosition(y, image::Y);
        image::Image<Pixel>::x_iterator ptr = image.row_begin(y);
        for (int x = 0; x != image.getWidth(); ++x, ++ptr) {
            double const fx = image.indexToPosition(x, image::X);
            Pixel const pixelVal = (*_kernelFunctionPtr)(fx, fy);
            *ptr = pixelVal;
            imSum += pixelVal;
        }
    }

    if (doNormalize && (imSum != 1)) {
        if (imSum == 0) {
            throw LSST_EXCEPT(pexExcept::OverflowError, "Cannot normalize; kernel sum is 0");
        }
        image /= imSum;
        imSum = 1;
    }

    return imSum;
}

void AnalyticKernel::setKernelParameter(unsigned int ind, double value) const {
    _kernelFunctionPtr->setParameter(ind, value);
}

// ------ Persistence ---------------------------------------------------------------------------------------

namespace {

struct AnalyticKernelPersistenceHelper : public Kernel::PersistenceHelper {
    table::Key<int> kernelFunction;

    explicit AnalyticKernelPersistenceHelper(int nSpatialFunctions)
            : Kernel::PersistenceHelper(nSpatialFunctions),
              kernelFunction(schema.addField<int>(
                      "kernelfunction", "archive ID for analytic function used to produce kernel images")) {}

    explicit AnalyticKernelPersistenceHelper(table::Schema const &schema_)
            : Kernel::PersistenceHelper(schema_), kernelFunction(schema["kernelfunction"]) {}
};

}  // namespace

class AnalyticKernel::Factory : public afw::table::io::PersistableFactory {
public:
    virtual std::shared_ptr<afw::table::io::Persistable> read(InputArchive const &archive,
                                                              CatalogVector const &catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        AnalyticKernelPersistenceHelper const keys(catalogs.front().getSchema());
        afw::table::BaseRecord const &record = catalogs.front().front();
        std::shared_ptr<AnalyticKernel::KernelFunction> kernelFunction =
                archive.get<AnalyticKernel::KernelFunction>(record.get(keys.kernelFunction));
        std::shared_ptr<AnalyticKernel> result;
        if (keys.spatialFunctions.isValid()) {
            result = std::make_shared<AnalyticKernel>(record.get(keys.dimensions.getX()),
                                                      record.get(keys.dimensions.getY()), *kernelFunction,
                                                      keys.readSpatialFunctions(archive, record));
        } else {
            result = std::make_shared<AnalyticKernel>(record.get(keys.dimensions.getX()),
                                                      record.get(keys.dimensions.getY()), *kernelFunction);
        }
        result->setCtr(record.get(keys.center));
        return result;
    }

    explicit Factory(std::string const &name) : afw::table::io::PersistableFactory(name) {}
};

namespace {

std::string getAnalyticKernelPersistenceName() { return "AnalyticKernel"; }

AnalyticKernel::Factory registration(getAnalyticKernelPersistenceName());

}  // namespace

std::string AnalyticKernel::getPersistenceName() const { return getAnalyticKernelPersistenceName(); }

void AnalyticKernel::write(OutputArchiveHandle &handle) const {
    AnalyticKernelPersistenceHelper const keys(_spatialFunctionList.size());
    std::shared_ptr<afw::table::BaseRecord> record = keys.write(handle, *this);
    record->set(keys.kernelFunction, handle.put(_kernelFunctionPtr.get()));
}
}  // namespace math
}  // namespace afw
}  // namespace lsst
