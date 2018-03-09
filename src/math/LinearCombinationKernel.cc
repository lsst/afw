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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelPersistenceHelper.h"
#include "lsst/afw/geom.h"

namespace pexExcept = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace math {

LinearCombinationKernel::LinearCombinationKernel()
        : Kernel(),
          _kernelList(),
          _kernelImagePtrList(),
          _kernelSumList(),
          _kernelParams(),
          _isDeltaFunctionBasis(false) {}

LinearCombinationKernel::LinearCombinationKernel(KernelList const &kernelList,
                                                 std::vector<double> const &kernelParameters)
        : Kernel(kernelList[0]->getWidth(), kernelList[0]->getHeight(), kernelList.size()),
          _kernelList(),
          _kernelImagePtrList(),
          _kernelSumList(),
          _kernelParams(kernelParameters),
          _isDeltaFunctionBasis(false) {
    if (kernelList.size() != kernelParameters.size()) {
        std::ostringstream os;
        os << "kernelList.size() = " << kernelList.size() << " != " << kernelParameters.size() << " = "
           << "kernelParameters.size()";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    checkKernelList(kernelList);
    _setKernelList(kernelList);
}

LinearCombinationKernel::LinearCombinationKernel(KernelList const &kernelList,
                                                 Kernel::SpatialFunction const &spatialFunction)
        : Kernel(kernelList[0]->getWidth(), kernelList[0]->getHeight(), kernelList.size(), spatialFunction),
          _kernelList(),
          _kernelImagePtrList(),
          _kernelSumList(),
          _kernelParams(std::vector<double>(kernelList.size())),
          _isDeltaFunctionBasis(false) {
    checkKernelList(kernelList);
    _setKernelList(kernelList);
}

LinearCombinationKernel::LinearCombinationKernel(
        KernelList const &kernelList, std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList)
        : Kernel(kernelList[0]->getWidth(), kernelList[0]->getHeight(), spatialFunctionList),
          _kernelList(),
          _kernelImagePtrList(),
          _kernelSumList(),
          _kernelParams(std::vector<double>(kernelList.size())),
          _isDeltaFunctionBasis(false) {
    if (kernelList.size() != spatialFunctionList.size()) {
        std::ostringstream os;
        os << "kernelList.size() = " << kernelList.size() << " != " << spatialFunctionList.size() << " = "
           << "spatialFunctionList.size()";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    checkKernelList(kernelList);
    _setKernelList(kernelList);
}

std::shared_ptr<Kernel> LinearCombinationKernel::clone() const {
    std::shared_ptr<Kernel> retPtr;
    if (this->isSpatiallyVarying()) {
        retPtr.reset(new LinearCombinationKernel(this->_kernelList, this->_spatialFunctionList));
    } else {
        retPtr.reset(new LinearCombinationKernel(this->_kernelList, this->_kernelParams));
    }
    retPtr->setCtr(this->getCtr());
    return retPtr;
}

std::shared_ptr<Kernel> LinearCombinationKernel::resized(int width, int height) const {
    KernelList kernelList;
    kernelList.reserve(getKernelList().size());
    for (const std::shared_ptr<Kernel> &kIter : getKernelList()) {
        kernelList.push_back(kIter->resized(width, height));
    }

    std::shared_ptr<Kernel> retPtr;
    if (isSpatiallyVarying()) {
        retPtr = std::make_shared<LinearCombinationKernel>(kernelList, _spatialFunctionList);
    } else {
        retPtr = std::make_shared<LinearCombinationKernel>(kernelList, _kernelParams);
    }

    return retPtr;
}

void LinearCombinationKernel::checkKernelList(const KernelList &kernelList) const {
    if (kernelList.empty()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, "kernelList has no elements");
    }

    geom::Extent2I const dim0 = kernelList[0]->getDimensions();
    geom::Point2I const ctr0 = kernelList[0]->getCtr();

    for (unsigned int ii = 0; ii < kernelList.size(); ++ii) {
        if (kernelList[ii]->getDimensions() != dim0) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                              (boost::format("kernel %d has different size than kernel 0") % ii).str());
        }
        if (kernelList[ii]->getCtr() != ctr0) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                              (boost::format("kernel %d has different center than kernel 0") % ii).str());
        }
        if (kernelList[ii]->isSpatiallyVarying()) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                              (boost::format("kernel %d is spatially varying") % ii).str());
        }
    }
}

KernelList const &LinearCombinationKernel::getKernelList() const { return _kernelList; }

std::vector<double> LinearCombinationKernel::getKernelSumList() const { return _kernelSumList; }

std::vector<double> LinearCombinationKernel::getKernelParameters() const { return _kernelParams; }

std::shared_ptr<Kernel> LinearCombinationKernel::refactor() const {
    if (!this->isSpatiallyVarying()) {
        return std::shared_ptr<Kernel>();
    }
    Kernel::SpatialFunctionPtr const firstSpFuncPtr = this->_spatialFunctionList[0];
    if (!firstSpFuncPtr->isLinearCombination()) {
        return std::shared_ptr<Kernel>();
    }

    typedef image::Image<Kernel::Pixel> KernelImage;
    typedef std::shared_ptr<KernelImage> KernelImagePtr;
    typedef std::vector<KernelImagePtr> KernelImageList;

    // create kernel images for new refactored basis kernels
    int const nSpatialParameters = this->getNSpatialParameters();
    KernelImageList newKernelImagePtrList;
    newKernelImagePtrList.reserve(nSpatialParameters);
    for (int i = 0; i < nSpatialParameters; ++i) {
        KernelImagePtr kernelImagePtr(new KernelImage(this->getDimensions()));
        newKernelImagePtrList.push_back(kernelImagePtr);
    }
    KernelImage kernelImage(this->getDimensions());
    std::vector<Kernel::SpatialFunctionPtr>::const_iterator spFuncPtrIter =
            this->_spatialFunctionList.begin();
    KernelList::const_iterator kIter = _kernelList.begin();
    KernelList::const_iterator const kEnd = _kernelList.end();
    auto & firstSpFunc = *firstSpFuncPtr;
    auto & firstType = typeid(firstSpFunc);     // noncopyable object of static storage duration
    for (; kIter != kEnd; ++kIter, ++spFuncPtrIter) {
        auto & spFunc = **spFuncPtrIter;
        if (typeid(spFunc) != firstType) {
            return std::shared_ptr<Kernel>();
        }

        (**kIter).computeImage(kernelImage, false);
        for (int i = 0; i < nSpatialParameters; ++i) {
            double spParam = (*spFuncPtrIter)->getParameter(i);
            newKernelImagePtrList[i]->scaledPlus(spParam, kernelImage);
        }
    }

    // create new kernel; the basis kernels are fixed kernels computed above
    // and the corresponding spatial model is the same function as the original kernel,
    // but with all coefficients zero except coeff_i = 1.0
    KernelList newKernelList;
    newKernelList.reserve(nSpatialParameters);
    KernelImageList::iterator newKImPtrIter = newKernelImagePtrList.begin();
    KernelImageList::iterator const newKImPtrEnd = newKernelImagePtrList.end();
    for (; newKImPtrIter != newKImPtrEnd; ++newKImPtrIter) {
        newKernelList.push_back(std::shared_ptr<Kernel>(new FixedKernel(**newKImPtrIter)));
    }
    std::vector<SpatialFunctionPtr> newSpFunctionPtrList;
    for (int i = 0; i < nSpatialParameters; ++i) {
        std::vector<double> newSpParameters(nSpatialParameters, 0.0);
        newSpParameters[i] = 1.0;
        SpatialFunctionPtr newSpFunctionPtr = firstSpFuncPtr->clone();
        newSpFunctionPtr->setParameters(newSpParameters);
        newSpFunctionPtrList.push_back(newSpFunctionPtr);
    }
    std::shared_ptr<LinearCombinationKernel> refactoredKernel(
            new LinearCombinationKernel(newKernelList, newSpFunctionPtrList));
    refactoredKernel->setCtr(this->getCtr());
    return refactoredKernel;
}

std::string LinearCombinationKernel::toString(std::string const &prefix) const {
    std::ostringstream os;
    os << prefix << "LinearCombinationKernel:" << std::endl;
    os << prefix << "..Kernels:" << std::endl;
    for (KernelList::const_iterator i = _kernelList.begin(); i != _kernelList.end(); ++i) {
        os << (*i)->toString(prefix + "\t");
    }
    os << "..parameters: [ ";
    for (std::vector<double>::const_iterator i = _kernelParams.begin(); i != _kernelParams.end(); ++i) {
        if (i != _kernelParams.begin()) os << ", ";
        os << *i;
    }
    os << " ]" << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
}

//
// Protected Member Functions
//
double LinearCombinationKernel::doComputeImage(image::Image<Pixel> &image, bool doNormalize) const {
    image = 0.0;
    double imSum = 0.0;
    std::vector<std::shared_ptr<image::Image<Pixel>>>::const_iterator kImPtrIter =
            _kernelImagePtrList.begin();
    std::vector<double>::const_iterator kSumIter = _kernelSumList.begin();
    std::vector<double>::const_iterator kParIter = _kernelParams.begin();
    for (; kImPtrIter != _kernelImagePtrList.end(); ++kImPtrIter, ++kSumIter, ++kParIter) {
        image.scaledPlus(*kParIter, **kImPtrIter);
        imSum += (*kSumIter) * (*kParIter);
    }

    if (doNormalize) {
        if (imSum == 0) {
            throw LSST_EXCEPT(pexExcept::OverflowError, "Cannot normalize; kernel sum is 0");
        }
        image /= imSum;
        imSum = 1;
    }

    return imSum;
}

void LinearCombinationKernel::setKernelParameter(unsigned int ind, double value) const {
    this->_kernelParams[ind] = value;
}

//
// Private Member Functions
//
void LinearCombinationKernel::_setKernelList(KernelList const &kernelList) {
    _kernelSumList.clear();
    _kernelImagePtrList.clear();
    _kernelList.clear();
    _isDeltaFunctionBasis = true;
    for (KernelList::const_iterator kIter = kernelList.begin(), kEnd = kernelList.end(); kIter != kEnd;
         ++kIter) {
        std::shared_ptr<Kernel> basisKernelPtr = (*kIter)->clone();
        if (dynamic_cast<DeltaFunctionKernel const *>(&(*basisKernelPtr)) == nullptr) {
            _isDeltaFunctionBasis = false;
        }
        _kernelList.push_back(basisKernelPtr);
        std::shared_ptr<image::Image<Pixel>> kernelImagePtr(new image::Image<Pixel>(this->getDimensions()));
        _kernelSumList.push_back(basisKernelPtr->computeImage(*kernelImagePtr, false));
        _kernelImagePtrList.push_back(kernelImagePtr);
    }
}

// ------ Persistence ---------------------------------------------------------------------------------------

namespace {

struct LinearCombinationKernelPersistenceHelper : public Kernel::PersistenceHelper {
    table::Key<table::Array<double>> amplitudes;
    table::Key<table::Array<int>> components;

    LinearCombinationKernelPersistenceHelper(int nComponents, bool isSpatiallyVarying)
            : Kernel::PersistenceHelper(isSpatiallyVarying ? nComponents : 0),
              components(schema.addField<table::Array<int>>("components", "archive IDs of component kernel",
                                                            nComponents)) {
        if (!isSpatiallyVarying) {
            amplitudes = schema.addField<table::Array<double>>("amplitudes", "amplitudes component kernel",
                                                               nComponents);
        }
    }

    explicit LinearCombinationKernelPersistenceHelper(table::Schema const &schema_)
            : Kernel::PersistenceHelper(schema_), components(schema["components"]) {
        if (!spatialFunctions.isValid()) {
            amplitudes = schema["amplitudes"];
            LSST_ARCHIVE_ASSERT(amplitudes.getSize() == components.getSize());
        } else {
            LSST_ARCHIVE_ASSERT(spatialFunctions.getSize() == components.getSize());
        }
    }
};

}  // namespace

class LinearCombinationKernel::Factory : public afw::table::io::PersistableFactory {
public:
    std::shared_ptr<afw::table::io::Persistable> read(InputArchive const &archive,
                                                              CatalogVector const &catalogs) const override {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        LinearCombinationKernelPersistenceHelper const keys(catalogs.front().getSchema());
        afw::table::BaseRecord const &record = catalogs.front().front();
        geom::Extent2I dimensions(record.get(keys.dimensions));
        std::vector<std::shared_ptr<Kernel>> componentList(keys.components.getSize());
        for (std::size_t i = 0; i < componentList.size(); ++i) {
            componentList[i] = archive.get<Kernel>(record[keys.components[i]]);
        }
        std::shared_ptr<LinearCombinationKernel> result;
        if (keys.spatialFunctions.isValid()) {
            std::vector<SpatialFunctionPtr> spatialFunctionList = keys.readSpatialFunctions(archive, record);
            result.reset(new LinearCombinationKernel(componentList, spatialFunctionList));
        } else {
            std::vector<double> kernelParameters(keys.amplitudes.getSize());
            for (std::size_t i = 0; i < kernelParameters.size(); ++i) {
                kernelParameters[i] = record[keys.amplitudes[i]];
            }
            result.reset(new LinearCombinationKernel(componentList, kernelParameters));
        }
        LSST_ARCHIVE_ASSERT(result->getDimensions() == dimensions);
        result->setCtr(record.get(keys.center));
        return result;
    }

    explicit Factory(std::string const &name) : afw::table::io::PersistableFactory(name) {}
};

namespace {

std::string getLinearCombinationKernelPersistenceName() { return "LinearCombinationKernel"; }

LinearCombinationKernel::Factory registration(getLinearCombinationKernelPersistenceName());

}  // anonymous

std::string LinearCombinationKernel::getPersistenceName() const {
    return getLinearCombinationKernelPersistenceName();
}

void LinearCombinationKernel::write(OutputArchiveHandle &handle) const {
    bool isVarying = isSpatiallyVarying();
    LinearCombinationKernelPersistenceHelper const keys(getNBasisKernels(), isVarying);
    std::shared_ptr<afw::table::BaseRecord> record = keys.write(handle, *this);
    if (isVarying) {
        for (int n = 0; n < keys.components.getSize(); ++n) {
            record->set(keys.components[n], handle.put(_kernelList[n]));
            record->set(keys.spatialFunctions[n], handle.put(_spatialFunctionList[n]));
        }
    } else {
        for (int n = 0; n < keys.components.getSize(); ++n) {
            record->set(keys.components[n], handle.put(_kernelList[n]));
            record->set(keys.amplitudes[n], _kernelParams[n]);
        }
    }
}
}  // namespace math
}  // namespace afw
}  // namespace lsst
