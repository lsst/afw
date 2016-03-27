// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
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
namespace afwMath = lsst::afw::math;
namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;

afwMath::LinearCombinationKernel::LinearCombinationKernel()
:
    Kernel(),
    _kernelList(),
    _kernelImagePtrList(),
    _kernelSumList(),
    _kernelParams(),
    _isDeltaFunctionBasis(false)
{ }

afwMath::LinearCombinationKernel::LinearCombinationKernel(
    KernelList const &kernelList,
    std::vector<double> const &kernelParameters
) :
    Kernel(kernelList[0]->getWidth(), kernelList[0]->getHeight(), kernelList.size()),
    _kernelList(),
    _kernelImagePtrList(),
    _kernelSumList(),
    _kernelParams(kernelParameters),
    _isDeltaFunctionBasis(false)
{
    if (kernelList.size() != kernelParameters.size()) {
        std::ostringstream os;
        os << "kernelList.size() = " << kernelList.size()
            << " != " << kernelParameters.size() << " = " << "kernelParameters.size()";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    checkKernelList(kernelList);
    _setKernelList(kernelList);
}

afwMath::LinearCombinationKernel::LinearCombinationKernel(
    KernelList const &kernelList,
    Kernel::SpatialFunction const &spatialFunction
) :
    Kernel(kernelList[0]->getWidth(), kernelList[0]->getHeight(), kernelList.size(), spatialFunction),
    _kernelList(),
    _kernelImagePtrList(),
    _kernelSumList(),
    _kernelParams(std::vector<double>(kernelList.size())),
    _isDeltaFunctionBasis(false)
{
    checkKernelList(kernelList);
    _setKernelList(kernelList);
}

afwMath::LinearCombinationKernel::LinearCombinationKernel(
    KernelList const &kernelList,
    std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList
) :
    Kernel(kernelList[0]->getWidth(), kernelList[0]->getHeight(), spatialFunctionList),
    _kernelList(),
    _kernelImagePtrList(),
    _kernelSumList(),
    _kernelParams(std::vector<double>(kernelList.size())),
    _isDeltaFunctionBasis(false)
{
    if (kernelList.size() != spatialFunctionList.size()) {
        std::ostringstream os;
        os << "kernelList.size() = " << kernelList.size()
            << " != " << spatialFunctionList.size() << " = " << "spatialFunctionList.size()";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    checkKernelList(kernelList);
    _setKernelList(kernelList);
}

PTR(afwMath::Kernel) afwMath::LinearCombinationKernel::clone() const {
    PTR(Kernel) retPtr;
    if (this->isSpatiallyVarying()) {
        retPtr.reset(new afwMath::LinearCombinationKernel(this->_kernelList, this->_spatialFunctionList));
    } else {
        retPtr.reset(new afwMath::LinearCombinationKernel(this->_kernelList, this->_kernelParams));
    }
    retPtr->setCtr(this->getCtr());
    return retPtr;
}

void afwMath::LinearCombinationKernel::checkKernelList(const KernelList &kernelList) const {
    if (kernelList.size() < 1) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, "kernelList has no elements");
    }

    afwGeom::Extent2I const dim0 = kernelList[0]->getDimensions();
    afwGeom::Point2I const ctr0 = kernelList[0]->getCtr();

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

afwMath::KernelList const & afwMath::LinearCombinationKernel::getKernelList() const {
    return _kernelList;
}

std::vector<double> afwMath::LinearCombinationKernel::getKernelSumList() const {
    return _kernelSumList;
}

std::vector<double> afwMath::LinearCombinationKernel::getKernelParameters() const {
    return _kernelParams;
}

PTR(afwMath::Kernel) afwMath::LinearCombinationKernel::refactor() const {
    if (!this->isSpatiallyVarying()) {
        return PTR(Kernel)();
    }
    Kernel::SpatialFunctionPtr const firstSpFuncPtr = this->_spatialFunctionList[0];
    if (!firstSpFuncPtr->isLinearCombination()) {
        return PTR(Kernel)();
    }
    
    typedef lsst::afw::image::Image<Kernel::Pixel> KernelImage;
    typedef boost::shared_ptr<KernelImage> KernelImagePtr;
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
    afwMath::KernelList::const_iterator kIter = _kernelList.begin();
    afwMath::KernelList::const_iterator const kEnd = _kernelList.end();
    for ( ; kIter != kEnd; ++kIter, ++spFuncPtrIter) {
        if (typeid(**spFuncPtrIter) != typeid(*firstSpFuncPtr)) {
            return PTR(Kernel)();
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
    afwMath::KernelList newKernelList;
    newKernelList.reserve(nSpatialParameters);
    KernelImageList::iterator newKImPtrIter = newKernelImagePtrList.begin();
    KernelImageList::iterator const newKImPtrEnd = newKernelImagePtrList.end();
    for ( ; newKImPtrIter != newKImPtrEnd; ++newKImPtrIter) {
        newKernelList.push_back(PTR(Kernel)(new afwMath::FixedKernel(**newKImPtrIter)));
    }
    std::vector<SpatialFunctionPtr> newSpFunctionPtrList;
    for (int i = 0; i < nSpatialParameters; ++i) {
        std::vector<double> newSpParameters(nSpatialParameters, 0.0);
        newSpParameters[i] = 1.0;
        SpatialFunctionPtr newSpFunctionPtr = firstSpFuncPtr->clone();
        newSpFunctionPtr->setParameters(newSpParameters);
        newSpFunctionPtrList.push_back(newSpFunctionPtr);
    }
    PTR(LinearCombinationKernel) refactoredKernel(
        new LinearCombinationKernel(newKernelList, newSpFunctionPtrList));
    refactoredKernel->setCtr(this->getCtr());
    return refactoredKernel;
}

std::string afwMath::LinearCombinationKernel::toString(std::string const& prefix) const {
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
double afwMath::LinearCombinationKernel::doComputeImage(
    afwImage::Image<Pixel> &image,
    bool doNormalize
) const {
    image = 0.0;
    double imSum = 0.0;
    std::vector<PTR(afwImage::Image<Pixel>)>::const_iterator kImPtrIter = _kernelImagePtrList.begin();
    std::vector<double>::const_iterator kSumIter = _kernelSumList.begin();
    std::vector<double>::const_iterator kParIter = _kernelParams.begin();
    for ( ; kImPtrIter != _kernelImagePtrList.end(); ++kImPtrIter, ++kSumIter, ++kParIter) {
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

void afwMath::LinearCombinationKernel::setKernelParameter(unsigned int ind, double value) const {
    this->_kernelParams[ind] = value;
}

//
// Private Member Functions
//
void afwMath::LinearCombinationKernel::_setKernelList(KernelList const &kernelList) {
    _kernelSumList.clear();
    _kernelImagePtrList.clear();
    _kernelList.clear();
    _isDeltaFunctionBasis = true;
    for (KernelList::const_iterator kIter = kernelList.begin(), kEnd = kernelList.end();
        kIter != kEnd; ++kIter) {
        PTR(Kernel) basisKernelPtr = (*kIter)->clone();
        if (dynamic_cast<afwMath::DeltaFunctionKernel const *>(&(*basisKernelPtr)) == 0) {
            _isDeltaFunctionBasis = false;
        }
        _kernelList.push_back(basisKernelPtr);
        PTR(afwImage::Image<Pixel>) kernelImagePtr(new afwImage::Image<Pixel>(this->getDimensions()));
        _kernelSumList.push_back(basisKernelPtr->computeImage(*kernelImagePtr, false));
        _kernelImagePtrList.push_back(kernelImagePtr);
    }
}

// ------ Persistence ---------------------------------------------------------------------------------------

namespace lsst { namespace afw { namespace math {

namespace {

struct LinearCombinationKernelPersistenceHelper : public Kernel::PersistenceHelper {
    table::Key< table::Array<double> > amplitudes;
    table::Key< table::Array<int> > components;

    LinearCombinationKernelPersistenceHelper(int nComponents, bool isSpatiallyVarying) :
        Kernel::PersistenceHelper(isSpatiallyVarying ? nComponents : 0),
        components(
            schema.addField< table::Array<int> >("components", "archive IDs of component kernel",
                                                 nComponents)
        )
    {
        if (!isSpatiallyVarying) {
            amplitudes = schema.addField< table::Array<double> >("amplitudes", "amplitudes component kernel",
                                                                 nComponents);
        }
    }

    LinearCombinationKernelPersistenceHelper(table::Schema const & schema_) :
        Kernel::PersistenceHelper(schema_), components(schema["components"])
    {
        if (!spatialFunctions.isValid()) {
            amplitudes = schema["amplitudes"];
            LSST_ARCHIVE_ASSERT(amplitudes.getSize() == components.getSize());
        } else {
            LSST_ARCHIVE_ASSERT(spatialFunctions.getSize() == components.getSize());
        }
    }

};

} // anonymous

class LinearCombinationKernel::Factory : public afw::table::io::PersistableFactory {
public:

    virtual PTR(afw::table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        LinearCombinationKernelPersistenceHelper const keys(catalogs.front().getSchema());
        afw::table::BaseRecord const & record = catalogs.front().front();
        geom::Extent2I dimensions(record.get(keys.dimensions));
        std::vector<PTR(Kernel)> componentList(keys.components.getSize());        
        for (std::size_t i = 0; i < componentList.size(); ++i) {
            componentList[i] = archive.get<Kernel>(record[keys.components[i]]);
        }
        PTR(LinearCombinationKernel) result;
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

    explicit Factory(std::string const & name) : afw::table::io::PersistableFactory(name) {}
};

namespace {

std::string getLinearCombinationKernelPersistenceName() { return "LinearCombinationKernel"; }

LinearCombinationKernel::Factory registration(getLinearCombinationKernelPersistenceName());

} // anonymous

std::string LinearCombinationKernel::getPersistenceName() const {
    return getLinearCombinationKernelPersistenceName();
}

void LinearCombinationKernel::write(OutputArchiveHandle & handle) const {
    bool isVarying = isSpatiallyVarying();
    LinearCombinationKernelPersistenceHelper const keys(getNBasisKernels(), isVarying);
    PTR(afw::table::BaseRecord) record = keys.write(handle, *this);
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

}}} // namespace lsst::afw::math
