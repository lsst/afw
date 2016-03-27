// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
#include <sstream>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelPersistenceHelper.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwGeom = lsst::afw::geom;
namespace afwMath = lsst::afw::math;
namespace afwImage = lsst::afw::image;

afwMath::AnalyticKernel::AnalyticKernel()
:
    Kernel(),
    _kernelFunctionPtr()
{}

afwMath::AnalyticKernel::AnalyticKernel(
    int width,
    int height,
    KernelFunction const &kernelFunction,
    Kernel::SpatialFunction const &spatialFunction
) :
    Kernel(width, height, kernelFunction.getNParameters(), spatialFunction),
    _kernelFunctionPtr(kernelFunction.clone())
{}

afwMath::AnalyticKernel::AnalyticKernel(
    int width,
    int height,
    KernelFunction const &kernelFunction,
    std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList
) :
    Kernel(width, height, spatialFunctionList),
    _kernelFunctionPtr(kernelFunction.clone())
{
    if (kernelFunction.getNParameters() != spatialFunctionList.size()) {
        std::ostringstream os;
        os << "kernelFunction.getNParameters() = " << kernelFunction.getNParameters()
            << " != " << spatialFunctionList.size() << " = " << "spatialFunctionList.size()";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
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
    retPtr->setCtr(this->getCtr());
    return retPtr;
}

double afwMath::AnalyticKernel::computeImage(
    lsst::afw::image::Image<Pixel> &image,
    bool doNormalize,
    double x,
    double y
) const {
    afwGeom::Extent2I llBorder = (image.getDimensions() - getDimensions()) / 2;
    image.setXY0(afwGeom::Point2I(-afwGeom::Extent2I(getCtr()+ llBorder)));
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(x, y);
    }
    return doComputeImage(image, doNormalize);
}

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
double afwMath::AnalyticKernel::doComputeImage(
    afwImage::Image<Pixel> &image,
    bool doNormalize
) const {
    double imSum = 0;
    for (int y = 0; y != image.getHeight(); ++y) {
        double const fy = image.indexToPosition(y, afwImage::Y);
        afwImage::Image<Pixel>::x_iterator ptr = image.row_begin(y);
        for (int x = 0; x != image.getWidth(); ++x, ++ptr) {
            double const fx = image.indexToPosition(x, afwImage::X);
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
