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
#include <stdexcept>
#include <numeric>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelPersistenceHelper.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwGeom = lsst::afw::geom;
namespace afwMath = lsst::afw::math;
namespace afwImage = lsst::afw::image;

afwMath::FixedKernel::FixedKernel()
:
    Kernel(),
    _image(),
    _sum(0) {
}

afwMath::FixedKernel::FixedKernel(
    afwImage::Image<Pixel> const &image
) :
    Kernel(image.getWidth(), image.getHeight(), 0),
    _image(image, true),
    _sum(0) {

    typedef afwImage::Image<Pixel>::x_iterator XIter;
    double imSum = 0.0;
    for (int y = 0; y != image.getHeight(); ++y) {
        for (XIter imPtr = image.row_begin(y), imEnd = image.row_end(y); imPtr != imEnd; ++imPtr) {
            imSum += *imPtr;
        }
    }
    this->_sum = imSum;
}

afwMath::FixedKernel::FixedKernel(
    afwMath::Kernel const& kernel,
    afwGeom::Point2D const& pos
) :
    Kernel(kernel.getWidth(), kernel.getHeight(), 0),
    _image(kernel.getDimensions()),
    _sum(0) {
    _sum = kernel.computeImage(_image, false, pos[0], pos[1]);
}

PTR(afwMath::Kernel) afwMath::FixedKernel::clone() const {
    PTR(afwMath::Kernel) retPtr(new afwMath::FixedKernel(_image));
    retPtr->setCtr(this->getCtr());
    return retPtr;
}

double afwMath::FixedKernel::doComputeImage(
    afwImage::Image<Pixel> &image,
    bool doNormalize
) const {
    double multFactor = 1.0;
    double imSum = this->_sum;
    if (doNormalize) {
        if (imSum == 0) {
            throw LSST_EXCEPT(pexExcept::OverflowError, "Cannot normalize; kernel sum is 0");
        }
        multFactor = 1.0/static_cast<double>(this->_sum);
        imSum = 1.0;
    }

    typedef afwImage::Image<Pixel>::x_iterator XIter;

    for (int y = 0; y != this->getHeight(); ++y) {
        for (XIter imPtr = image.row_begin(y), imEnd = image.row_end(y), kPtr = this->_image.row_begin(y);
            imPtr != imEnd; ++imPtr, ++kPtr) {
            imPtr[0] = multFactor*kPtr[0];
        }
    }

    return imSum;
}

std::string afwMath::FixedKernel::toString(std::string const& prefix) const {
    std::ostringstream os;
    os << prefix << "FixedKernel:" << std::endl;
    os << prefix << "..sum: " << _sum << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
}

// ------ Persistence ---------------------------------------------------------------------------------------

namespace lsst { namespace afw { namespace math {

namespace {

struct FixedKernelPersistenceHelper : public Kernel::PersistenceHelper {
    table::Key< table::Array<Kernel::Pixel> > image;

    explicit FixedKernelPersistenceHelper(geom::Extent2I const & dimensions) :
        Kernel::PersistenceHelper(0),
        image(
            schema.addField< table::Array<Kernel::Pixel> >(
                "image", "pixel values (row-major)", dimensions.getX() * dimensions.getY()
            )
        )
    {}

    explicit FixedKernelPersistenceHelper(table::Schema const & schema_) :
        Kernel::PersistenceHelper(schema_),
        image(schema["image"])
    {}
};

} // anonymous

class FixedKernel::Factory : public afw::table::io::PersistableFactory {
public:

    virtual PTR(afw::table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        FixedKernelPersistenceHelper const keys(catalogs.front().getSchema());
        afw::table::BaseRecord const & record = catalogs.front().front();
        image::Image<Pixel> image(geom::Extent2I(record.get(keys.dimensions)));
        ndarray::flatten<1>(
            ndarray::static_dimension_cast<2>(image.getArray())
        ) = record[keys.image];
        PTR(FixedKernel) result = std::make_shared<FixedKernel>(image);
        result->setCtr(record.get(keys.center));
        return result;
    }

    explicit Factory(std::string const & name) : afw::table::io::PersistableFactory(name) {}
};

namespace {

std::string getFixedKernelPersistenceName() { return "FixedKernel"; }

FixedKernel::Factory registration(getFixedKernelPersistenceName());

} // anonymous

std::string FixedKernel::getPersistenceName() const { return getFixedKernelPersistenceName(); }

void FixedKernel::write(OutputArchiveHandle & handle) const {
    FixedKernelPersistenceHelper const keys(getDimensions());
    PTR(afw::table::BaseRecord) record = keys.write(handle, *this);
    (*record)[keys.image] = ndarray::flatten<1>(ndarray::copy(_image.getArray()));
}

}}} // namespace lsst::afw::math
