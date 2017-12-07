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

namespace lsst {
namespace afw {
namespace math {

FixedKernel::FixedKernel() : Kernel(), _image(), _sum(0) {}

FixedKernel::FixedKernel(image::Image<Pixel> const& image)
        : Kernel(image.getWidth(), image.getHeight(), 0), _image(image, true), _sum(0) {
    typedef image::Image<Pixel>::x_iterator XIter;
    double imSum = 0.0;
    for (int y = 0; y != image.getHeight(); ++y) {
        for (XIter imPtr = image.row_begin(y), imEnd = image.row_end(y); imPtr != imEnd; ++imPtr) {
            imSum += *imPtr;
        }
    }
    this->_sum = imSum;
}

FixedKernel::FixedKernel(Kernel const& kernel, geom::Point2D const& pos)
        : Kernel(kernel.getWidth(), kernel.getHeight(), 0), _image(kernel.getDimensions()), _sum(0) {
    _sum = kernel.computeImage(_image, false, pos[0], pos[1]);
}

std::shared_ptr<Kernel> FixedKernel::clone() const {
    std::shared_ptr<Kernel> retPtr(new FixedKernel(_image));
    retPtr->setCtr(this->getCtr());
    return retPtr;
}

std::shared_ptr<Kernel> FixedKernel::resized(int width, int height) const {
    if ((width <= 0) || (height <= 0)) {
        std::ostringstream os;
        os << "Cannot create FixedKernel with dimensions (" << width << ", " << height << "). ";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    if ((width - getWidth()) % 2 || (height - getHeight()) % 2) {
        std::ostringstream os;
        os << "Cannot resize FixedKernel from (" << getWidth() << ", " << getHeight() << ") to ("
           << width << ", " << height << "), because at least one dimension would change by an odd value.";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }

    lsst::afw::geom::Box2I bboxNew(lsst::afw::geom::Point2I((1 - width) / 2, (1 - height) / 2),
                                   lsst::afw::geom::Extent2I(width, height));
    std::shared_ptr<image::Image<Pixel>> imNew = std::make_shared<image::Image<Pixel>>(bboxNew);

    // getBBox() instantiates a new BBox from member data _width, _height, _ctrX, _ctrY
    // so modifying it is OK
    lsst::afw::geom::Box2I bboxIntersect = getBBox();
    bboxIntersect.clip(bboxNew);

    // Kernel (this) and member image (this->_image) do not always have same XY0.
    // Member image of resized kernel will not have same BBox as orig,
    // but BBox of member image is ignored by the kernel.
    int offsetX = _image.getX0() - getBBox().getMinX();
    int offsetY = _image.getY0() - getBBox().getMinY();
    lsst::afw::geom::Box2I bboxIntersectShifted = lsst::afw::geom::Box2I(
            lsst::afw::geom::Point2I(bboxIntersect.getMinX() + offsetX,
                                     bboxIntersect.getMinY() + offsetY),
            bboxIntersect.getDimensions());
    image::Image<Pixel> imIntersect = image::Image<Pixel>(_image, bboxIntersectShifted);

    imNew->assign(imIntersect, bboxIntersect);
    std::shared_ptr<Kernel> retPtr = std::make_shared<FixedKernel>(*imNew);
    return retPtr;
}

double FixedKernel::doComputeImage(image::Image<Pixel>& image, bool doNormalize) const {
    double multFactor = 1.0;
    double imSum = this->_sum;
    if (doNormalize) {
        if (imSum == 0) {
            throw LSST_EXCEPT(pexExcept::OverflowError, "Cannot normalize; kernel sum is 0");
        }
        multFactor = 1.0 / static_cast<double>(this->_sum);
        imSum = 1.0;
    }

    typedef image::Image<Pixel>::x_iterator XIter;

    for (int y = 0; y != this->getHeight(); ++y) {
        for (XIter imPtr = image.row_begin(y), imEnd = image.row_end(y), kPtr = this->_image.row_begin(y);
             imPtr != imEnd; ++imPtr, ++kPtr) {
            imPtr[0] = multFactor * kPtr[0];
        }
    }

    return imSum;
}

std::string FixedKernel::toString(std::string const& prefix) const {
    std::ostringstream os;
    os << prefix << "FixedKernel:" << std::endl;
    os << prefix << "..sum: " << _sum << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
}

// ------ Persistence ---------------------------------------------------------------------------------------

namespace {

struct FixedKernelPersistenceHelper : public Kernel::PersistenceHelper {
    table::Key<table::Array<Kernel::Pixel> > image;

    explicit FixedKernelPersistenceHelper(geom::Extent2I const& dimensions)
            : Kernel::PersistenceHelper(0),
              image(schema.addField<table::Array<Kernel::Pixel> >("image", "pixel values (row-major)",
                                                                  dimensions.getX() * dimensions.getY())) {}

    explicit FixedKernelPersistenceHelper(table::Schema const& schema_)
            : Kernel::PersistenceHelper(schema_), image(schema["image"]) {}
};

}  // anonymous

class FixedKernel::Factory : public afw::table::io::PersistableFactory {
public:
    std::shared_ptr<afw::table::io::Persistable> read(InputArchive const& archive,
                                                              CatalogVector const& catalogs) const override {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        FixedKernelPersistenceHelper const keys(catalogs.front().getSchema());
        afw::table::BaseRecord const& record = catalogs.front().front();
        image::Image<Pixel> image(geom::Extent2I(record.get(keys.dimensions)));
        ndarray::flatten<1>(ndarray::static_dimension_cast<2>(image.getArray())) = record[keys.image];
        std::shared_ptr<FixedKernel> result = std::make_shared<FixedKernel>(image);
        result->setCtr(record.get(keys.center));
        return result;
    }

    explicit Factory(std::string const& name) : afw::table::io::PersistableFactory(name) {}
};

namespace {

std::string getFixedKernelPersistenceName() { return "FixedKernel"; }

FixedKernel::Factory registration(getFixedKernelPersistenceName());

}  // anonymous

std::string FixedKernel::getPersistenceName() const { return getFixedKernelPersistenceName(); }

void FixedKernel::write(OutputArchiveHandle& handle) const {
    FixedKernelPersistenceHelper const keys(getDimensions());
    std::shared_ptr<afw::table::BaseRecord> record = keys.write(handle, *this);
    (*record)[keys.image] = ndarray::flatten<1>(ndarray::copy(_image.getArray()));
}
}
}
}  // namespace lsst::afw::math
