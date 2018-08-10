/*
 * LSST Data Management System
 * Copyright 2008-2017 LSST Corporation.
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

#include <exception>
#include <memory>
#include <ostream>
#include <sstream>
#include <vector>

#include "astshim.h"
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/geom/detail/transformUtils.h"
#include "lsst/afw/geom/Endpoint.h"
#include "lsst/afw/geom/Transform.h"
#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/pex/exceptions/Exception.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/OutputArchive.h"

namespace lsst {
namespace afw {
namespace geom {

template <class FromEndpoint, class ToEndpoint>
Transform<FromEndpoint, ToEndpoint>::Transform(ast::Mapping const &mapping, bool simplify)
        : _fromEndpoint(mapping.getNIn()),
          _mapping(simplify ? mapping.simplified() : mapping.copy()),
          _toEndpoint(mapping.getNOut()) {}

template <typename FromEndpoint, typename ToEndpoint>
Transform<FromEndpoint, ToEndpoint>::Transform(ast::FrameSet const &frameSet, bool simplify)
        : _fromEndpoint(frameSet.getNIn()), _mapping(), _toEndpoint(frameSet.getNOut()) {
    auto frameSetCopy = frameSet.copy();
    // Normalize the base and current frame in a way that affects its behavior as a mapping.
    // To do this one must set the current frame to the frame to be normalized
    // and normalize the frame set as a frame (i.e. normalize the frame "in situ").
    // The obvious alternative of normalizing a shallow copy of the frame does not work;
    // the frame is altered but not the associated mapping!

    // Normalize the current frame by normalizing the frameset as a frame
    _toEndpoint.normalizeFrame(frameSetCopy);

    // Normalize the base frame by temporarily making it the current frame,
    // normalizing the frameset as a frame, then making it the base frame again
    const int baseIndex = frameSetCopy->getBase();
    const int currentIndex = frameSetCopy->getCurrent();
    frameSetCopy->setCurrent(baseIndex);
    _fromEndpoint.normalizeFrame(frameSetCopy);
    frameSetCopy->setBase(baseIndex);
    frameSetCopy->setCurrent(currentIndex);
    _mapping = simplify ? frameSetCopy->getMapping()->simplified() : frameSetCopy->getMapping();
}

template <typename FromEndpoint, typename ToEndpoint>
Transform<FromEndpoint, ToEndpoint>::Transform(std::shared_ptr<ast::Mapping> mapping)
        : _fromEndpoint(mapping->getNIn()), _mapping(mapping), _toEndpoint(mapping->getNOut()) {}

template <class FromEndpoint, class ToEndpoint>
typename ToEndpoint::Point Transform<FromEndpoint, ToEndpoint>::applyForward(
        typename FromEndpoint::Point const &point) const {
    auto const rawFromData = _fromEndpoint.dataFromPoint(point);
    auto rawToData = _mapping->applyForward(rawFromData);
    return _toEndpoint.pointFromData(rawToData);
}

template <class FromEndpoint, class ToEndpoint>
typename ToEndpoint::Array Transform<FromEndpoint, ToEndpoint>::applyForward(
        typename FromEndpoint::Array const &array) const {
    auto const rawFromData = _fromEndpoint.dataFromArray(array);
    auto rawToData = _mapping->applyForward(rawFromData);
    return _toEndpoint.arrayFromData(rawToData);
}

template <class FromEndpoint, class ToEndpoint>
typename FromEndpoint::Point Transform<FromEndpoint, ToEndpoint>::applyInverse(
        typename ToEndpoint::Point const &point) const {
    auto const rawFromData = _toEndpoint.dataFromPoint(point);
    auto rawToData = _mapping->applyInverse(rawFromData);
    return _fromEndpoint.pointFromData(rawToData);
}

template <class FromEndpoint, class ToEndpoint>
typename FromEndpoint::Array Transform<FromEndpoint, ToEndpoint>::applyInverse(
        typename ToEndpoint::Array const &array) const {
    auto const rawFromData = _toEndpoint.dataFromArray(array);
    auto rawToData = _mapping->applyInverse(rawFromData);
    return _fromEndpoint.arrayFromData(rawToData);
}

template <class FromEndpoint, class ToEndpoint>
std::shared_ptr<Transform<ToEndpoint, FromEndpoint>> Transform<FromEndpoint, ToEndpoint>::getInverse() const {
    auto inverse = std::dynamic_pointer_cast<ast::Mapping>(_mapping->inverted());
    if (!inverse) {
        // don't throw std::bad_cast because it doesn't let you provide debugging info
        std::ostringstream buffer;
        buffer << "Mapping.inverted() does not return a Mapping. Called from: " << _mapping;
        throw LSST_EXCEPT(pex::exceptions::LogicError, buffer.str());
    }
    return std::make_shared<Transform<ToEndpoint, FromEndpoint>>(*inverse);
}

template <class FromEndpoint, class ToEndpoint>
Eigen::MatrixXd Transform<FromEndpoint, ToEndpoint>::getJacobian(FromPoint const &x) const {
    int const nIn = _fromEndpoint.getNAxes();
    int const nOut = _toEndpoint.getNAxes();
    std::vector<double> const point = _fromEndpoint.dataFromPoint(x);

    Eigen::MatrixXd jacobian(nOut, nIn);
    for (int i = 0; i < nOut; ++i) {
        for (int j = 0; j < nIn; ++j) {
            jacobian(i, j) = _mapping->rate(point, i + 1, j + 1);
        }
    }
    return jacobian;
}

template <class FromEndpoint, class ToEndpoint>
std::string Transform<FromEndpoint, ToEndpoint>::getShortClassName() {
    std::ostringstream os;
    os << "Transform" << FromEndpoint::getClassPrefix() << "To" << ToEndpoint::getClassPrefix();
    return os.str();
}

template <class FromEndpoint, class ToEndpoint>
std::shared_ptr<Transform<FromEndpoint, ToEndpoint>> Transform<FromEndpoint, ToEndpoint>::readStream(
        std::istream &is) {
    return detail::readStream<Transform<FromEndpoint, ToEndpoint>>(is);
}

template <class FromEndpoint, class ToEndpoint>
std::shared_ptr<Transform<FromEndpoint, ToEndpoint>> Transform<FromEndpoint, ToEndpoint>::readString(
        std::string &str) {
    std::istringstream is(str);
    return Transform<FromEndpoint, ToEndpoint>::readStream(is);
}

template <class FromEndpoint, class ToEndpoint>
void Transform<FromEndpoint, ToEndpoint>::writeStream(std::ostream &os) const {
    detail::writeStream<Transform<FromEndpoint, ToEndpoint>>(*this, os);
}

template <class FromEndpoint, class ToEndpoint>
std::string Transform<FromEndpoint, ToEndpoint>::writeString() const {
    std::ostringstream os;
    writeStream(os);
    return os.str();
}

template <class FromEndpoint, class ToEndpoint>
template <class NextToEndpoint>
std::shared_ptr<Transform<FromEndpoint, NextToEndpoint>> Transform<FromEndpoint, ToEndpoint>::then(
        Transform<ToEndpoint, NextToEndpoint> const &next, bool simplify) const {
    if (_toEndpoint.getNAxes() == next.getFromEndpoint().getNAxes()) {
        auto nextMapping = next.getMapping();
        auto combinedMapping = getMapping()->then(*next.getMapping());
        if (simplify) {
            return std::make_shared<Transform<FromEndpoint, NextToEndpoint>>(*combinedMapping.simplified());
        } else {
            return std::make_shared<Transform<FromEndpoint, NextToEndpoint>>(combinedMapping);
        }
    } else {
        auto message = "Cannot match " + std::to_string(_toEndpoint.getNAxes()) + "-D to-endpoint to " +
                       std::to_string(next.getFromEndpoint().getNAxes()) + "-D from-endpoint.";
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, message);
    }
}

template <class FromEndpoint, class ToEndpoint>
std::ostream &operator<<(std::ostream &os, Transform<FromEndpoint, ToEndpoint> const &transform) {
    os << "Transform<" << transform.getFromEndpoint() << ", " << transform.getToEndpoint() << ">";
    return os;
};

namespace {

class TransformPersistenceHelper {
public:
    table::Schema schema;
    table::Key<table::Array<std::uint8_t>> bytes;

    static TransformPersistenceHelper const &get() {
        static TransformPersistenceHelper instance;
        return instance;
    }

    // No copying
    TransformPersistenceHelper(TransformPersistenceHelper const &) = delete;
    TransformPersistenceHelper &operator=(TransformPersistenceHelper const &) = delete;

    // No moving
    TransformPersistenceHelper(TransformPersistenceHelper &&) = delete;
    TransformPersistenceHelper &operator=(TransformPersistenceHelper &&) = delete;

private:
    TransformPersistenceHelper()
            : schema(),
              bytes(schema.addField<table::Array<std::uint8_t>>(
                      "bytes", "a bytestring containing the output of Transform.writeString", "")) {
        schema.getCitizen().markPersistent();
    }
};

template <typename FromEndpoint, typename ToEndpoint>
class TransformFactory : public table::io::PersistableFactory {
public:
    explicit TransformFactory(std::string const &name) : table::io::PersistableFactory(name) {}

    virtual std::shared_ptr<table::io::Persistable> read(InputArchive const &archive,
                                                         CatalogVector const &catalogs) const {
        auto const &keys = TransformPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
        auto const &record = catalogs.front().front();
        std::string stringRep = formatters::bytesToString(record.get(keys.bytes));
        return Transform<FromEndpoint, ToEndpoint>::readString(stringRep);
    }
};

}  // namespace

template <class FromEndpoint, class ToEndpoint>
void Transform<FromEndpoint, ToEndpoint>::write(OutputArchiveHandle &handle) const {
    auto const &keys = TransformPersistenceHelper::get();
    table::BaseCatalog cat = handle.makeCatalog(keys.schema);
    std::shared_ptr<table::BaseRecord> record = cat.addNew();
    record->set(keys.bytes, formatters::stringToBytes(writeString()));
    handle.saveCatalog(cat);
}

#define INSTANTIATE_OVERLOADS(FromEndpoint, ToEndpoint, NextToEndpoint) \
    template std::shared_ptr<Transform<FromEndpoint, NextToEndpoint>>   \
    Transform<FromEndpoint, ToEndpoint>::then<NextToEndpoint>(          \
            Transform<ToEndpoint, NextToEndpoint> const &next, bool) const;

#define INSTANTIATE_TRANSFORM(FromEndpoint, ToEndpoint)                                \
    template class Transform<FromEndpoint, ToEndpoint>;                                \
    template std::ostream &operator<<<FromEndpoint, ToEndpoint>(                       \
            std::ostream &os, Transform<FromEndpoint, ToEndpoint> const &transform);   \
    namespace {                                                                        \
    TransformFactory<FromEndpoint, ToEndpoint> registration##FromEndpoint##ToEndpoint( \
            Transform<FromEndpoint, ToEndpoint>::getShortClassName());                 \
    }                                                                                  \
    INSTANTIATE_OVERLOADS(FromEndpoint, ToEndpoint, GenericEndpoint)                   \
    INSTANTIATE_OVERLOADS(FromEndpoint, ToEndpoint, Point2Endpoint)                    \
    INSTANTIATE_OVERLOADS(FromEndpoint, ToEndpoint, SpherePointEndpoint)

// explicit instantiations
INSTANTIATE_TRANSFORM(GenericEndpoint, GenericEndpoint);
INSTANTIATE_TRANSFORM(GenericEndpoint, Point2Endpoint);
INSTANTIATE_TRANSFORM(GenericEndpoint, SpherePointEndpoint);
INSTANTIATE_TRANSFORM(Point2Endpoint, GenericEndpoint);
INSTANTIATE_TRANSFORM(Point2Endpoint, Point2Endpoint);
INSTANTIATE_TRANSFORM(Point2Endpoint, SpherePointEndpoint);
INSTANTIATE_TRANSFORM(SpherePointEndpoint, GenericEndpoint);
INSTANTIATE_TRANSFORM(SpherePointEndpoint, Point2Endpoint);
INSTANTIATE_TRANSFORM(SpherePointEndpoint, SpherePointEndpoint);

}  // namespace geom
}  // namespace afw
}  // namespace lsst
