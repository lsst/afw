// -*- LSST-C++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2014 LSST Corporation.
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

#include <memory>

#include "lsst/afw/image/ImageSummary.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"

namespace lsst {
namespace afw {

template std::shared_ptr<image::ImageSummary> table::io::PersistableFacade<image::ImageSummary>::dynamicCast(
        std::shared_ptr<table::io::Persistable> const&);

namespace image {

ImageSummary::ImageSummary(table::Schema const& schema) {
    auto tbl = table::BaseTable::make(schema);
    _internal = tbl->makeRecord();
}

ImageSummary::ImageSummary(ImageSummary const& other) {
    _internal = other._internal->getTable()->copyRecord(*other._internal);
}

ImageSummary& ImageSummary::operator=(ImageSummary const& other) {
    _internal->assign(*other._internal);
    return *this;
}


namespace {

class ImageSummaryFactory : public table::io::PersistableFactory {
public:
    std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                 CatalogVector const& catalogs) const override {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        std::shared_ptr<ImageSummary> result = std::make_shared<ImageSummary>(catalogs.front().getSchema());
        result->setInternal(catalogs.front().front());
        return result;
    }

    ImageSummaryFactory(std::string const& name) : afw::table::io::PersistableFactory(name) {}
};

std::string getImageSummaryPersistenceName() { return "ImageSummary"; }

ImageSummaryFactory registration(getImageSummaryPersistenceName());

}  // namespace

bool ImageSummary::isPersistable() const noexcept {return true;}

std::string ImageSummary::getPersistenceName() const { return getImageSummaryPersistenceName(); }

std::string ImageSummary::getPythonModule() const { return "lsst.afw.image"; }

void ImageSummary::write(OutputArchiveHandle& handle) const {
    table::BaseCatalog catalog = handle.makeCatalog(_internal->getSchema());
    std::shared_ptr<table::BaseRecord> record = catalog.addNew();
    record->assign(*_internal);
    handle.saveCatalog(catalog);
}

std::shared_ptr<typehandling::Storable> ImageSummary::cloneStorable() const {
    return std::make_unique<ImageSummary>(*this);
}

void ImageSummary::setBool(std::string const& key, bool value) {
    _internal->set(_internal->getSchema().find<table::Flag>(key).key, value);
}

void ImageSummary::setInt64(std::string const& key, std::int64_t value) {
    _internal->set(_internal->getSchema().find<std::int64_t>(key).key, value);
}

void ImageSummary::setDouble(std::string const& key, double value) {
    _internal->set(_internal->getSchema().find<double>(key).key, value);
}

}  // namespace image
}  // namespace afw
}  // namespace lsst
