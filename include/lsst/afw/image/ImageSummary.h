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

#ifndef LSST_AFW_IMAGE_IMAGE_SUMMARY_H_INCLUDED
#define LSST_AFW_IMAGE_IMAGE_SUMMARY_H_INCLUDED

#include <string>

#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/typehandling/Storable.h"
#include "lsst/afw/table/BaseRecord.h"

namespace lsst {
namespace afw {
namespace image {

/**

 */
class ImageSummary final : public table::io::PersistableFacade<ImageSummary>, public typehandling::Storable {
    using Internal = std::shared_ptr<table::BaseRecord>;

public:
    ImageSummary(table::Schema const&);
    ImageSummary(ImageSummary const&);
    ImageSummary(ImageSummary&&) = default;
    ImageSummary& operator=(ImageSummary const&);
    ImageSummary& operator=(ImageSummary&&) = default;
    ~ImageSummary() override = default;

    /// Whether the map is persistable (true IFF all contained BoundedFields are persistable).
    bool isPersistable() const noexcept override;

    /// Create a new ImageSummary that is a copy of this one.
    std::shared_ptr<typehandling::Storable> cloneStorable() const override;

    table::Schema getSchema() const {
        return _internal->getSchema();
    }

    template <typename T>
    typename table::Field<T>::Value getKey(table::Key<T> const& key) const {
        return _internal->get(key);
    }

    void setBool(std::string const& key, bool value);
    void setInt64(std::string const& key, std::int64_t value);
    void setDouble(std::string const& key, double value);
    void setInternal(table::BaseRecord const & other){
        _internal->assign(other);
    }

private:
    std::string getPersistenceName() const override;

    std::string getPythonModule() const override;

    void write(OutputArchiveHandle& handle) const override;

    Internal _internal;
};
}  // namespace image
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_IMAGE_IMAGE_SUMMARY_H_INCLUDED
