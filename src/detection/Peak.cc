// -*- lsst-c++ -*-
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
#include <typeinfo>

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/detection/Peak.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst {
namespace afw {
namespace detection {

//-----------------------------------------------------------------------------------------------------------
//----- PeakFitsWriter ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsWriter for Peak - this just sets the AFW_TYPE key to PEAK, which should ensure
// we use PeakFitsReader to read it.

namespace {

class PeakFitsWriter : public afw::table::io::FitsWriter {
public:
    explicit PeakFitsWriter(Fits* fits, int flags) : afw::table::io::FitsWriter(fits, flags) {}

protected:
    void _writeTable(std::shared_ptr<afw::table::BaseTable const> const& table, std::size_t nRows) override;
};

void PeakFitsWriter::_writeTable(std::shared_ptr<afw::table::BaseTable const> const& t, std::size_t nRows) {
    std::shared_ptr<PeakTable const> table = std::dynamic_pointer_cast<PeakTable const>(t);
    if (!table) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,
                          "Cannot use a PeakFitsWriter on a non-Peak table.");
    }
    afw::table::io::FitsWriter::_writeTable(table, nRows);
    _fits->writeKey("AFW_TYPE", "PEAK", "Tells lsst::afw to load this as a Peak table.");
}

}  // namespace

//-----------------------------------------------------------------------------------------------------------
//----- PeakFitsReader ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsReader for PeakTable/Record - this gets registered with name SIMPLE, so it should get used
// whenever we read a table with AFW_TYPE set to that value.

namespace {

class PeakFitsReader : public afw::table::io::FitsReader {
public:
    PeakFitsReader() : afw::table::io::FitsReader("PEAK") {}

    std::shared_ptr<afw::table::BaseTable> makeTable(afw::table::io::FitsSchemaInputMapper& mapper,
                                                     std::shared_ptr<daf::base::PropertyList> metadata,
                                                     int ioFlags, bool stripMetadata) const override {
        std::shared_ptr<PeakTable> table = PeakTable::make(mapper.finalize());
        table->setMetadata(metadata);
        return table;
    }
};

// registers the reader so FitsReader::make can use it.
static PeakFitsReader const peakFitsReader;

}  // namespace

//-----------------------------------------------------------------------------------------------------------
//----- PeakTable/Record member function implementations -----------------------------------------------
//-----------------------------------------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& os, PeakRecord const& record) {
    return os << (boost::format("%d: (%d,%d)  (%.3f,%.3f)") % record.getId() % record.getIx() %
                  record.getIy() % record.getFx() % record.getFy());
}

std::shared_ptr<PeakTable> PeakTable::make(afw::table::Schema const& schema, bool forceNewTable) {
    typedef std::list<std::weak_ptr<PeakTable> > CachedTableList;
    static CachedTableList cache;
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "Schema for Peak must contain at least the keys defined by makeMinimalSchema().");
    }
    if (forceNewTable) {
        return std::shared_ptr<PeakTable>(new PeakTable(schema, afw::table::IdFactory::makeSimple()));
    }
    CachedTableList::iterator iter = cache.begin();
    while (iter != cache.end()) {
        std::shared_ptr<PeakTable> p = iter->lock();
        if (!p) {
            iter = cache.erase(iter);
        } else {
            if (p->getSchema().compare(schema, afw::table::Schema::IDENTICAL) ==
                afw::table::Schema::IDENTICAL) {
                // Move the one we found to the front of the list, so it's easier to find
                // the same thing repeatedly
                if (iter != cache.begin()) {
                    cache.splice(cache.begin(), cache, iter);
                }
                return p;
            }
            ++iter;
        }
    }
    // No match: we create a new table and put it in the cache
    std::shared_ptr<PeakTable> newTable(new PeakTable(schema, afw::table::IdFactory::makeSimple()));
    cache.push_front(newTable);
    return newTable;
}

PeakTable::PeakTable(afw::table::Schema const& schema,
                     std::shared_ptr<afw::table::IdFactory> const& idFactory)
        : afw::table::BaseTable(schema), _idFactory(idFactory) {}

PeakTable::PeakTable(PeakTable const& other)
        : afw::table::BaseTable(other),
          _idFactory(other._idFactory ? other._idFactory->clone() : other._idFactory) {}
// Delegate to copy-constructor for backwards-compatibility
PeakTable::PeakTable(PeakTable&& other) : PeakTable(other) {}

PeakTable::~PeakTable() = default;

PeakTable::MinimalSchema::MinimalSchema() {
    id = schema.addField<afw::table::RecordId>("id", "unique ID");
    fx = schema.addField<float>("f_x", "subpixel column position", "pixel");
    fy = schema.addField<float>("f_y", "subpixel row position", "pixel");
    ix = schema.addField<int>("i_x", "column position of highest pixel", "pixel");
    iy = schema.addField<int>("i_y", "row position of highest pixel", "pixel");
    peakValue = schema.addField<float>("peakValue", "value of [smoothed] image at peak position", "count");
    schema.getCitizen().markPersistent();
}

PeakTable::MinimalSchema& PeakTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

std::shared_ptr<afw::table::io::FitsWriter> PeakTable::makeFitsWriter(fits::Fits* fitsfile, int flags) const {
    return std::make_shared<PeakFitsWriter>(fitsfile, flags);
}

std::shared_ptr<afw::table::BaseTable> PeakTable::_clone() const {
    return std::shared_ptr<PeakTable>(new PeakTable(*this));
}

std::shared_ptr<afw::table::BaseRecord> PeakTable::_makeRecord() {
    auto record = constructRecord<PeakRecord>();
    if (getIdFactory()) record->setId((*getIdFactory())());
    return record;
}

}  // namespace detection

namespace table {

template class CatalogT<afw::detection::PeakRecord>;
template class CatalogT<afw::detection::PeakRecord const>;
}  // namespace table
}  // namespace afw
}  // namespace lsst
