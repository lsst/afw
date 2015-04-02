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

namespace lsst { namespace afw { namespace detection {

//-----------------------------------------------------------------------------------------------------------
//----- Private PeakTable/Record classes ---------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// These private derived classes are what you actually get when you do PeakTable::make; like the
// private classes in BaseTable.cc, it's more convenient to have an extra set of trivial derived
// classes than to do a lot of friending.

namespace {

class PeakTableImpl;

class PeakRecordImpl : public PeakRecord {
public:

    explicit PeakRecordImpl(PTR(PeakTable) const & table) : PeakRecord(table) {}

};

class PeakTableImpl : public PeakTable {
public:

    explicit PeakTableImpl(afw::table::Schema const & schema, PTR(afw::table::IdFactory) const & idFactory) :
        PeakTable(schema, idFactory)
    {}

    PeakTableImpl(PeakTableImpl const & other) : PeakTable(other) {}

private:

    virtual PTR(afw::table::BaseTable) _clone() const {
        return boost::make_shared<PeakTableImpl>(*this);
    }

    virtual PTR(afw::table::BaseRecord) _makeRecord() {
        PTR(PeakRecord) record = boost::make_shared<PeakRecordImpl>(getSelf<PeakTableImpl>());
        if (getIdFactory()) record->setId((*getIdFactory())());
        return record;
    }

};

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- PeakFitsWriter ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsWriter for Peak - this just sets the AFW_TYPE key to PEAK, which should ensure
// we use PeakFitsReader to read it.

namespace {

class PeakFitsWriter : public afw::table::io::FitsWriter {
public:

    explicit PeakFitsWriter(Fits * fits, int flags) : afw::table::io::FitsWriter(fits, flags) {}

protected:

    virtual void _writeTable(CONST_PTR(afw::table::BaseTable) const & table, std::size_t nRows);

};

void PeakFitsWriter::_writeTable(CONST_PTR(afw::table::BaseTable) const & t, std::size_t nRows) {
    CONST_PTR(PeakTable) table = boost::dynamic_pointer_cast<PeakTable const>(t);
    if (!table) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicError,
            "Cannot use a PeakFitsWriter on a non-Peak table."
        );
    }
    afw::table::io::FitsWriter::_writeTable(table, nRows);
    _fits->writeKey("AFW_TYPE", "PEAK", "Tells lsst::afw to load this as a Peak table.");
}

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- PeakFitsReader ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsReader for PeakTable/Record - this gets registered with name SIMPLE, so it should get used
// whenever we read a table with AFW_TYPE set to that value.

namespace {

class PeakFitsReader : public afw::table::io::FitsReader {
public:

    explicit PeakFitsReader(Fits * fits, PTR(afw::table::io::InputArchive) archive, int flags) :
        afw::table::io::FitsReader(fits, archive, flags) {}

protected:

    virtual PTR(afw::table::BaseTable) _readTable();

};

PTR(afw::table::BaseTable) PeakFitsReader::_readTable() {
    PTR(daf::base::PropertyList) metadata = boost::make_shared<daf::base::PropertyList>();
    _fits->readMetadata(*metadata, true);
    afw::table::Schema schema(*metadata, true);
    PTR(PeakTable) table = PeakTable::make(schema, PTR(afw::table::IdFactory)());
    _startRecords(*table);
    if (metadata->exists("AFW_TYPE")) metadata->remove("AFW_TYPE");
    table->setMetadata(metadata);
    return table;
}

// registers the reader so FitsReader::make can use it.
static afw::table::io::FitsReader::FactoryT<PeakFitsReader> referenceFitsReaderFactory("PEAK");

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- PeakTable/Record member function implementations -----------------------------------------------
//-----------------------------------------------------------------------------------------------------------

PeakRecord::PeakRecord(PTR(PeakTable) const & table) : BaseRecord(table) {}

std::ostream & operator<<(std::ostream & os, PeakRecord const & record) {
    return os << (boost::format("%d: (%d,%d)  (%.3f,%.3f)")
                  % record.getId()
                  % record.getIx() % record.getIy()
                  % record.getFx() % record.getFy());
}

PTR(PeakTable) PeakTable::make(
    afw::table::Schema const & schema,
    bool forceNewTable
) {
    typedef std::list< boost::weak_ptr<PeakTable> > CachedTableList;
    static CachedTableList cache;
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError,
            "Schema for Peak must contain at least the keys defined by makeMinimalSchema()."
        );
    }
    if (forceNewTable) {
        return boost::make_shared<PeakTableImpl>(schema, afw::table::IdFactory::makeSimple());
    }
    CachedTableList::iterator iter = cache.begin();
    while (iter != cache.end()) {
        PTR(PeakTable) p = iter->lock();
        if (!p) {
            iter = cache.erase(iter);
        } else {
            if (p->getSchema().compare(schema, afw::table::Schema::IDENTICAL)
                == afw::table::Schema::IDENTICAL) {
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
    PTR(PeakTable) newTable = boost::make_shared<PeakTableImpl>(
        schema, afw::table::IdFactory::makeSimple()
    );
    cache.push_front(newTable);
    return newTable;
}

PeakTable::PeakTable(afw::table::Schema const & schema, PTR(afw::table::IdFactory) const & idFactory) :
    afw::table::BaseTable(schema), _idFactory(idFactory) {}

PeakTable::PeakTable(PeakTable const & other) :
    afw::table::BaseTable(other),
    _idFactory(other._idFactory ? other._idFactory->clone() : other._idFactory) {}

PeakTable::MinimalSchema::MinimalSchema() {
    id = schema.addField<afw::table::RecordId>("id", "unique ID");
    fx = schema.addField<float>("f.x", "subpixel column position", "pixels");
    fy = schema.addField<float>("f.y", "subpixel row position", "pixels");
    ix = schema.addField<int>("i.x", "column position of highest pixel", "pixels");
    iy = schema.addField<int>("i.y", "row position of highest pixel", "pixels");
    peakValue = schema.addField<float>("peakValue", "value of [smoothed] image at peak position", "dn");
    schema.getCitizen().markPersistent();
}

PeakTable::MinimalSchema & PeakTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

PTR(afw::table::io::FitsWriter)
PeakTable::makeFitsWriter(fits::Fits * fitsfile, int flags) const {
    return boost::make_shared<PeakFitsWriter>(fitsfile, flags);
}

} // namespace detection

namespace table {


template class CatalogT<afw::detection::PeakRecord>;
template class CatalogT<afw::detection::PeakRecord const>;

}}} // namespace lsst::afw::table
