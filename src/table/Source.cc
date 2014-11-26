// -*- lsst-c++ -*-
#include <typeinfo>

#include "boost/iterator/transform_iterator.hpp"

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/detection/FootprintCtrl.h"
#include "lsst/afw/detection/HeavyFootprint.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"

namespace lsst { namespace afw { namespace table {

//-----------------------------------------------------------------------------------------------------------
//----- Private SourceTable/Record classes ------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// These private derived classes are what you actually get when you do SourceTable::make; like the
// private classes in BaseTable.cc, it's more convenient to have an extra set of trivial derived
// classes than to do a lot of friending.

namespace {

class SourceTableImpl;

class SourceRecordImpl : public SourceRecord {
public:

    explicit SourceRecordImpl(PTR(SourceTable) const & table) : SourceRecord(table) {}

};

class SourceTableImpl : public SourceTable {
public:

    explicit SourceTableImpl(Schema const & schema, PTR(IdFactory) const & idFactory) :
        SourceTable(schema, idFactory) {}

    SourceTableImpl(SourceTableImpl const & other) : SourceTable(other) {}

private:

    virtual PTR(BaseTable) _clone() const {
        return boost::make_shared<SourceTableImpl>(*this);
    }

    virtual PTR(BaseRecord) _makeRecord() {
        PTR(SourceRecord) record = boost::make_shared<SourceRecordImpl>(getSelf<SourceTableImpl>());
        if (getIdFactory()) record->setId((*getIdFactory())());
        return record;
    }

};

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- PersistenceHelpers ----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------


namespace {

struct PersistenceHelper {
    SchemaMapper mapper;
    afw::table::Key<int> footprintKey;
};

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- SourceFitsWriter ------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsWriter for Sources.  It also sets the AFW_TYPE key to SOURCE, which should
// ensure we use SourceFitsReader to read it.

// Because it also holds Footprints, a SourceCatalog isn't persisted with the same Schema
// it has in-memory; instead, it's saved in a Schema that has an additional int field
// appended to the end, which contains the afw::table::io "archive ID" that's used to
// extract the Footprint from additional FITS HDUs.  (If we disable saving Footprints via
// SourceFitsFlags, we do save using the original Schema).

// The only public access point to this class is SourceTable::makeFitsWriter.  If we
// subclass SourceTable someday, it may be necessary to put SourceFitsWriter in a header
// file so we can subclass it too.

namespace {

class SourceFitsWriter : public io::FitsWriter {
public:

    explicit SourceFitsWriter(Fits * fits, int flags) :
        io::FitsWriter(fits, flags)
    {}

protected:

    virtual void _writeTable(CONST_PTR(BaseTable) const & table, std::size_t nRows);

    virtual void _writeRecord(BaseRecord const & record);

    virtual void _finish() {
        if (!(_flags & SOURCE_IO_NO_FOOTPRINTS)) {
            _archive.writeFits(*_fits);
        }
    }

private:
    SchemaMapper _mapper;
    PTR(BaseRecord) _outRecord;
    PTR(BaseTable) _outTable;
    Key<int> _footprintKey;
    io::OutputArchive _archive;
};

void SourceFitsWriter::_writeTable(CONST_PTR(BaseTable) const & t, std::size_t nRows) {
    CONST_PTR(SourceTable) table = boost::dynamic_pointer_cast<SourceTable const>(t);
    if (!table) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicError,
            "Cannot use a SourceFitsWriter on a non-Source table."
        );
    }
    if (!(_flags & SOURCE_IO_NO_FOOTPRINTS)) {
        _mapper = SchemaMapper(t->getSchema()) ;
        _mapper.addMinimalSchema(t->getSchema(), true);
        _footprintKey = _mapper.editOutputSchema().addField<int>("footprint", "archive ID for Footprint");
        _outTable = BaseTable::make(_mapper.getOutputSchema());
        PTR(daf::base::PropertyList) metadata = table->getMetadata();
        if (metadata) {
            metadata = boost::static_pointer_cast<daf::base::PropertyList>(metadata->deepCopy());
        } else {
            metadata.reset(new daf::base::PropertyList());
        }
        // HDU 1 is empty (primary HDU can't be a table)
        // HDU 2 is the SourceCatalog's records
        // HDU 3 is the index for the afw::table::io archive that holds more complex objects
        metadata->set(
            "AR_HDU", 3, "HDU containing the archive index for non-record data (e.g. Footprints)"
        );
        _outTable->setMetadata(metadata);
        _outRecord = _outTable->makeRecord(); // make temporary record to use as a workspace
        io::FitsWriter::_writeTable(_outTable, nRows);
    } else {
        io::FitsWriter::_writeTable(table, nRows);
    }
    _fits->writeKey("AFW_TYPE", "SOURCE", "Tells lsst::afw to load this as a Source table.");
}

void SourceFitsWriter::_writeRecord(BaseRecord const & r) {
    SourceRecord const & record = static_cast<SourceRecord const &>(r);
    if (!(_flags & SOURCE_IO_NO_FOOTPRINTS)) {
        _outRecord->assign(record, _mapper);
        PTR(afw::detection::Footprint) footprint = record.getFootprint();
        if (footprint) {
            if ((_flags & SOURCE_IO_NO_HEAVY_FOOTPRINTS) && footprint->isHeavy()) {
                footprint.reset(new afw::detection::Footprint(*footprint));
            }
            int footprintArchiveId = _archive.put(footprint);
            _outRecord->set(_footprintKey, footprintArchiveId);
        }
        io::FitsWriter::_writeRecord(*_outRecord);
    } else {
        io::FitsWriter::_writeRecord(record);
    }
}

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- SourceFitsReader ------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsReader for Sources - this reads footprints as variable-length arrays, and adds header
// keys that define the slots.  It gets registered with name SOURCE, so it should get used whenever
// we read a table with AFW_TYPE set to that value.

// As noted in the comments for SourceFitsWriter, we have to modify the Schema by adding a column for the
// Footprint archive ID when we save a SourceCatalog.
// Things are a bit more complicated than that when reading, because we also need to be able to read files
// saved with an older version of the pipeline, in which there were 2-5 additional columns, all variable-
// length arrays, holding the Spans, Peaks, and HeavyFootprint arrays.  Those are handled by explicit
// calls to the FITS I/O routines here.

// The only public access point to this class is through the registry.  If we subclass SourceTable
// someday, it may be necessary to put SourceFitsReader in a header file so we can subclass it too.

namespace {

class SourceFitsReader : public io::FitsReader {
public:

    // The archive argument here was added to all the derived-class FitsReader classes, just to support
    // one use case with ExposureCatalog (which is unusual in that it's used to implement afw::table::io
    // persistence for other objects e.g. CoaddPsf, but also uses afw::table::io persistence to save
    // itself).  SourceCatalog now does the latter, but not the former, so we'll continue ignoring this
    // archive argument and creating our own.  And hope someday we have time to clean up this hack.
    explicit SourceFitsReader(Fits * fits, PTR(io::InputArchive) archive, int flags) :
        io::FitsReader(fits, archive, flags),
        _spanCol(-1), _peakCol(-1), _heavyPixCol(-1), _heavyMaskCol(-1), _heavyVarCol(-1)
    {
        if (archive) {
            throw LSST_EXCEPT(
                pex::exceptions::LogicErrorException,
                "SourceCatalog does not support reading from an external archive"
            );
        }
    }

protected:

    virtual PTR(BaseTable) _readTable();

    virtual PTR(BaseRecord) _readRecord(PTR(BaseTable) const & table);

private:
    PTR(BaseTable) _inTable;
    PTR(io::InputArchive) _archive;
    SchemaMapper _mapper;
    Key<int> _footprintKey;
    int _spanCol;      // all *Col data members are for backwards-compatibility reading of old
    int _peakCol;      // Footprint persistence
    int _heavyPixCol;
    int _heavyMaskCol;
    int _heavyVarCol;
};

namespace {

// Predicate for SchemaMapper::addMappingsWhere to map all fields in a schema except the one holding
// the Footprint's archive ID.
struct FieldIsNotFootprint {

    template <typename T>
    bool operator()(SchemaItem<T> const & item) const { return true; }

    bool operator()(SchemaItem<int> const & item) const {
        return item.field.getName() != "footprint";
    }

};

} // anonymous

PTR(BaseTable) SourceFitsReader::_readTable() {
    PTR(daf::base::PropertyList) metadata = boost::make_shared<daf::base::PropertyList>();
    _fits->readMetadata(*metadata, true);
    if (metadata->exists("AFW_TYPE")) metadata->remove("AFW_TYPE");
    // if there's an archive attached, it's the new way of persisting Footprints
    int archiveHdu = metadata->get("AR_HDU", -1);
    if (archiveHdu > 0) {
        // If we have an AR_HDU key, we have new-style persistence of Footprints,
        // using afw::table::io archives.  We strip the key from the metadata, but we
        // don't read the archive if the flags tell us not to.
        metadata->remove("AR_HDU");
        if (!(_flags & SOURCE_IO_NO_FOOTPRINTS)) {
            int oldHdu = _fits->getHdu();
            _fits->setHdu(archiveHdu);
            _archive.reset(new io::InputArchive(io::InputArchive::readFits(*_fits)));
            _fits->setHdu(oldHdu);
        }
    } else {
        // Old-style persistence of Footprints, for backwards compatibility, OR we didn't persist the
        // Footprints at all.  If it's old-style persistence, we remove the appropriate Keys from the
        // metadata before we construct a Schema, to keep those fields from getting added to the Schema.
        // If we just didn't persist any Footprints, then none of these keys should be present, and
        // we don't have to do anything to get the Schema we want.
        _spanCol = metadata->get("SPANCOL", 0);
        if (_spanCol > 0) {
            metadata->remove("SPANCOL");
            metadata->remove((boost::format("TTYPE%d") % _spanCol).str());
            metadata->remove((boost::format("TFORM%d") % _spanCol).str());
        }
        _peakCol = metadata->get("PEAKCOL", 0);
        if (_peakCol > 0) {
            metadata->remove("PEAKCOL");
            metadata->remove((boost::format("TTYPE%d") % _peakCol).str());
            metadata->remove((boost::format("TFORM%d") % _peakCol).str());
        }
        _heavyPixCol  = metadata->get("HVYPIXCO", 0);
        if (_heavyPixCol > 0) {
            metadata->remove("HVYPIXCO");
            metadata->remove((boost::format("TTYPE%d") % _heavyPixCol).str());
            metadata->remove((boost::format("TFORM%d") % _heavyPixCol).str());
        }
        _heavyMaskCol  = metadata->get("HVYMSKCO", 0);
        if (_heavyMaskCol > 0) {
            metadata->remove("HVYMSKCO");
            metadata->remove((boost::format("TTYPE%d") % _heavyMaskCol).str());
            metadata->remove((boost::format("TFORM%d") % _heavyMaskCol).str());
            metadata->remove((boost::format("TZERO%d") % _heavyMaskCol).str());
            metadata->remove((boost::format("TSCAL%d") % _heavyMaskCol).str());
        }
        _heavyVarCol  = metadata->get("HVYVARCO", 0);
        if (_heavyVarCol > 0) {
            metadata->remove("HVYVARCO");
            metadata->remove((boost::format("TTYPE%d") % _heavyVarCol).str());
            metadata->remove((boost::format("TFORM%d") % _heavyVarCol).str());
        }
        --_spanCol; // switch to 0-indexed rather than 1-indexed convention.
        --_peakCol;
        --_heavyPixCol;
        --_heavyMaskCol;
        --_heavyVarCol;
    }
    Schema schema(*metadata, true);
    if (archiveHdu > 0) {
        // If an archive was present, regardless of whether or not we actually read it, the Schema we just
        // constructed has an extra int field holding the Footprint archive IDs.
        // We need to create a SchemaMapper to go from that Schema to one without those ints (even if we
        // aren't reading the Footprints), and an intermediate BaseTable with that Schema.
        // This table being non-null is what we'll use later to indicate whether we need to use the
        // SchemaMapper.
        _mapper = SchemaMapper(schema);
        _mapper.addMappingsWhere(FieldIsNotFootprint());
        _inTable = BaseTable::make(schema);
        _footprintKey = schema["footprint"];
        schema = _mapper.getOutputSchema();
    }
    PTR(SourceTable) table =  SourceTable::make(schema, PTR(IdFactory)());
    table->setMetadata(metadata);
    _startRecords(*table);
    return table;
}

PTR(BaseRecord) SourceFitsReader::_readRecord(PTR(BaseTable) const & table) {
    PTR(SourceRecord) record;
    if (_inTable) { // New-style persisted Footprints
        PTR(BaseRecord) inRecord = io::FitsReader::_readRecord(_inTable);
        if (inRecord) {
            record = boost::static_pointer_cast<SourceRecord>(table->makeRecord());
            record->assign(*inRecord, _mapper);
            if (_archive) { // archive is only initialized if we should read Footprints
                PTR(afw::detection::Footprint) footprint =
                    _archive->get<afw::detection::Footprint>(inRecord->get(_footprintKey));
                if (footprint && footprint->isHeavy() && (_flags & SOURCE_IO_NO_HEAVY_FOOTPRINTS)) {
                    // It sort of defeats the purpose of the flag if we have to do the I/O to read
                    // a HeavyFootprint before we can downgrade it to a regular Footprint, but that's
                    // what we're going to do - at least this will save on on some memory usage, which
                    // might still be useful.  That's far from ideal, but it'd be really hard to fix
                    // (because we have no way to pass something like the flags to the InputArchive).
                    // The good news is that if someone's concerned about performance of reading
                    // SourceCatalogs, they'll almost certainly use SOURCE_IO_NO_FOOTPRINTS, which
                    // will do what we want.  SOURCE_IO_NO_HEAVY_FOOTPRINTS is more useful for writing
                    // sources, and that works just fine.
                    footprint.reset(new afw::detection::Footprint(*footprint));
                }
                record->setFootprint(footprint);
            }
            return record;
        } else {
            return record;
        }
    }
    // Old-style persisted Footprints, or no persisted Footprints.
    record = boost::static_pointer_cast<SourceRecord>(io::FitsReader::_readRecord(table));
    if (_flags & SOURCE_IO_NO_FOOTPRINTS || !record) return record;
    int spanElementCount = (_spanCol >= 0) ? _fits->getTableArraySize(_row, _spanCol) : 0;
    int peakElementCount = (_peakCol >= 0) ? _fits->getTableArraySize(_row, _peakCol) : 0;
    int heavyPixElementCount  = (_heavyPixCol  >= 0) ? _fits->getTableArraySize(_row, _heavyPixCol)  : 0;
    int heavyMaskElementCount = (_heavyMaskCol >= 0) ? _fits->getTableArraySize(_row, _heavyMaskCol) : 0;
    int heavyVarElementCount  = (_heavyVarCol  >= 0) ? _fits->getTableArraySize(_row, _heavyVarCol)  : 0;
    if (spanElementCount || peakElementCount) {
        PTR(Footprint) fp = boost::make_shared<Footprint>();
        if (spanElementCount) {
            if (spanElementCount % 3) {
                throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    afw::fits::makeErrorMessage(
                        _fits->fptr, _fits->status,
                        boost::format("Number of span elements (%d) must divisible by 3 (row %d)")
                        % spanElementCount % _row
                    )
                );
            }
            std::vector<int> spanElements(spanElementCount);
            _fits->readTableArray(_row, _spanCol, spanElementCount, &spanElements.front());
            std::vector<int>::iterator j = spanElements.begin();
            while (j != spanElements.end()) {
                int y = *j++;
                int x0 = *j++;
                int x1 = *j++;
                fp->addSpan(y, x0, x1);
            }
        }
        if (peakElementCount) {
            if (peakElementCount % 3) {
                throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    afw::fits::makeErrorMessage(
                        _fits->fptr, _fits->status,
                        boost::format("Number of peak elements (%d) must divisible by 3 (row %d)")
                        % peakElementCount % _row
                    )
                );
            }
            std::vector<float> peakElements(peakElementCount);
            _fits->readTableArray(_row, _peakCol, peakElementCount, &peakElements.front());
            std::vector<float>::iterator j = peakElements.begin();
            while (j != peakElements.end()) {
                float x = *j++;
                float y = *j++;
                float value = *j++;
                fp->getPeaks().push_back(boost::make_shared<detection::Peak>(x, y, value));
            }
        }
        record->setFootprint(fp);

        if (
            !(_flags & SOURCE_IO_NO_HEAVY_FOOTPRINTS)
            && heavyPixElementCount && heavyMaskElementCount && heavyVarElementCount
        ) {
            int N = fp->getArea();
            if ((heavyPixElementCount  != N) ||
                (heavyMaskElementCount != N) ||
                (heavyVarElementCount  != N)) {
                throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    afw::fits::makeErrorMessage(
                        _fits->fptr, _fits->status,
                        boost::format("Number of HeavyFootprint elements (pix %d, mask %d, var %d) must all be equal to footprint area (%d)")
                        % heavyPixElementCount % heavyMaskElementCount % heavyVarElementCount % N
                        ));
            }
            // float HeavyFootprints were the only kind we ever saved using the old format
            typedef detection::HeavyFootprint<float,image::MaskPixel,image::VariancePixel> HeavyFootprint;
            PTR(HeavyFootprint) heavy = boost::make_shared<HeavyFootprint>(*fp);
            _fits->readTableArray(_row, _heavyPixCol,  N, heavy->getImageArray().getData());
            _fits->readTableArray(_row, _heavyMaskCol, N, heavy->getMaskArray().getData());
            _fits->readTableArray(_row, _heavyVarCol,  N, heavy->getVarianceArray().getData());
            record->setFootprint(heavy);
        }
    }
    return record;
}

// registers the reader so FitsReader::make can use it.
static io::FitsReader::FactoryT<SourceFitsReader> sourceFitsReaderFactory("SOURCE");

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- SourceTable/Record member function implementations --------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

SourceRecord::SourceRecord(PTR(SourceTable) const & table) : SimpleRecord(table) {}

void SourceRecord::updateCoord(image::Wcs const & wcs) {
    set(SourceTable::getCoordKey(), *wcs.pixelToSky(getCentroid()));
}

void SourceRecord::updateCoord(image::Wcs const & wcs, Key< Point<double> > const & key) {
    set(SourceTable::getCoordKey(), *wcs.pixelToSky(get(key)));
}

void SourceRecord::_assign(BaseRecord const & other) {
    try {
        SourceRecord const & s = dynamic_cast<SourceRecord const &>(other);
        _footprint = s._footprint;
    } catch (std::bad_cast&) {}
}

PTR(SourceTable) SourceTable::make(Schema const & schema, PTR(IdFactory) const & idFactory) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError,
            "Schema for Source must contain at least the keys defined by getMinimalSchema()."
        );
    }
    return boost::make_shared<SourceTableImpl>(schema, idFactory);
}

SourceTable::SourceTable(
    Schema const & schema,
    PTR(IdFactory) const & idFactory
) : SimpleTable(schema, idFactory), _slots(schema) {}

SourceTable::SourceTable(SourceTable const & other) :
    SimpleTable(other), _slots(other._slots)
{}

void SourceTable::handleAliasChange(std::string const & alias) {
    if (alias.compare(0, 4, "slot") != 0) {
        return;
    }
    _slots.handleAliasChange(alias, getSchema());
}

SourceTable::MinimalSchema::MinimalSchema() {
    schema = SimpleTable::makeMinimalSchema();
    parent = schema.addField<RecordId>("parent", "unique ID of parent source");
    schema.getCitizen().markPersistent();
}

SourceTable::MinimalSchema & SourceTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

PTR(io::FitsWriter) SourceTable::makeFitsWriter(fits::Fits * fitsfile, int flags) const {
    return boost::make_shared<SourceFitsWriter>(fitsfile, flags);
}


template class CatalogT<SourceRecord>;
template class CatalogT<SourceRecord const>;

template class SortedCatalogT<SourceRecord>;
template class SortedCatalogT<SourceRecord const>;

}}} // namespace lsst::afw::table
