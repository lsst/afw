// -*- lsst-c++ -*-
#include <typeinfo>

#include "boost/iterator/transform_iterator.hpp"

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/detail/Access.h"

// Some boilerplate macros for saving/loading Source slot aliases to/from FITS headers.
// Didn't seem to be quite enough to give the file the full M4 treatment.

#define SAVE_MEAS_SLOT(NAME, Name, TYPE, Type)                          \
    if (table->get ## Name ## Type ## Key().isValid()) {                \
        std::string s = table->getSchema().find(table->get ## Name ## Type ## Key()).field.getName(); \
        std::replace(s.begin(), s.end(), '.', '_');                     \
        _fits->writeKey(#NAME #TYPE "_SLOT", s.c_str(), "Defines the " #Name #Type " slot"); \
    }                                                                   \
    if (table->get ## Name ## Type ## ErrKey().isValid()) {             \
        std::string s = table->getSchema().find(table->get ## Name ## Type ## ErrKey()).field.getName(); \
        std::replace(s.begin(), s.end(), '.', '_');                     \
        _fits->writeKey(#NAME #TYPE "_ERR_SLOT", s.c_str(),         \
                        "Defines the " #Name #Type "Err slot");         \
    }                                                                   \
    if (table->get ## Name ## Type ## Flag ## Key().isValid()) {        \
        std::string s = table->getSchema().find(table->get ## Name ## Type ## FlagKey()).field.getName(); \
        std::replace(s.begin(), s.end(), '.', '_');                     \
        _fits->writeKey(#NAME #TYPE "_FLAG_SLOT", s.c_str(),        \
                        "Defines the " #Name #Type "Flag slot");        \
    }

#define SAVE_FLUX_SLOT(NAME, Name) SAVE_MEAS_SLOT(NAME ## _, Name, FLUX, Flux)
#define SAVE_CENTROID_SLOT() SAVE_MEAS_SLOT(, , CENTROID, Centroid)
#define SAVE_SHAPE_SLOT() SAVE_MEAS_SLOT(, , SHAPE, Shape)

#define LOAD_MEAS_SLOT(NAME, Name, TYPE, Type)                          \
    {                                                                   \
        _fits->alwaysCheck = false;                                     \
        std::string s, sErr, sFlag;                                     \
        _fits->readKey(#NAME #TYPE "_SLOT", s);                         \
        _fits->readKey(#NAME #TYPE "_ERR_SLOT", sErr);                  \
        _fits->readKey(#NAME #TYPE "_FLAG_SLOT", sFlag);                \
        if (_fits->status == 0) {                                       \
            metadata->remove(#NAME #TYPE "_SLOT");                      \
            metadata->remove(#NAME #TYPE "_ERR_SLOT");                  \
            metadata->remove(#NAME #TYPE "_FLAG_SLOT");                 \
            std::replace(s.begin(), s.end(), '_', '.');                 \
            std::replace(sErr.begin(), sErr.end(), '_', '.');           \
            std::replace(sFlag.begin(), sFlag.end(), '_', '.');         \
            table->define ## Name ## Type(schema[s], schema[sErr], schema[sFlag]); \
        } else {                                                        \
            _fits->status = 0;                                          \
        }                                                               \
        _fits->alwaysCheck = true;                                      \
    }
    

#define LOAD_FLUX_SLOT(NAME, Name) LOAD_MEAS_SLOT(NAME ## _, Name, FLUX, Flux)
#define LOAD_CENTROID_SLOT() LOAD_MEAS_SLOT(, , CENTROID, Centroid)
#define LOAD_SHAPE_SLOT() LOAD_MEAS_SLOT(, , SHAPE, Shape)

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

    explicit SourceTableImpl(
        Schema const & schema,
        PTR(IdFactory) const & idFactory
    ) : SourceTable(schema, idFactory) {}

    SourceTableImpl(SourceTableImpl const & other) : SourceTable(other) {}

private:

    virtual PTR(BaseTable) _clone() const {
        return boost::make_shared<SourceTableImpl>(*this);
    }

    virtual PTR(BaseRecord) _makeRecord() {
        PTR(SourceRecord) record = boost::make_shared<SourceRecordImpl>(getSelf<SourceTableImpl>());
        record->setId((*getIdFactory())());
        return record;
    }

};

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- SourceFitsWriter ------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsWriter for Sources - this saves footprints as variable-length arrays, and adds header
// keys that define the slots.  It also sets the AFW_TYPE key to SOURCE, which should ensure we use
// SourceFitsReader to read it.

// The only public access point to this class is SourceTable::makeFitsWriter.  If we subclass SourceTable
// someday, it may be necessary to put SourceFitsWriter in a header file so we can subclass it too.

namespace {

class SourceFitsWriter : public io::FitsWriter {
public:

    explicit SourceFitsWriter(Fits * fits) : io::FitsWriter(fits) {}

protected:
    
    virtual void _writeTable(CONST_PTR(BaseTable) const & table);

    virtual void _writeRecord(BaseRecord const & record);

private:
    int _spanCol;
    int _peakCol;
};

void SourceFitsWriter::_writeTable(CONST_PTR(BaseTable) const & t) {
    CONST_PTR(SourceTable) table = boost::dynamic_pointer_cast<SourceTable const>(t);
    if (!table) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Cannot use a SourceFitsWriter on a non-Source table."
        );
    }
    io::FitsWriter::_writeTable(table);
    _spanCol = _fits->addColumn<int>("spans", 0, "footprint spans (y, x0, x1)");
    _peakCol = _fits->addColumn<float>("peaks", 0, "footprint peaks (fx, fy, peakValue)");
    _fits->writeKey("SPANCOL", _spanCol + 1, "Column with footprint spans.");
    _fits->writeKey("PEAKCOL", _peakCol + 1, "Column with footprint peaks (float values).");
    _fits->writeKey("AFW_TYPE", "SOURCE", "Tells lsst::afw to load this as a Source table.");
    SAVE_FLUX_SLOT(PSF, Psf);
    SAVE_FLUX_SLOT(MODEL, Model);
    SAVE_FLUX_SLOT(AP, Ap);
    SAVE_FLUX_SLOT(INST, Inst);
    SAVE_CENTROID_SLOT();
    SAVE_SHAPE_SLOT();
}

void SourceFitsWriter::_writeRecord(BaseRecord const & r) {
    SourceRecord const & record = static_cast<SourceRecord const &>(r);
    io::FitsWriter::_writeRecord(record);
    if (record.getFootprint()) {
        Footprint::SpanList const & spans = record.getFootprint()->getSpans();
        Footprint::PeakList const & peaks = record.getFootprint()->getPeaks();
        if (!spans.empty()) {
            std::vector<int> vec;
            vec.reserve(3 * spans.size());
            for (Footprint::SpanList::const_iterator j = spans.begin(); j != spans.end(); ++j) {
                vec.push_back((**j).getY());
                vec.push_back((**j).getX0());
                vec.push_back((**j).getX1());
            }
            _fits->writeTableArray(_row, _spanCol, vec.size(), &vec.front());
        }
        if (!peaks.empty()) {
            std::vector<float> vec;
            vec.reserve(3 * peaks.size());
            for (Footprint::PeakList::const_iterator j = peaks.begin(); j != peaks.end(); ++j) {
                vec.push_back((**j).getFx());
                vec.push_back((**j).getFy());
                vec.push_back((**j).getPeakValue());}
            _fits->writeTableArray(_row, _peakCol, vec.size(), &vec.front());
        }
    }
}

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- SourceFitsReader ------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsReader for Sources - this reads footprints as variable-length arrays, and adds header
// keys that define the slots.  It gets registered with name SOURCE, so it should get used whenever
// we read a table with AFW_TYPE set to that value.

// The only public access point to this class is through the registry.  If we subclass SourceTable
// someday, it may be necessary to put SourceFitsReader in a header file so we can subclass it too.

namespace {

class SourceFitsReader : public io::FitsReader {
public:

    explicit SourceFitsReader(Fits * fits) : io::FitsReader(fits), _spanCol(-1), _peakCol(-1) {}

protected:

    virtual PTR(BaseTable) _readTable();

    virtual PTR(BaseRecord) _readRecord(PTR(BaseTable) const & table);

private:
    int _spanCol;
    int _peakCol;
};

PTR(BaseTable) SourceFitsReader::_readTable() {
    PTR(daf::base::PropertyList) metadata = boost::make_shared<daf::base::PropertyList>();
    _fits->readMetadata(*metadata, true);
    _spanCol = metadata->get("SPANCOL", 0);
    if (_spanCol > 0) { 
        // we remove these from the metadata so the Schema constructor doesn't try to parse
        // the footprint columns
        metadata->remove("SPANCOL");
        metadata->remove((boost::format("TTYPE%d") % _spanCol).str());
        metadata->remove((boost::format("TFORM%d") % _spanCol).str());
    }
    _peakCol = metadata->get("PEAKCOL", 0);
    if (_peakCol >= 0) {
        metadata->remove("PEAKCOL");
        metadata->remove((boost::format("TTYPE%d") % _peakCol).str());
        metadata->remove((boost::format("TFORM%d") % _peakCol).str());
    }
    if (metadata->exists("AFW_TYPE")) metadata->remove("AFW_TYPE");
    --_spanCol; // switch to 0-indexed rather than 1-indexed convention.
    --_peakCol;
    Schema schema(*metadata, true);
    PTR(SourceTable) table =  SourceTable::make(schema);
    LOAD_FLUX_SLOT(PSF, Psf);
    LOAD_FLUX_SLOT(MODEL, Model);
    LOAD_FLUX_SLOT(AP, Ap);
    LOAD_FLUX_SLOT(INST, Inst);
    LOAD_CENTROID_SLOT();
    LOAD_SHAPE_SLOT();
    _startRecords();
    table->setMetadata(metadata);
    return table;
}

PTR(BaseRecord) SourceFitsReader::_readRecord(PTR(BaseTable) const & table) {
    PTR(SourceRecord) record = boost::static_pointer_cast<SourceRecord>(io::FitsReader::_readRecord(table));
    if (!record) return record;
    boost::static_pointer_cast<SourceTable>(table)->getIdFactory()->notify(record->getId());
    int spanElementCount = (_spanCol >= 0) ? _fits->getTableArraySize(_row, _spanCol) : 0;
    int peakElementCount = (_peakCol >= 0) ? _fits->getTableArraySize(_row, _peakCol) : 0;
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
    }
    return record;
}

// registers the reader so FitsReader::make can use it.
static io::FitsReader::FactoryT<SourceFitsReader> sourceFitsReaderFactory("SOURCE");

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- SourceTable/Record member function implementations --------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

SourceRecord::SourceRecord(PTR(SourceTable) const & table) : BaseRecord(table) {}

void SourceRecord::_assign(BaseRecord const & other) {
    try {
        SourceRecord const & s = dynamic_cast<SourceRecord const &>(other);
        _footprint = s._footprint;
    } catch (std::bad_cast&) {}
}

PTR(SourceTable) SourceTable::make(
    Schema const & schema,
    PTR(IdFactory) const & idFactory
) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            "Schema for Source must contain at least the keys defined by getMinimalSchema()."
        );
    }
    return boost::make_shared<SourceTableImpl>(schema, idFactory);
}

SourceTable::SourceTable(
    Schema const & schema,
    PTR(IdFactory) const & idFactory
) : BaseTable(schema), _idFactory(idFactory)
{
    if (!_idFactory) _idFactory = IdFactory::makeSimple();
}

SourceTable::SourceTable(SourceTable const & other) :
    BaseTable(other),
    _idFactory(other._idFactory->clone()),
    _slotFlux(other._slotFlux), _slotCentroid(other._slotCentroid), _slotShape(other._slotShape)
{}

SourceTable::MinimalSchema::MinimalSchema() {
    detail::Access::markPersistent(schema);
    id = schema.addField<RecordId>("id", "unique ID for source");
    parent = schema.addField<RecordId>("parent", "unique ID of parent source");
    coord = schema.addField<Coord>("coord", "position of source in ra/dec", "radians");
}

SourceTable::MinimalSchema & SourceTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

PTR(io::FitsWriter) SourceTable::makeFitsWriter(io::FitsWriter::Fits * fits) const {
    return boost::make_shared<SourceFitsWriter>(fits);
}

//-----------------------------------------------------------------------------------------------------------
//----- Convenience functions for adding common measurements to Schemas -------------------------------------
//-----------------------------------------------------------------------------------------------------------

KeyTuple<Centroid> addCentroidFields(
    Schema & schema,
    std::string const & name,
    std::string const & doc
) {
    KeyTuple<Centroid> keys;
    keys.meas = schema.addField<Centroid::MeasTag>(name, doc, "pixels");
    keys.err = schema.addField<Centroid::ErrTag>(
        name + ".err", "covariance matrix for " + name, "pixels^2"
    );
    keys.flag = schema.addField<Flag>(
        name + ".flags", "set if the " + name + " measurement did not fully succeed"
    );
    return keys;
}

KeyTuple<Shape> addShapeFields(
    Schema & schema,
    std::string const & name,
    std::string const & doc
) {
    KeyTuple<Shape> keys;
    keys.meas = schema.addField<Shape::MeasTag>(
        name, doc, "pixels^2"
    );
    keys.err = schema.addField<Shape::ErrTag>(
        name + ".err", "covariance matrix for " + name, "pixels^4"
    );
    keys.flag = schema.addField<Flag>(
        name + ".flags", "set if the " + name + " measurement failed"
    );
    return keys;
}

KeyTuple<Flux> addFluxFields(
    Schema & schema,
    std::string const & name,
    std::string const & doc
) {
    KeyTuple<Flux> keys;
    keys.meas = schema.addField<Flux::MeasTag>(
        name, doc, "dn"
    );
    keys.err = schema.addField<Flux::ErrTag>(
        name + ".err", "uncertainty for " + name, "dn"
    );
    keys.flag = schema.addField<Flag>(
        name + ".flags", "set if the " + name + " measurement failed"
    );
    return keys;
}

template class VectorT<SourceRecord>;
template class VectorT<SourceRecord const>;
template class SourceVectorT<SourceRecord>;
template class SourceVectorT<SourceRecord const>;

}}} // namespace lsst::afw::table
