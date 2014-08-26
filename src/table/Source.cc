// -*- lsst-c++ -*-
#include <typeinfo>

#include "boost/iterator/transform_iterator.hpp"

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/detection/FootprintCtrl.h"
#include "lsst/afw/detection/HeavyFootprint.h"

// We ASSUME, for FITS persistence:
typedef float HeavyFootprintPixelT;

// Some boilerplate macros for saving/loading Source slot aliases to/from FITS headers.
// Didn't seem to be quite enough to give the file the full M4 treatment.

#define SAVE_MEAS_SLOT(NAME, Name, TYPE, Type)                              \
    if (table->getVersion() == 0) {                                         \
        if (table->has ## Name ## Type ## Slot()) {                \
            std::string s = table->get ## Name ## Type ## Definition(); \
            std::replace(s.begin(), s.end(), '.', '_');                     \
            _fits->writeKey(#NAME #TYPE "_SLOT", s.c_str(), "Defines the " #Name #Type " slot"); \
        }                                                                   \
        if (table->has ## Name ## Type ## Slot()) {             \
            std::string s = table->get ## Name ## Type ## Definition() + ".err"; \
            std::replace(s.begin(), s.end(), '.', '_');                     \
            _fits->writeKey(#NAME #TYPE "_ERR_SLOT", s.c_str(),             \
                            "Defines the " #Name #Type "Err slot");         \
        }                                                                   \
        if (table->get ## Name ## Type ## Flag ## Key().isValid()) {        \
            std::string s = table->get ## Name ## Type ## Definition() + ".flags"; \
            std::replace(s.begin(), s.end(), '.', '_');                     \
            _fits->writeKey(#NAME #TYPE "_FLAG_SLOT", s.c_str(),            \
                            "Defines the " #Name #Type "Flag slot");        \
        }                                                                   \
    }                                                                       \
    else {                                                                  \
        std::string s = table->get ## Name ## Type ## Definition();         \
        if (s.size() > 0) {                                                 \
            _fits->writeKey(#NAME #TYPE "_SLOT", s.c_str(),                 \
                            "Defines the " #Name #Type " slot");            \
        }                                                                   \
    }

#define SAVE_FLUX_SLOT(NAME, Name) SAVE_MEAS_SLOT(NAME ## _, Name, FLUX, Flux)
#define SAVE_CENTROID_SLOT() SAVE_MEAS_SLOT(, , CENTROID, Centroid)
#define SAVE_SHAPE_SLOT() SAVE_MEAS_SLOT(, , SHAPE, Shape)

#define LOAD_MEAS_SLOT(NAME, Name, TYPE, Type)                          \
    {                                                                   \
        if (table->getVersion() == 0) {                                     \
            _fits->behavior &= ~fits::Fits::AUTO_CHECK;                     \
            std::string s, sErr, sFlag;                                     \
            _fits->readKey(#NAME #TYPE "_SLOT", s);                         \
            std::replace(s.begin(), s.end(), '_', '.');                 \
            if (_fits->status == 0) {                                       \
                metadata->remove(#NAME #TYPE "_SLOT");                      \
                metadata->remove(#NAME #TYPE "_ERR_SLOT");                  \
                metadata->remove(#NAME #TYPE "_FLAG_SLOT");                 \
                table->define ## Name ## Type(s); \
            } else {                                                        \
                _fits->status = 0;                                          \
            }                                                               \
            _fits->behavior |= fits::Fits::AUTO_CHECK;                      \
        }                                                                   \
        else {                                                              \
            _fits->behavior &= ~fits::Fits::AUTO_CHECK;                     \
            std::string s;                                                  \
            _fits->readKey(#NAME #TYPE "_SLOT", s);                         \
            if (_fits->status == 0) {                                       \
                metadata->remove(#NAME #TYPE "_SLOT");                      \
                table->define ## Name ## Type(s);                           \
            } else {                                                        \
                _fits->status = 0;                                          \
            }                                                               \
            _fits->behavior |= fits::Fits::AUTO_CHECK;                      \
        }                                                                   \
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

    explicit SourceFitsWriter(Fits * fits, int flags) :
        io::FitsWriter(fits, flags),
        _heavyPixCol(-1),
        _heavyMaskCol(-1),
        _heavyVarCol(-1)
    {}

protected:
    
    virtual void _writeTable(CONST_PTR(BaseTable) const & table, std::size_t nRows);

    virtual void _writeRecord(BaseRecord const & record);

private:
    int _spanCol;
    int _peakCol;
    int _heavyPixCol;
    int _heavyMaskCol;
    int _heavyVarCol;
};

void SourceFitsWriter::_writeTable(CONST_PTR(BaseTable) const & t, std::size_t nRows) {
    CONST_PTR(SourceTable) table = boost::dynamic_pointer_cast<SourceTable const>(t);
    if (!table) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicError,
            "Cannot use a SourceFitsWriter on a non-Source table."
        );
    }
    io::FitsWriter::_writeTable(table, nRows);
    if (!(_flags & SOURCE_IO_NO_FOOTPRINTS)) {
        _spanCol = _fits->addColumn<int>("spans", 0, "footprint spans (y, x0, x1)");
        _peakCol = _fits->addColumn<float>("peaks", 0, "footprint peaks (fx, fy, peakValue)");
        _fits->writeKey("SPANCOL", _spanCol + 1, "Column with footprint spans.");
        _fits->writeKey("PEAKCOL", _peakCol + 1, "Column with footprint peaks (float values).");
        if (!(_flags & SOURCE_IO_NO_HEAVY_FOOTPRINTS)) {
            _heavyPixCol  = _fits->addColumn<HeavyFootprintPixelT>("heavyPix", 0, "HeavyFootprint pixels");
            _heavyMaskCol = _fits->addColumn<image::MaskPixel>("heavyMask", 0, "HeavyFootprint masks");
            _heavyVarCol  = _fits->addColumn<image::VariancePixel>("heavyVar", 0, "HeavyFootprint variance");
            _fits->writeKey("HVYPIXCO", _heavyPixCol  + 1, "Column with HeavyFootprint pix");
            _fits->writeKey("HVYMSKCO", _heavyMaskCol + 1, "Column with HeavyFootprint mask");
            _fits->writeKey("HVYVARCO", _heavyVarCol  + 1, "Column with HeavyFootprint variance");
        }
    }
    _fits->writeKey("AFW_TYPE", "SOURCE", "Tells lsst::afw to load this as a Source table.");
    SAVE_FLUX_SLOT(PSF, Psf);
    SAVE_FLUX_SLOT(MODEL, Model);
    SAVE_FLUX_SLOT(AP, Ap);
    SAVE_FLUX_SLOT(INST, Inst);
    SAVE_CENTROID_SLOT();
    SAVE_SHAPE_SLOT();
}

void SourceFitsWriter::_writeRecord(BaseRecord const & r) {
    typedef detection::HeavyFootprint<HeavyFootprintPixelT,image::MaskPixel,image::VariancePixel>
        HeavyFootprint;
    SourceRecord const & record = static_cast<SourceRecord const &>(r);
    io::FitsWriter::_writeRecord(record);
    if (record.getFootprint() && !(_flags & SOURCE_IO_NO_FOOTPRINTS)) {
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
        if (!(_flags & SOURCE_IO_NO_HEAVY_FOOTPRINTS) && record.getFootprint()->isHeavy()) {
            assert((_heavyPixCol >= 0) && (_heavyMaskCol >= 0) && (_heavyVarCol >= 0));
            PTR(HeavyFootprint) heavy = boost::static_pointer_cast<HeavyFootprint>(record.getFootprint());
            int N = heavy->getArea();
            _fits->writeTableArray(_row, _heavyPixCol,  N, heavy->getImageArray().getData());
            _fits->writeTableArray(_row, _heavyMaskCol, N, heavy->getMaskArray().getData());
            _fits->writeTableArray(_row, _heavyVarCol,  N, heavy->getVarianceArray().getData());
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

    explicit SourceFitsReader(Fits * fits, PTR(io::InputArchive) archive, int flags) :
        io::FitsReader(fits, archive, flags), _spanCol(-1), _peakCol(-1),
        _heavyPixCol(-1), _heavyMaskCol(-1), _heavyVarCol(-1)
        {}

protected:

    virtual PTR(BaseTable) _readTable();

    virtual PTR(BaseRecord) _readRecord(PTR(BaseTable) const & table);

private:
    int _spanCol;
    int _peakCol;
    int _heavyPixCol;
    int _heavyMaskCol;
    int _heavyVarCol;
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
    Schema schema(*metadata, true);
    PTR(SourceTable) table =  SourceTable::make(schema, PTR(IdFactory)());
    table->setMetadata(metadata);
    _startRecords(*table);
    // None of the code below depends on _startRecords?
    LOAD_FLUX_SLOT(PSF, Psf);
    LOAD_FLUX_SLOT(MODEL, Model);
    LOAD_FLUX_SLOT(AP, Ap);
    LOAD_FLUX_SLOT(INST, Inst);
    LOAD_CENTROID_SLOT();
    LOAD_SHAPE_SLOT();
    return table;
}

PTR(BaseRecord) SourceFitsReader::_readRecord(PTR(BaseTable) const & table) {
    typedef detection::HeavyFootprint<HeavyFootprintPixelT,image::MaskPixel,image::VariancePixel>
        HeavyFootprint;
    PTR(SourceRecord) record = boost::static_pointer_cast<SourceRecord>(io::FitsReader::_readRecord(table));
    if (!record) return record;
    if (_flags & SOURCE_IO_NO_FOOTPRINTS) return record;
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

// some helpers for centroid/slot defines
namespace {

// goes away entirely in C++11
std::vector<std::string> makePointNames() {
    std::vector<std::string> v;
    v.push_back("x");
    v.push_back("y");
    return v;
}

std::vector<std::string> const & getPointNames() {
    static std::vector<std::string> v = makePointNames();
    return v;
}

// goes away entirely in C++11
std::vector<std::string> makeQuadrupoleNames() {
    std::vector<std::string> v;
    v.push_back("xx");
    v.push_back("yy");
    v.push_back("xy");
    return v;
}

std::vector<std::string> const & getQuadrupoleNames() {
    static std::vector<std::string> v = makeQuadrupoleNames();
    return v;
}

} // anonymous

void SourceTable::defineCentroid(std::string const & name) {
    Schema schema = getSchema();
    _slotCentroid.name = name;
    SubSchema sub = schema[name];
    if (getVersion() == 0) { // this block will be retired someday
        Centroid::MeasKey measKey = sub;
        _slotCentroid.pos = lsst::afw::table::Point2DKey(measKey);
        try {
            Centroid::ErrKey errKey = sub["err"];
            _slotCentroid.posErr = lsst::afw::table::CovarianceMatrixKey<float,2>(errKey);
        } catch (pex::exceptions::NotFoundError) {}
        try {
            _slotCentroid.flag = sub["flags"];
        } catch (pex::exceptions::NotFoundError) {}
        return;
    }
    _slotCentroid.pos = lsst::afw::table::Point2DKey(sub);
    try {
        _slotCentroid.posErr = CovarianceMatrixKey<float,2>(sub, getPointNames());
    } catch (pex::exceptions::NotFoundError) {}
    try {
        _slotCentroid.flag = sub["flag"];
    } catch (pex::exceptions::NotFoundError) {}
}

void SourceTable::defineShape(std::string const & name) {
    Schema schema = getSchema();
    _slotShape.name = name;
    SubSchema sub = schema[name];
    if (getVersion() == 0) { // this block will be retired someday
        Shape::MeasKey measKey = sub;
        _slotShape.quadrupole = lsst::afw::table::QuadrupoleKey(
            measKey.getIxx(), measKey.getIyy(), measKey.getIxy()
        );
        try {
            Shape::ErrKey errKey = sub["err"];
            _slotShape.quadrupoleErr = lsst::afw::table::CovarianceMatrixKey<float,3>(errKey);
        } catch (pex::exceptions::NotFoundError) {}
        try {
            _slotShape.flag = sub["flags"];
        } catch (pex::exceptions::NotFoundError) {}
        return;
    }
    _slotShape.quadrupole = lsst::afw::table::QuadrupoleKey(sub);
    try {
        _slotShape.quadrupoleErr = CovarianceMatrixKey<float,3>(sub, getQuadrupoleNames());
    } catch (pex::exceptions::NotFoundError) {}
    try {
        _slotShape.flag = sub["flag"];
    } catch (pex::exceptions::NotFoundError) {}
}

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
) : SimpleTable(schema, idFactory) {}

SourceTable::SourceTable(SourceTable const & other) :
    SimpleTable(other),
    _slotFlux(other._slotFlux), _slotCentroid(other._slotCentroid), _slotShape(other._slotShape)
{}

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

template class CatalogT<SourceRecord>;
template class CatalogT<SourceRecord const>;

template class SortedCatalogT<SourceRecord>;
template class SortedCatalogT<SourceRecord const>;

}}} // namespace lsst::afw::table
