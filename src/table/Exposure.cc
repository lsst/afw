// -*- lsst-c++ -*-
#include <typeinfo>

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/Exposure.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/detection/Psf.h"

namespace lsst { namespace afw { namespace table {

//-----------------------------------------------------------------------------------------------------------
//----- Private ExposureTable/Record classes ---------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// These private derived classes are what you actually get when you do ExposureTable::make; like the
// private classes in BaseTable.cc, it's more convenient to have an extra set of trivial derived
// classes than to do a lot of friending.

namespace {

class ExposureTableImpl;

class ExposureRecordImpl : public ExposureRecord {
public:

    explicit ExposureRecordImpl(PTR(ExposureTable) const & table) : ExposureRecord(table) {}

};

class ExposureTableImpl : public ExposureTable {
public:

    explicit ExposureTableImpl(Schema const & schema) : 
        ExposureTable(schema)
    {}

    ExposureTableImpl(ExposureTableImpl const & other) : ExposureTable(other) {}

private:

    virtual PTR(BaseTable) _clone() const {
        return boost::make_shared<ExposureTableImpl>(*this);
    }

    virtual PTR(BaseRecord) _makeRecord() {
        return boost::make_shared<ExposureRecordImpl>(getSelf<ExposureTableImpl>());
    }

};

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- ExposureFitsWriter ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsWriter for Exposure - this just sets the AFW_TYPE key to EXPOSURE, which should ensure
// we use ExposureFitsReader to read it.

namespace {

class ExposureFitsWriter : public io::FitsWriter {
public:

    ExposureFitsWriter(Fits * fits, PTR(io::OutputArchive) archive = PTR(io::OutputArchive)())
        : io::FitsWriter(fits), _doWriteArchive(false), _archive(archive), _psfCol(-1), _wcsCol(-1)
    {
        if (!_archive) {
            _doWriteArchive = true;
            _archive.reset(new io::OutputArchive());
        }
    }

protected:
    
    virtual void _writeTable(CONST_PTR(BaseTable) const & table, std::size_t nRows);

    virtual void _writeRecord(BaseRecord const & r);

    virtual void _finish() {
        if (_doWriteArchive) _archive->writeFits(*_fits);
    }

    bool _doWriteArchive;
    PTR(io::OutputArchive) _archive;
    int _psfCol;
    int _wcsCol;
};

void ExposureFitsWriter::_writeTable(CONST_PTR(BaseTable) const & t, std::size_t nRows) {
    CONST_PTR(ExposureTable) table = boost::dynamic_pointer_cast<ExposureTable const>(t);
    if (!table) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Cannot use a ExposureFitsWriter on a non-Exposure table."
        );
    }
    io::FitsWriter::_writeTable(table, nRows);
    _psfCol = _fits->addColumn<int>("psf", 1, "archive ID of PSF");
    _wcsCol = _fits->addColumn<int>("wcs", 1, "archive ID of WCS");
    _fits->writeKey("PSF_COL", _psfCol + 1, "Column with archive ID of PSF");
    _fits->writeKey("WCS_COL", _wcsCol + 1, "Column with archive ID of WCS");
    _fits->writeKey("AFW_TYPE", "EXPOSURE", "Tells lsst::afw to load this as an Exposure table.");
}

void ExposureFitsWriter::_writeRecord(BaseRecord const & r) {
    ExposureRecord const & record = static_cast<ExposureRecord const &>(r);
    io::FitsWriter::_writeRecord(record);
    int psfId = _archive->put(record.getPsf());
    int wcsId = _archive->put(record.getWcs());
    _fits->writeTableScalar(_row, _psfCol, psfId);
    _fits->writeTableScalar(_row, _wcsCol, wcsId);
    
}

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- ExposureFitsReader ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsReader for ExposureTable/Record - this gets registered with name EXPOSURE, so it should get
// used whenever we read a table with AFW_TYPE set to that value.

namespace {

class ExposureFitsReader : public io::FitsReader {
public:

    explicit ExposureFitsReader(Fits * fits, PTR(io::InputArchive) archive) :
        io::FitsReader(fits, archive), _archive(archive)
    {
        if (!_archive) {
            int oldHdu = _fits->getHdu();
            _fits->setHdu(oldHdu + 1);
            _archive.reset(new io::InputArchive(io::InputArchive::readFits(*_fits)));
            _fits->setHdu(oldHdu);
        }
    }

protected:

    virtual PTR(BaseTable) _readTable();

    virtual PTR(BaseRecord) _readRecord(PTR(BaseTable) const & table);

    int _psfCol;
    int _wcsCol;
    PTR(io::InputArchive) _archive;
};

PTR(BaseTable) ExposureFitsReader::_readTable() {
    PTR(daf::base::PropertyList) metadata = boost::make_shared<daf::base::PropertyList>();
    _fits->readMetadata(*metadata, true);
    _psfCol = metadata->get("PSF_COL", 0);
    if (_psfCol >= 0) {
        metadata->remove("PSF_COL");
        metadata->remove((boost::format("TTYPE%d") % _psfCol).str());
        metadata->remove((boost::format("TFORM%d") % _psfCol).str());
    }
    _wcsCol = metadata->get("WCS_COL", 0);
    if (_wcsCol >= 0) {
        metadata->remove("WCS_COL");
        metadata->remove((boost::format("TTYPE%d") % _wcsCol).str());
        metadata->remove((boost::format("TFORM%d") % _wcsCol).str());
    }
    Schema schema(*metadata, true);
    PTR(ExposureTable) table =  ExposureTable::make(schema);
    _startRecords(*table);
    if (metadata->exists("AFW_TYPE")) metadata->remove("AFW_TYPE");
    table->setMetadata(metadata);
    return table;
}

PTR(BaseRecord) ExposureFitsReader::_readRecord(PTR(BaseTable) const & table) {
    PTR(ExposureRecord) record = boost::static_pointer_cast<ExposureRecord>(
        io::FitsReader::_readRecord(table)
    );
    if (!record) return record;
    if (_psfCol >= 0) {
        int psfId = 0;
        _fits->readTableScalar(_row, _psfCol, psfId);
        record->setPsf(_archive->get<detection::Psf>(psfId));
    }
    if (_wcsCol >= 0) {
        int wcsId = 0;
        _fits->readTableScalar(_row, _wcsCol, wcsId);
        record->setWcs(_archive->get<image::Wcs>(wcsId));
    }
    return record;
}

// registers the reader so FitsReader::make can use it.
static io::FitsReader::FactoryT<ExposureFitsReader> referenceFitsReaderFactory("EXPOSURE");

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- ExposureTable/Record member function implementations -----------------------------------------------
//-----------------------------------------------------------------------------------------------------------

geom::Box2I ExposureRecord::getBBox() const {
    return geom::Box2I(get(ExposureTable::getBBoxMinKey()), get(ExposureTable::getBBoxMaxKey()));
}

void ExposureRecord::setBBox(geom::Box2I const & bbox) {
    set(ExposureTable::getBBoxMinKey(), bbox.getMin());
    set(ExposureTable::getBBoxMaxKey(), bbox.getMax());
}

bool ExposureRecord::contains(Coord const & coord) const {
    if (!getWcs()) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            "ExposureRecord does not have a Wcs; cannot call contains()"
        );
    }
    geom::Point2D point = getWcs()->skyToPixel(coord);
    return geom::Box2D(getBBox()).contains(point);
}

bool ExposureRecord::contains(geom::Point2D const & point, Wcs const & wcs) const {
    return contains(*wcs.pixelToSky(point));
}

bool ExposureRecord::overlaps(geom::Box2D const & box, Wcs const & wcs) const {
    if (!getWcs()) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            "ExposureRecord does not have a Wcs; cannot call overlaps()"
        );
    }
    geom::Box2D bbox(getBBox());
    return bbox.contains(getWcs()->skyToPixel(*wcs.pixelToSky(box.getMin())))
        || bbox.contains(getWcs()->skyToPixel(*wcs.pixelToSky(box.getMax())))
        || bbox.contains(getWcs()->skyToPixel(*wcs.pixelToSky(box.getMinX(), box.getMaxY())))
        || bbox.contains(getWcs()->skyToPixel(*wcs.pixelToSky(box.getMaxX(), box.getMinY())));
}

ExposureRecord::ExposureRecord(PTR(ExposureTable) const & table) : BaseRecord(table) {}

PTR(ExposureTable) ExposureTable::make(Schema const & schema) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            "Schema for Exposure must contain at least the keys defined by makeMinimalSchema()."
        );
    }
    return boost::make_shared<ExposureTableImpl>(schema);
}

ExposureTable::ExposureTable(Schema const & schema) :
    BaseTable(schema) {}

ExposureTable::ExposureTable(ExposureTable const & other) :
    BaseTable(other) {}

ExposureTable::MinimalSchema::MinimalSchema() {
    id = schema.addField<RecordId>("id", "unique ID");
    bboxMin = schema.addField< Point<int> >("bbox.min", "bbox minimum point", "pixels");
    bboxMax = schema.addField< Point<int> >("bbox.max", "bbox maximum point", "pixels");
    schema.getCitizen().markPersistent();
}

ExposureTable::MinimalSchema & ExposureTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

PTR(io::FitsWriter)
ExposureTable::makeFitsWriter(io::FitsWriter::Fits * fits) const {
    return boost::make_shared<ExposureFitsWriter>(fits);
}

PTR(io::FitsWriter)
ExposureTable::makeFitsWriter(io::FitsWriter::Fits * fits, PTR(io::OutputArchive) archive) const {
    return boost::make_shared<ExposureFitsWriter>(fits, archive);
}

template class CatalogT<ExposureRecord>;
template class CatalogT<ExposureRecord const>;

template class SortedCatalogT<ExposureRecord>;
template class SortedCatalogT<ExposureRecord const>;

template class ExposureCatalogT<ExposureRecord>;
template class ExposureCatalogT<ExposureRecord const>;

}}} // namespace lsst::afw::table
