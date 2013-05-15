// -*- lsst-c++ -*-
#include <typeinfo>

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/Exposure.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/Calib.h"
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

    explicit ExposureRecordImpl(PTR(ExposureTable) const & table) : ExposureRecord(table) {
        // Want to make default bbox empty, not a single pixel at 0,0
        this->set(ExposureTable::getBBoxMaxKey(), geom::Point2I(-1,-1));
    }

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

    ExposureFitsWriter(Fits * fits, PTR(io::OutputArchive) archive)
        : io::FitsWriter(fits, archive)
    {}

protected:

    virtual void _writeTable(CONST_PTR(BaseTable) const & table, std::size_t nRows);

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
    _fits->writeKey("AFW_TYPE", "EXPOSURE", "Tells lsst::afw to load this as an Exposure table.");
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
        io::FitsReader(fits, archive)
    {}

protected:

    virtual PTR(BaseTable) _readTable();

};

PTR(BaseTable) ExposureFitsReader::_readTable() {
    PTR(daf::base::PropertyList) metadata = boost::make_shared<daf::base::PropertyList>();
    _fits->readMetadata(*metadata, true);
    Schema schema(*metadata, true);
    PTR(ExposureTable) table = ExposureTable::make(schema);
    if (metadata->exists("AFW_TYPE")) metadata->remove("AFW_TYPE");
    table->setMetadata(metadata);
    _startRecords(*table);
    return table;
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

CONST_PTR(image::Wcs) ExposureRecord::getWcs() const {
    return boost::static_pointer_cast<image::Wcs>(get(ExposureTable::getWcsKey()));
}
void ExposureRecord::setWcs(CONST_PTR(image::Wcs) wcs) {
    set(ExposureTable::getWcsKey(), boost::const_pointer_cast<image::Wcs>(wcs));
}

CONST_PTR(detection::Psf) ExposureRecord::getPsf() const {
    return boost::static_pointer_cast<detection::Psf>(get(ExposureTable::getPsfKey()));
}
void ExposureRecord::setPsf(CONST_PTR(detection::Psf) psf) {
    set(ExposureTable::getPsfKey(), boost::const_pointer_cast<detection::Psf>(psf));
}

CONST_PTR(image::Calib) ExposureRecord::getCalib() const {
    return boost::static_pointer_cast<image::Calib>(get(ExposureTable::getCalibKey()));
}
void ExposureRecord::setCalib(CONST_PTR(image::Calib) calib) {
    set(ExposureTable::getCalibKey(), boost::const_pointer_cast<image::Calib>(calib));
}

bool ExposureRecord::contains(Coord const & coord) const {
    if (!getWcs()) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            "ExposureRecord does not have a Wcs; cannot call contains()"
        );
    }
    try {
        geom::Point2D point = getWcs()->skyToPixel(coord);
        return geom::Box2D(getBBox()).contains(point);
    } catch (pex::exceptions::DomainErrorException &) {
        // Wcs can throw if the given coordinate is outside the region
        // where the Wcs is valid.
        return false;
    }
}

bool ExposureRecord::contains(geom::Point2D const & point, image::Wcs const & wcs) const {
    return contains(*wcs.pixelToSky(point));
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
    wcs = schema.addField<PTR(io::Persistable)>("wcs", "world coordinate system");
    psf = schema.addField<PTR(io::Persistable)>("psf", "point spread function");
    calib = schema.addField<PTR(io::Persistable)>("calib", "photometric calibration");
    schema.getCitizen().markPersistent();
}

ExposureTable::MinimalSchema & ExposureTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

PTR(io::FitsWriter)
ExposureTable::makeFitsWriter(fits::Fits * fitsfile, PTR(io::OutputArchive) archive) const {
    return boost::make_shared<ExposureFitsWriter>(fitsfile, archive);
}

//-----------------------------------------------------------------------------------------------------------
//----- ExposureCatalogT member function implementations ----------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

template <typename RecordT>
ExposureCatalogT<RecordT>
ExposureCatalogT<RecordT>::subsetContaining(Coord const & coord) const {
    ExposureCatalogT result(this->getTable());
    for (const_iterator i = this->begin(); i != this->end(); ++i) {
        if (i->contains(coord)) result.push_back(i);
    }
    return result;
}

template <typename RecordT>
ExposureCatalogT<RecordT>
ExposureCatalogT<RecordT>::subsetContaining(geom::Point2D const & point, image::Wcs const & wcs) const {
    ExposureCatalogT result(this->getTable());
    for (const_iterator i = this->begin(); i != this->end(); ++i) {
        if (i->contains(point, wcs)) result.push_back(i);
    }
    return result;
}

//-----------------------------------------------------------------------------------------------------------
//----- Explicit instantiation ------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

template class CatalogT<ExposureRecord>;
template class CatalogT<ExposureRecord const>;

template class SortedCatalogT<ExposureRecord>;
template class SortedCatalogT<ExposureRecord const>;

template class ExposureCatalogT<ExposureRecord>;
template class ExposureCatalogT<ExposureRecord const>;

}}} // namespace lsst::afw::table
