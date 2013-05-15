// -*- lsst-c++ -*-
/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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
#ifndef AFW_TABLE_Exposure_h_INCLUDED
#define AFW_TABLE_Exposure_h_INCLUDED

#include "lsst/afw/geom/Box.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/SortedCatalog.h"
#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/io/FitsWriter.h"

namespace lsst { namespace afw {

namespace image {
class Wcs;
class Calib;
} // namespace image

namespace detection {
class Psf;
} // namespace detection

namespace table {

class ExposureRecord;
class ExposureTable;

template <typename RecordT> class ExposureCatalogT;

namespace io {

class OutputArchiveHandle;
class InputArchive;

} // namespace io

/**
 *  @brief Record class used to store exposure metadata.
 */
class ExposureRecord : public BaseRecord {
public:

    typedef ExposureTable Table;
    typedef ColumnViewT<ExposureRecord> ColumnView;
    typedef ExposureCatalogT<ExposureRecord> Catalog;
    typedef ExposureCatalogT<ExposureRecord const> ConstCatalog;

    CONST_PTR(ExposureTable) getTable() const {
        return boost::static_pointer_cast<ExposureTable const>(BaseRecord::getTable());
    }

    RecordId getId() const;
    void setId(RecordId id);

    geom::Box2I getBBox() const;
    void setBBox(geom::Box2I const & bbox);

    /**
     *  @brief Return true if the bounding box contains the given celestial coordinate point, taking
     *         into account the Wcs of the ExposureRecord.
     *
     *  @throw LogicErrorException if the ExposureRecord has no Wcs.
     */
    bool contains(Coord const & coord) const;

    /**
     *  @brief Return true if the bounding box contains the given point, taking into account its Wcs
     *         (given) and the Wcs of the ExposureRecord.
     *
     *  @throw LogicErrorException if the ExposureRecord has no Wcs.
     */
    bool contains(geom::Point2D const & point, image::Wcs const & wcs) const;

    //@{
    /// Get/Set the the attached Wcs, Psf, or Calib.  No copies are made.
    CONST_PTR(image::Wcs) getWcs() const { return _wcs; }
    void setWcs(CONST_PTR(image::Wcs) wcs) { _wcs = wcs; }

    CONST_PTR(detection::Psf) getPsf() const { return _psf; }
    void setPsf(CONST_PTR(detection::Psf) psf) { _psf = psf; }

    CONST_PTR(image::Calib) getCalib() const { return _calib; }
    void setCalib(CONST_PTR(image::Calib) calib) { _calib = calib; }
    //@}

protected:

    ExposureRecord(PTR(ExposureTable) const & table);

    virtual void _assign(BaseRecord const & other);

private:
    CONST_PTR(image::Wcs) _wcs;
    CONST_PTR(detection::Psf) _psf;
    CONST_PTR(image::Calib) _calib;
};

/**
 *  @brief Table class used to store exposure metadata.
 *
 *  @copydetails ExposureRecord
 */
class ExposureTable : public BaseTable {
public:

    typedef ExposureRecord Record;
    typedef ColumnViewT<ExposureRecord> ColumnView;
    typedef ExposureCatalogT<Record> Catalog;
    typedef ExposureCatalogT<Record const> ConstCatalog;

    /**
     *  @brief Construct a new table.
     *
     *  @param[in] schema            Schema that defines the fields, offsets, and record size for the table.
     */
    static PTR(ExposureTable) make(Schema const & schema);

    /**
     *  @brief Return a minimal schema for Exposure tables and records.
     *
     *  The returned schema can and generally should be modified further,
     *  but many operations on ExposureRecords will assume that at least the fields
     *  provided by this routine are present.
     */
    static Schema makeMinimalSchema() { return getMinimalSchema().schema; }

    /**
     *  @brief Return true if the given schema is a valid ExposureTable schema.
     *
     *  This will always be true if the given schema was originally constructed
     *  using makeMinimalSchema(), and will rarely be true otherwise.
     */
    static bool checkSchema(Schema const & other) {
        return other.contains(getMinimalSchema().schema);
    }

    //@{
    /**
     *  Get keys for standard fields shared by all references.
     *
     *  These keys are used to implement getters and setters on ExposureRecord.
     */
    /// @brief Key for the unique ID.
    static Key<RecordId> getIdKey() { return getMinimalSchema().id; }
    /// @brief Key for the minimum point of the bbox.
    static Key< Point<int> > getBBoxMinKey() { return getMinimalSchema().bboxMin; }
    /// @brief Key for the maximum point of the bbox.
    static Key< Point<int> > getBBoxMaxKey() { return getMinimalSchema().bboxMax; }
    //@}

    /// @copydoc BaseTable::clone
    PTR(ExposureTable) clone() const { return boost::static_pointer_cast<ExposureTable>(_clone()); }

    /// @copydoc BaseTable::makeRecord
    PTR(ExposureRecord) makeRecord() { return boost::static_pointer_cast<ExposureRecord>(_makeRecord()); }

    /// @copydoc BaseTable::copyRecord
    PTR(ExposureRecord) copyRecord(BaseRecord const & other) {
        return boost::static_pointer_cast<ExposureRecord>(BaseTable::copyRecord(other));
    }

    /// @copydoc BaseTable::copyRecord
    PTR(ExposureRecord) copyRecord(BaseRecord const & other, SchemaMapper const & mapper) {
        return boost::static_pointer_cast<ExposureRecord>(BaseTable::copyRecord(other, mapper));
    }

protected:

    ExposureTable(Schema const & schema);

    ExposureTable(ExposureTable const & other);

private:

    // Struct that holds the minimal schema and the special keys we've added to it.
    struct MinimalSchema {
        Schema schema;
        Key<RecordId> id;
        Key< Point<int> > bboxMin;
        Key< Point<int> > bboxMax;

        MinimalSchema();
    };

    // Return the singleton minimal schema.
    static MinimalSchema & getMinimalSchema();

    friend class io::FitsWriter;

    template <typename RecordT> friend class ExposureCatalogT;

     // Return a writer object that knows how to save in FITS format.  See also FitsWriter.
    virtual PTR(io::FitsWriter) makeFitsWriter(
        fits::Fits * fitsfile,
        PTR(io::OutputArchive) archive = PTR(io::OutputArchive)()
    ) const;

};

#ifndef SWIG

/**
 *  @brief Custom catalog class for ExposureRecord/Table.
 *
 *  We don't expect to subclass ExposureRecord/Table, so unlike other Catalogs we can (and do) define
 *  some ExposureCatalogT member functions in Exposure.cc where the explicit instantiation is done.
 */
template <typename RecordT>
class ExposureCatalogT : public SortedCatalogT<RecordT> {
    typedef SortedCatalogT<RecordT> Base;
public:

    typedef RecordT Record;
    typedef typename Record::Table Table;

    typedef typename Base::iterator iterator;
    typedef typename Base::const_iterator const_iterator;

    /**
     *  @brief Construct a vector from a table (or nothing).
     *
     *  A vector with no table is considered invalid; a valid table must be assigned to it
     *  before it can be used.
     */
    explicit ExposureCatalogT(PTR(Table) const & table = PTR(Table)()) : Base(table) {}

    /// @brief Construct a vector from a schema, creating a table with Table::make(schema).
    explicit ExposureCatalogT(Schema const & schema) : Base(schema) {}

    /**
     *  @brief Construct a vector from a table and an iterator range.
     *
     *  If deep is true, new records will be created using table->copyRecord before being inserted.
     *  If deep is false, records will be not be copied, but they must already be associated with
     *  the given table.  The table itself is never deep-copied.
     *
     *  The iterator must dereference to a record reference or const reference rather than a pointer,
     *  but should be implicitly convertible to a record pointer as well (see CatalogIterator).
     *
     *  If InputIterator models RandomAccessIterator (according to std::iterator_traits) and deep
     *  is true, table->preallocate will be used to ensure that the resulting records are
     *  contiguous in memory and can be used with ColumnView.  To ensure this is the case for
     *  other iterator types, the user must preallocate the table manually.
     */
    template <typename InputIterator>
    ExposureCatalogT(PTR(Table) const & table, InputIterator first, InputIterator last, bool deep=false) :
        Base(table, first, last, deep)
    {}

    /**
     *  @brief Shallow copy constructor from a container containing a related record type.
     *
     *  This conversion only succeeds if OtherRecordT is convertible to RecordT and OtherTable is
     *  convertible to Table.
     */
    template <typename OtherRecordT>
    ExposureCatalogT(ExposureCatalogT<OtherRecordT> const & other) : Base(other) {}

    /// @brief Read a FITS binary table.
    static ExposureCatalogT readFits(std::string const & filename, int hdu=0) {
        return io::FitsReader::apply<ExposureCatalogT>(filename, hdu);
    }
    /// @brief Read a FITS binary table.
    static ExposureCatalogT readFits(fits::MemFileManager & manager, int hdu=0) {
        return io::FitsReader::apply<ExposureCatalogT>(manager, hdu);
    }
    /// @brief Read a FITS binary table from a file object already at the correct extension.
    static ExposureCatalogT readFits(fits::Fits & fitsfile) {
        return io::FitsReader::apply<ExposureCatalogT>(fitsfile);
    }
    /**
     *  @brief Read a FITS binary table from a file object already at the correct extension.
     *
     *  This overload reads nested Persistables from the given archive instead of loading
     *  a new archive from the HDUs following the catalog.
     */
    static ExposureCatalogT readFits(fits::Fits & fitsfile, PTR(io::InputArchive) archive) {
        return io::FitsReader::apply<ExposureCatalogT>(fitsfile, archive);
    }

    /**
     *  @brief Convenience output function for Persistables that contain an ExposureCatalog.
     *
     *  Unlike writeFits, this saves main catalog to one of the tables within the archive,
     *  as part of a Persistable's set of catalogs, rather than saving it to a separate HDU
     *  not managed by the archive.
     */
    void writeToArchive(io::OutputArchiveHandle & handle, bool ignoreUnpersistable=true) const;

    /**
     *  @brief Convenience input function for Persistables that contain an ExposureCatalog.
     *
     *  Unlike the FITS read methods, this reader is not polymorphically aware - it always
     *  tries to create an ExposureTable rather than infer the type of table from the data.
     */
    static ExposureCatalogT readFromArchive(io::InputArchive const & archive, BaseCatalog const & catalog);

    /**
     * @brief Shallow copy a subset of another ExposureCatalog.  Mostly here for
     * use from python.
     */
    ExposureCatalogT subset(std::ptrdiff_t startd, std::ptrdiff_t stopd, std::ptrdiff_t step) const {
        return ExposureCatalogT(Base::subset(startd, stopd, step));
    }

    /**
     *  @brief Return a shallow subset of the catalog with only those records that contain the
     *         given point.
     *
     *  @sa ExposureRecord::contains
     */
    ExposureCatalogT subsetContaining(Coord const & coord) const;

    /**
     *  @brief Return a shallow subset of the catalog with only those records that contain the
     *         given point.
     *
     *  @sa ExposureRecord::contains
     */
    ExposureCatalogT subsetContaining(geom::Point2D const & point, image::Wcs const & wcs) const;

protected:
    explicit ExposureCatalogT(Base const & other) : Base(other) {}
};

typedef ColumnViewT<ExposureRecord> ExposureColumnView;
typedef ExposureCatalogT<ExposureRecord> ExposureCatalog;
typedef ExposureCatalogT<ExposureRecord const> ConstExposureCatalog;

inline RecordId ExposureRecord::getId() const { return get(ExposureTable::getIdKey()); }
inline void ExposureRecord::setId(RecordId id) { set(ExposureTable::getIdKey(), id); }

#endif // !SWIG

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Exposure_h_INCLUDED
