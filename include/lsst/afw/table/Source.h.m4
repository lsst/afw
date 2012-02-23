changecom(`###')dnl
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
define(`m4def', defn(`define'))dnl
m4def(`DECLARE_SLOT_GETTERS',
`/// @brief Get the value of the $1$2 slot measurement.
    $2::MeasValue get$1$2() const;

    /// @brief Get the uncertainty on the $1$2 slot measurement.
    $2::ErrValue get$1$2Err() const;

    /// @brief Return true if the measurement in the $1$2 slot was successful.
    bool get$1$2Flag() const;
')dnl
m4def(`DECLARE_FLUX_GETTERS', `DECLARE_SLOT_GETTERS($1, `Flux')')dnl
m4def(`DECLARE_CENTROID_GETTERS', `DECLARE_SLOT_GETTERS(`', `Centroid')')dnl
m4def(`DECLARE_SHAPE_GETTERS', `DECLARE_SLOT_GETTERS(`', `Shape')')dnl
m4def(`DEFINE_SLOT_GETTERS',
`inline $2::MeasValue SourceRecord::get$1$2() const {
    return this->get(getTable()->get$1$2Key());
}

inline $2::ErrValue SourceRecord::get$1$2Err() const {
    return this->get(getTable()->get$1$2ErrKey());
}

inline bool SourceRecord::get$1$2Flag() const {
    return this->get(getTable()->get$1$2FlagKey());
}
')dnl
m4def(`DEFINE_FLUX_GETTERS', `DEFINE_SLOT_GETTERS($1, `Flux')')dnl
m4def(`DEFINE_CENTROID_GETTERS', `DEFINE_SLOT_GETTERS(`', `Centroid')')dnl
m4def(`DEFINE_SHAPE_GETTERS', `DEFINE_SLOT_GETTERS(`', `Shape')')dnl
m4def(`DECLARE_SLOT_DEFINERS',
`/**
     * @brief Set the measurement used for the $1$2 slot using Keys.
     */
    void define$1$2($2::MeasKey const & meas, $2::ErrKey const & err, Key<Flag> const & flag) {
        _slot$2$3 = KeyTuple<$2>(meas, err, flag);
    }

    /**
     *  @brief Set the measurement used for the $1$2 slot with a field name.
     *
     *  This requires that the measurement adhere to the convention of having
     *  "<name>", "<name>.err", and "<name>.flags" fields.
     */
    void define$1$2(std::string const & name) {
        Schema schema = getSchema();
        _slot$2$3 = KeyTuple<$2>(schema[name], schema[name]["err"], schema[name]["flags"]);
    }

    /// @brief Return the name of the field used for the $1$2 slot.
    std::string get$1$2Definition() const {
        return getSchema().find(_slot$2$3.meas).field.getName();
    }

    /// @brief Return the key used for the $1$2 slot.
    $2::MeasKey get$1$2Key() const { return _slot$2$3.meas; }

    /// @brief Return the key used for $1$2 slot error or covariance.
    $2::ErrKey get$1$2ErrKey() const { return _slot$2$3.err; }

    /// @brief Return the key used for the $1$2 slot success flag.
    Key<Flag> get$1$2FlagKey() const { return _slot$2$3.flag; }
')dnl
m4def(`DECLARE_FLUX_DEFINERS', `DECLARE_SLOT_DEFINERS($1, `Flux', `[FLUX_SLOT_`'translit($1, `a-z', `A-Z')]')')dnl
m4def(`DECLARE_CENTROID_DEFINERS', `DECLARE_SLOT_DEFINERS(`', `Centroid', `')')dnl
m4def(`DECLARE_SHAPE_DEFINERS', `DECLARE_SLOT_DEFINERS(`', `Shape', `')')dnl
#ifndef AFW_TABLE_Source_h_INCLUDED
#define AFW_TABLE_Source_h_INCLUDED

#include "boost/array.hpp"
#include "boost/type_traits/is_convertible.hpp"

#include "lsst/daf/base/PropertyList.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/IdFactory.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/io/FitsWriter.h"

namespace lsst { namespace afw { namespace table {

typedef lsst::afw::detection::Footprint Footprint;

class SourceRecord;
class SourceTable;

template <typename RecordT> class SourceCatalogT;

/// @brief A collection of types that correspond to common measurements.
template <typename MeasTagT, typename ErrTagT>
struct Measurement {
    typedef MeasTagT MeasTag;  ///< the tag (template parameter) type used for the measurement
    typedef ErrTagT ErrTag;    ///< the tag (template parameter) type used for the uncertainty
    typedef typename Field<MeasTag>::Value MeasValue; ///< the value type used for the measurement
    typedef typename Field<ErrTag>::Value ErrValue;   ///< the value type used for the uncertainty
    typedef Key<MeasTag> MeasKey;  ///< the Key type for the actual measurement
    typedef Key<ErrTag> ErrKey;    ///< the Key type for the error on the measurement
};

#ifndef SWIG

/// A collection of types useful for flux measurement algorithms.
struct Flux : public Measurement<double,double> {};

/// A collection of types useful for centroid measurement algorithms.
struct Centroid : public Measurement< Point<double>, Covariance< Point<double> > > {};

/// A collection of types useful for shape measurement algorithms.
struct Shape : public Measurement< Moments<double>, Covariance< Moments<double> > > {};

/// An enum for all the special flux aliases in Source.
enum FluxSlotEnum {
    FLUX_SLOT_PSF=0,
    FLUX_SLOT_MODEL,
    FLUX_SLOT_AP,
    FLUX_SLOT_INST,
    N_FLUX_SLOTS
};
/**
 *  @brief A three-element tuple of measurement, uncertainty, and flag keys.
 *
 *  Most measurement should have more than one flag key to indicate different kinds of failures.
 *  This flag key should usually be set to be a logical OR of all of them, so it is set whenever
 *  a measurement cannot be fully trusted.
 */
template <typename MeasurementT>
struct KeyTuple {
    typename MeasurementT::MeasKey meas; ///< Key used for the measured value.
    typename MeasurementT::ErrKey err;   ///< Key used for the uncertainty.
    Key<Flag> flag;                      ///< Failure bit; set if the measurement did not fully succeed.

    /// Default-constructor; all keys will be invalid.
    KeyTuple() {}

    /// Main constructor.
    KeyTuple(
        typename MeasurementT::MeasKey const & meas_,
        typename MeasurementT::ErrKey const & err_,
        Key<Flag> const & flag_
    ) : meas(meas_), err(err_), flag(flag_) {}

};

/// Convenience function to setup fields for centroid measurement algorithms.
KeyTuple<Centroid> addCentroidFields(Schema & schema, std::string const & name, std::string const & doc);

/// Convenience function to setup fields for shape measurement algorithms.
KeyTuple<Shape> addShapeFields(Schema & schema, std::string const & name, std::string const & doc);

/// Convenience function to setup fields for flux measurement algorithms.
KeyTuple<Flux> addFluxFields(Schema & schema, std::string const & name, std::string const & doc);

#endif // !SWIG

/**
 *  @brief Record class that contains measurements made on a single exposure.
 *
 *  Sources provide four additions to BaseRecord/BaseRecord:
 *   - Specific fields that must always be present, with specialized getters on source (e.g. getId()).
 *     The schema for a SourceTable should always be constructed by starting with the result of
 *     SourceTable::makeMinimalSchema.
 *   - A shared_ptr to a Footprint for each record.
 *   - A system of aliases (called slots) in which a SourceTable instance stores keys for particular
 *     measurements (a centroid, a shape, and a number of different fluxes) and SourceRecord uses
 *     this keys to provide custom getters and setters.  These are not separate fields, but rather
 *     aliases that can point to custom fields.
 *   - SourceTables hold an ID factory, which is used to initialize the unique ID field when a 
 *     new SourceRecord is created.
 */
class SourceRecord : public BaseRecord {
public:

    typedef SourceTable Table;
    typedef SourceCatalogT<SourceRecord> Catalog;
    typedef SourceCatalogT<SourceRecord const> ConstCatalog;

    PTR(Footprint) getFootprint() const { return _footprint; }

    void setFootprint(PTR(Footprint) const & footprint) { _footprint = footprint; }

    CONST_PTR(SourceTable) getTable() const {
        return boost::static_pointer_cast<SourceTable const>(BaseRecord::getTable());
    }

    //@{
    /// @brief Convenience accessors for the keys in the minimal source schema.
    RecordId getId() const;
    void setId(RecordId id);

    RecordId getParent() const;
    void setParent(RecordId id);

    IcrsCoord getCoord() const;
    void setCoord(IcrsCoord const & coord);
    void setCoord(Coord const & coord);
    //@}

    /// @brief Equivalent to getCoord().getRa() (but possibly faster if only ra is needed).
    Angle getRa() const;

    /// @brief Equivalent to getCoord().getDec() (but possibly faster if only dec is needed).
    Angle getDec() const;

    DECLARE_FLUX_GETTERS(`Psf')
    DECLARE_FLUX_GETTERS(`Model')
    DECLARE_FLUX_GETTERS(`Ap')
    DECLARE_FLUX_GETTERS(`Inst')
    DECLARE_CENTROID_GETTERS
    DECLARE_SHAPE_GETTERS

    /// @brief Return the centroid slot x coordinate.
    double getX() const;

    /// @brief Return the centroid slot y coordinate.
    double getY() const;

    /// @brief Return the shape slot Ixx value.
    double getIxx() const;

    /// @brief Return the shape slot Iyy value.
    double getIyy() const;

    /// @brief Return the shape slot Ixy value.
    double getIxy() const;

protected:

    SourceRecord(PTR(SourceTable) const & table);

    virtual void _assign(BaseRecord const & other);

private:
    PTR(Footprint) _footprint;
};

/**
 *  @brief Table class that contains measurements made on a single exposure.
 *
 *  @copydetails SourceRecord
 */
class SourceTable : public BaseTable {
public:

    typedef SourceRecord Record;
    typedef SourceCatalogT<Record> Catalog;
    typedef SourceCatalogT<Record const> ConstCatalog;

    /**
     *  @brief Construct a new table.
     *
     *  @param[in] schema            Schema that defines the fields, offsets, and record size for the table.
     *  @param[in] idFactory         Factory class to generate record IDs when they are not explicitly given.
     *                               If empty, defaults to a simple counter that starts at 1.
     */
    static PTR(SourceTable) make(
        Schema const & schema,
        PTR(IdFactory) const & idFactory = PTR(IdFactory)()
    );

    /**
     *  @brief Return a minimal schema for Source tables and records.
     *
     *  The returned schema can and generally should be modified further,
     *  but many operations on sources will assume that at least the fields
     *  provided by this routine are present.
     *
     *  Keys for the standard fields added by this routine can be obtained
     *  from other static member functions of the SourceTable class.
     */
    static Schema makeMinimalSchema() { return getMinimalSchema().schema; }

    /**
     *  @brief Return true if the given schema is a valid SourceTable schema.
     *  
     *  This will always be true if the given schema was originally constructed
     *  using makeMinimalSchema(), and will rarely be true otherwise.
     */
    static bool checkSchema(Schema const & other) {
        return other.contains(getMinimalSchema().schema);
    }

    //@{
    /**
     *  Get keys for standard fields shared by all sources.
     *
     *  These keys are used to implement getters and setters on SourceRecord.
     */

    /// @brief Key for the source ID.
    static Key<RecordId> getIdKey() { return getMinimalSchema().id; }

    /// @brief Key for the parent ID.
    static Key<RecordId> getParentKey() { return getMinimalSchema().parent; }

    /// @brief Key for the ra/dec of the source.
    static Key<Coord> getCoordKey() { return getMinimalSchema().coord; }

    //@}

    /// @brief Return the object that generates IDs for the table.
    PTR(IdFactory) getIdFactory() { return _idFactory; }

    /// @brief Return the object that generates IDs for the table.
    CONST_PTR(IdFactory) getIdFactory() const { return _idFactory; }

    /// @copydoc BaseTable::clone
    PTR(SourceTable) clone() const { return boost::static_pointer_cast<SourceTable>(_clone()); }

    /// @copydoc BaseTable::makeRecord
    PTR(SourceRecord) makeRecord() { return boost::static_pointer_cast<SourceRecord>(_makeRecord()); }

    /// @copydoc BaseTable::copyRecord
    PTR(SourceRecord) copyRecord(BaseRecord const & other) {
        return boost::static_pointer_cast<SourceRecord>(BaseTable::copyRecord(other));
    }

    /// @copydoc BaseTable::copyRecord
    PTR(SourceRecord) copyRecord(BaseRecord const & other, SchemaMapper const & mapper) {
        return boost::static_pointer_cast<SourceRecord>(BaseTable::copyRecord(other, mapper));
    }

    DECLARE_FLUX_DEFINERS(`Psf')
    DECLARE_FLUX_DEFINERS(`Model')
    DECLARE_FLUX_DEFINERS(`Ap')
    DECLARE_FLUX_DEFINERS(`Inst')
    DECLARE_CENTROID_DEFINERS
    DECLARE_SHAPE_DEFINERS
protected:

    SourceTable(
        Schema const & schema,
        PTR(IdFactory) const & idFactory
    );

    SourceTable(SourceTable const & other);

private:

    // Struct that holds the minimal schema and the special keys we've added to it.
    struct MinimalSchema {
        Schema schema;
        Key<RecordId> id;
        Key<RecordId> parent;
        Key<Coord> coord;

        MinimalSchema();
    };
    
    // Return the singleton minimal schema.
    static MinimalSchema & getMinimalSchema();

    friend class io::FitsWriter;

     // Return a writer object that knows how to save in FITS format.  See also FitsWriter.
    virtual PTR(io::FitsWriter) makeFitsWriter(io::FitsWriter::Fits * fits) const;

    PTR(IdFactory) _idFactory;        // generates IDs for new records
    boost::array< KeyTuple<Flux>, N_FLUX_SLOTS > _slotFlux; // aliases for flux measurements
    KeyTuple<Centroid> _slotCentroid;  // alias for a centroid measurement
    KeyTuple<Shape> _slotShape;  // alias for a shape measurement
};

#ifndef SWIG

template <typename RecordT>
class SourceCatalogT : public CatalogT<RecordT> {
    typedef CatalogT<RecordT> Base;
public:

    typedef RecordT Record;
    typedef typename Record::Table Table;

    typedef typename Base::iterator iterator;
    typedef typename Base::const_iterator const_iterator;

    using Base::isSorted;
    using Base::sort;
    using Base::find;

    /// @brief Return true if the vector is in ascending ID order.
    bool isSorted() const { return this->isSorted(SourceTable::getIdKey()); }

    /// @brief Sort the vector in-place by ID.
    void sort() { this->sort(SourceTable::getIdKey()); }

    //@{
    /**
     *  @brief Return an iterator to the record with the given ID.
     *
     *  @note The vector must be sorted in ascending ID order before calling find (i.e. 
     *        isSorted() must be true).
     *
     *  Returns end() if the Record cannot be found.
     */
    iterator find(RecordId id) { return this->find(id, SourceTable::getIdKey()); }
    const_iterator find(RecordId id) const { return this->find(id, SourceTable::getIdKey()); }
    //@}

    /**
     *  @brief Construct a vector from a table (or nothing).
     *
     *  A vector with no table is considered invalid; a valid table must be assigned to it
     *  before it can be used.
     */
    explicit SourceCatalogT(PTR(Table) const & table = PTR(Table)()) : Base(table) {}

    /// @brief Construct a vector from a schema, creating a table with Table::make(schema).
    explicit SourceCatalogT(Schema const & schema) : Base(schema) {}

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
    SourceCatalogT(PTR(Table) const & table, InputIterator first, InputIterator last, bool deep=false) :
        Base(table, first, last, deep)
    {}

    /**
     *  @brief Shallow copy constructor from a container containing a related record type.
     *
     *  This conversion only succeeds if OtherRecordT is convertible to RecordT and OtherTable is
     *  convertible to Table.
     */
    template <typename OtherRecordT>
    SourceCatalogT(SourceCatalogT<OtherRecordT> const & other) : Base(other) {}

    /// Read a FITS binary table.
    static SourceCatalogT readFits(std::string const & filename) {
        return io::FitsReader::apply<SourceCatalogT>(filename);
    }

};


typedef SourceCatalogT<SourceRecord> SourceCatalog;
typedef SourceCatalogT<SourceRecord const> ConstSourceCatalog;

DEFINE_FLUX_GETTERS(`Psf')
DEFINE_FLUX_GETTERS(`Model')
DEFINE_FLUX_GETTERS(`Ap')
DEFINE_FLUX_GETTERS(`Inst')
DEFINE_CENTROID_GETTERS
DEFINE_SHAPE_GETTERS

inline RecordId SourceRecord::getId() const { return get(SourceTable::getIdKey()); }
inline void SourceRecord::setId(RecordId id) { set(SourceTable::getIdKey(), id); }

inline RecordId SourceRecord::getParent() const { return get(SourceTable::getParentKey()); }
inline void SourceRecord::setParent(RecordId id) { set(SourceTable::getParentKey(), id); }

inline IcrsCoord SourceRecord::getCoord() const { return get(SourceTable::getCoordKey()); }
inline void SourceRecord::setCoord(IcrsCoord const & coord) { set(SourceTable::getCoordKey(), coord); }
inline void SourceRecord::setCoord(Coord const & coord) { set(SourceTable::getCoordKey(), coord); }

inline Angle SourceRecord::getRa() const { return get(SourceTable::getCoordKey().getRa()); }
inline Angle SourceRecord::getDec() const { return get(SourceTable::getCoordKey().getDec()); }

inline double SourceRecord::getX() const { return get(getTable()->getCentroidKey().getX()); }
inline double SourceRecord::getY() const { return get(getTable()->getCentroidKey().getY()); }

inline double SourceRecord::getIxx() const { return get(getTable()->getShapeKey().getIxx()); }
inline double SourceRecord::getIyy() const { return get(getTable()->getShapeKey().getIyy()); }
inline double SourceRecord::getIxy() const { return get(getTable()->getShapeKey().getIxy()); }

#endif // !SWIG

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Source_h_INCLUDED
