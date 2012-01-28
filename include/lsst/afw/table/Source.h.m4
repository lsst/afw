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
    $2::ErrValue get$1$2$3() const;
')dnl
m4def(`DECLARE_FLUX_GETTERS', `DECLARE_SLOT_GETTERS($1, `Flux', `Err')')dnl
m4def(`DECLARE_CENTROID_GETTERS', `DECLARE_SLOT_GETTERS(`', `Centroid', `Cov')')dnl
m4def(`DECLARE_SHAPE_GETTERS', `DECLARE_SLOT_GETTERS(`', `Shape', `Cov')')dnl
m4def(`DEFINE_SLOT_GETTERS',
`inline $2::MeasValue SourceRecord::get$1$2() const {
    return this->get(getTable()->get$1$2Key());
}

inline $2::ErrValue SourceRecord::get$1$2$3() const {
    return this->get(getTable()->get$1$2$3Key());
}
')dnl
m4def(`DEFINE_FLUX_GETTERS', `DEFINE_SLOT_GETTERS($1, `Flux', `Err')')dnl
m4def(`DEFINE_CENTROID_GETTERS', `DEFINE_SLOT_GETTERS(`', `Centroid', `Cov')')dnl
m4def(`DEFINE_SHAPE_GETTERS', `DEFINE_SLOT_GETTERS(`', `Shape', `Cov')')dnl
m4def(`DECLARE_SLOT_DEFINERS',
`/**
     * @brief Set the measurement used for the $1$2 slot using Keys.
     */
    void define$1$2($2::MeasKey const & meas, $2::ErrKey const & err) {
        _slot$2$4 = KeyPair<$2>(meas, err);
    }

    /**
     *  @brief Set the measurement used for the $1$2 slot with a field name.
     *
     *  This requires that the measurement adhere to the convention of having
     *  "<name>" and "<name>.translit($3, `A-Z', `a-z')" fields.
     */
    void define$1$2(std::string const & name) {
        Schema schema = getSchema();
        _slot$2$4 = KeyPair<$2>(schema[name], schema[name]["translit($3, `A-Z', `a-z')"]);
    }

    /// @brief Return the name of the field used for the $1$2 slot.
    std::string get$1$2Definition() const {
        return getSchema().find(_slot$2$4.meas).field.getName();
    }

    /// @brief Return the key used for the $1$2 slot.
    $2::MeasKey get$1$2Key() const { return _slot$2$4.meas; }

    /// @brief Return the key used for $1$2 slot error or covariance.
    $2::ErrKey get$1$2$3Key() const { return _slot$2$4.err; }
')dnl
m4def(`DECLARE_FLUX_DEFINERS', `DECLARE_SLOT_DEFINERS($1, `Flux', `Err', `[FLUX_SLOT_`'translit($1, `a-z', `A-Z')]')')dnl
m4def(`DECLARE_CENTROID_DEFINERS', `DECLARE_SLOT_DEFINERS(`', `Centroid', `Cov', `')')dnl
m4def(`DECLARE_SHAPE_DEFINERS', `DECLARE_SLOT_DEFINERS(`', `Shape', `Cov', `')')dnl
#ifndef AFW_TABLE_Source_h_INCLUDED
#define AFW_TABLE_Source_h_INCLUDED

#include "boost/array.hpp"
#include "boost/type_traits/is_convertible.hpp"

#include "lsst/daf/base/PropertyList.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/IdFactory.h"
#include "lsst/afw/table/Set.h"
#include "lsst/afw/table/Vector.h"
#include "lsst/afw/table/io/FitsWriter.h"

namespace lsst { namespace afw { namespace table {

typedef lsst::afw::detection::Footprint Footprint;

class SourceRecord;
class SourceTable;

template <typename RecordT=SourceRecord, typename TableT=typename RecordT::Table> class SourceSetT;

template <typename MeasTagT, typename ErrTagT>
struct Measurement {
    typedef MeasTagT MeasTag;
    typedef ErrTagT ErrTag;
    typedef typename Field<MeasTag>::Value MeasValue;
    typedef typename Field<ErrTag>::Value ErrValue;
    typedef Key<MeasTag> MeasKey;
    typedef Key<ErrTag> ErrKey;
};

#ifndef SWIG

struct Flux : public Measurement<double,double> {};
struct Centroid : public Measurement< Point<double>, Covariance< Point<double> > > {};
struct Shape : public Measurement< Moments<double>, Covariance< Moments<double> > > {};

enum FluxSlotEnum {
    FLUX_SLOT_PSF=0,
    FLUX_SLOT_MODEL,
    FLUX_SLOT_AP,
    FLUX_SLOT_INST,
    N_FLUX_SLOTS
};

template <typename MeasurementT>
struct KeyPair {
    typename MeasurementT::MeasKey meas;
    typename MeasurementT::ErrKey err;

    KeyPair() {}

    KeyPair(
        typename MeasurementT::MeasKey const & meas_,
        typename MeasurementT::ErrKey const & err_
    ) : meas(meas_), err(err_) {}
};

#endif // !SWIG

/**
 *  @brief Record class that contains measurements made on a single exposure.
 */
class SourceRecord : public BaseRecord {
public:

    typedef SourceTable Table;
    typedef VectorT<SourceRecord,Table> Vector;
    typedef VectorT<SourceRecord const,Table> ConstVector;
    typedef SourceSetT<SourceRecord,Table> Set;
    typedef SourceSetT<SourceRecord const,Table> ConstSet;

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

    float getSky() const;
    void setSky(float v);

    float getSkyErr() const;
    void setSkyErr(float v);

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

protected:

    SourceRecord(PTR(SourceTable) const & table);

    virtual void _assign(BaseRecord const & other);

private:
    PTR(Footprint) _footprint;
};

/**
 *  @brief Table class that contains measurements made on a single exposure.
 */
class SourceTable : public BaseTable {
public:

    typedef SourceRecord Record;
    typedef VectorT<Record,SourceTable> Vector;
    typedef VectorT<Record const,SourceTable> ConstVector;
    typedef SourceSetT<Record,SourceTable> Set;
    typedef SourceSetT<Record const,SourceTable> ConstSet;

    /**
     *  @brief Construct a new table.
     *
     *  @param[in] schema            Schema that defines the fields, offsets, and record size for the table.
     *  @param[in] metadata          Flexible metadata for the table.  An empty PropertyList will be used
     *                               if an empty pointer is passed.
     *  @param[in] idFactory         Factory class to generate record IDs when they are not explicitly given.
     *                               If empty, defaults to a simple counter that starts at 1.
     */
    static PTR(SourceTable) make(
        Schema const & schema,
        PTR(daf::base::PropertyList) const & metadata = PTR(daf::base::PropertyList)(),
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
     *  from other static member functions of the Source tag class.
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

    /// @brief Key for the sky background at the location of the source.
    static Key<float> getSkyKey() { return getMinimalSchema().sky; }

    /// @brief Key for the sky background uncertainty at the location of the source.
    static Key<float> getSkyErrKey() { return getMinimalSchema().skyErr; }

    /// @brief Key for the ra/dec of the source.
    static Key<Coord> getCoordKey() { return getMinimalSchema().coord; }

    //@}

    /// @brief Return the object that generates IDs for the table.
    PTR(IdFactory) getIdFactory() { return _idFactory; }

    /// @brief Return the object that generates IDs for the table.
    CONST_PTR(IdFactory) getIdFactory() const { return _idFactory; }

    /// @brief Return the flexible metadata associated with the source table.
    PTR(daf::base::PropertyList) getMetadata() const { return _metadata; }

    /// @brief Set the flexible metadata associated with the source table.
    void setMetadata(PTR(daf::base::PropertyList) const & metadata) { _metadata = metadata; }

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
        PTR(daf::base::PropertyList) const & metadata,
        PTR(IdFactory) const & idFactory
    );

    SourceTable(SourceTable const & other);

private:

    struct MinimalSchema {
        Schema schema;
        Key<RecordId> id;
        Key<RecordId> parent;
        Key<float> sky;
        Key<float> skyErr;
        Key<Coord> coord;

        MinimalSchema();
    };
    
    static MinimalSchema & getMinimalSchema();

    friend class io::FitsWriter;

    virtual PTR(io::FitsWriter) makeFitsWriter(io::FitsWriter::Fits * fits) const;

    PTR(daf::base::PropertyList) _metadata;
    PTR(IdFactory) _idFactory;
    boost::array< KeyPair<Flux>, N_FLUX_SLOTS > _slotFlux;
    KeyPair<Centroid> _slotCentroid;
    KeyPair<Shape> _slotShape;
};

#ifndef SWIG

template <typename RecordT, typename TableT>
class SourceSetT : public SetT<RecordT,TableT> {
    BOOST_STATIC_ASSERT( (boost::is_convertible<RecordT*,SourceRecord const*>::value) );
public:
    
    explicit SourceSetT(PTR(TableT) const & table = PTR(TableT)()) :
       SetT<RecordT,TableT>(table, SourceTable::getIdKey()) {}

    explicit SourceSetT(Schema const & schema) : SetT<RecordT,TableT>(schema, SourceTable::getIdKey()) {}

    template <typename InputIterator>
    SourceSetT(PTR(TableT) const & table, InputIterator first, InputIterator last, bool deep=false) :
        SetT<RecordT,TableT>(table, SourceTable::getIdKey(), first, last, deep)
    {}

    template <typename OtherRecordT, typename OtherTableT>
    explicit SourceSetT(VectorT<OtherRecordT,OtherTableT> const & other) : 
        SetT<RecordT,TableT>(other.getTable(), SourceTable::getIdKey(), other.begin(), other.end(), false)
    {}

    static SourceSetT readFits(std::string const & filename) {
        return io::FitsReader::apply<SourceSetT>(filename);
    }

};

typedef SourceSetT<SourceRecord,SourceTable> SourceSet;
typedef SourceSetT<SourceRecord const,SourceTable> ConstSourceSet;

typedef VectorT<SourceRecord,SourceTable> SourceVector;
typedef VectorT<SourceRecord const,SourceTable> ConstSourceVector;

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

inline float SourceRecord::getSky() const { return get(SourceTable::getSkyKey()); }
inline void SourceRecord::setSky(float v) { set(SourceTable::getSkyKey(), v); }

inline float SourceRecord::getSkyErr() const { return get(SourceTable::getSkyErrKey()); }
inline void SourceRecord::setSkyErr(float v) { set(SourceTable::getSkyErrKey(), v); }

inline IcrsCoord SourceRecord::getCoord() const { return get(SourceTable::getCoordKey()); }
inline void SourceRecord::setCoord(IcrsCoord const & coord) { set(SourceTable::getCoordKey(), coord); }
inline void SourceRecord::setCoord(Coord const & coord) { set(SourceTable::getCoordKey(), coord); }

inline Angle SourceRecord::getRa() const { return get(SourceTable::getCoordKey().getRa()); }
inline Angle SourceRecord::getDec() const { return get(SourceTable::getCoordKey().getDec()); }

#endif // !SWIG

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Source_h_INCLUDED
