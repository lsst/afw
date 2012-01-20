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
changecom(`###')dnl
define(`m4def', defn(`define'))dnl
m4def(`ADD_SLOT_GETTERS',
`$2::MeasValue get$1$2() const {
        return get(getTableAux()->translit($2, `A-Z', `a-z')$4.meas);
    }
    $2::ErrValue get$1$2$3() const {
        return get(getTableAux()->translit($2, `A-Z', `a-z')$4.err);
    }')dnl
m4def(`ADD_FLUX_GETTERS', `ADD_SLOT_GETTERS($1, `Flux', `Err', `[FLUX_SLOT_`'translit($1, `a-z', `A-Z')]')')dnl
m4def(`ADD_CENTROID_GETTERS', `ADD_SLOT_GETTERS(`', `Centroid', `Cov', `')')dnl
m4def(`ADD_SHAPE_GETTERS', `ADD_SLOT_GETTERS(`', `Shape', `Cov', `')')dnl
m4def(`ADD_SLOT_DEFINERS',
`/**
     * @brief Set the measurement used for the $1$2 slot using Keys.
     */
    void define$1$2($2::MeasKey const & meas, $2::ErrKey const & err)  const {
        getAux()->translit($2, `A-Z', `a-z')$4 = KeyPair<$2>(meas, err);
    }

    /**
     *  @brief Set the measurement used for the $1$2 slot with a field name.
     *
     *  This requires that the measurement adhere to the convention of having
     *  "<name>" and "<name>.translit($3, `A-Z', `a-z')" fields.
     */
    void define$1$2(std::string const & name) const {
        Schema schema = getSchema();
        getAux()->translit($2, `A-Z', `a-z')$4 = KeyPair<$2>(schema[name], schema[name]["translit($3, `A-Z', `a-z')"]);
    }

    /// @brief Return the name of the field used for the $1$2 slot.
    std::string get$1$2Definition() const {
        return getSchema().find(getAux()->translit($2, `A-Z', `a-z')$4.meas).field.getName();
    }

    /**
     *  @brief Return the key used for the $1$2 slot.
     *
     *  If performance is critical it may be faster to get and cache this
     *  key locally rather than use the SourceRecord getters; looking up
     *  the key each time requires an additional function call that
     *  cannot be inlined.
     */
    $2::MeasKey get$1$2Key() const {
         return getAux()->translit($2, `A-Z', `a-z')$4.meas;
     }

    /**
     *  @brief Return the key used for $1$2 slot error or covariance.
     *
     *  If performance is critical it may be faster to get and cache this
     *  key locally rather than use the SourceRecord getters, as looking up
     *  the key each time requires an additional function call that
     *  cannot be inlined.
     */
    $2::ErrKey get$1$2$3() const {
        return getAux()->translit($2, `A-Z', `a-z')$4.err;
    }
')dnl
m4def(`ADD_FLUX_DEFINERS', `ADD_SLOT_DEFINERS($1, `Flux', `Err', `[FLUX_SLOT_`'translit($1, `a-z', `A-Z')]')')dnl
m4def(`ADD_CENTROID_DEFINERS', `ADD_SLOT_DEFINERS(`', `Centroid', `Cov', `')')dnl
m4def(`ADD_SHAPE_DEFINERS', `ADD_SLOT_DEFINERS(`', `Shape', `Cov', `')')dnl
#ifndef AFW_TABLE_Source_h_INCLUDED
#define AFW_TABLE_Source_h_INCLUDED
#include "lsst/daf/base/PropertyList.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/table/RecordInterface.h"
#include "lsst/afw/table/TableInterface.h"

namespace lsst { namespace afw { namespace table {

typedef detection::Footprint Footprint;

class SourceRecord;
class SourceTable;

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

/// @brief A tag class for SourceTable and SourceRecord to be used with the interface classes.
struct Source {
    typedef SourceRecord Record;
    typedef SourceTable Table;
    
#ifndef SWIG

    class RecordAux : public AuxBase {
    public:
        Footprint footprint;
        
        explicit RecordAux(Footprint const & fp) : footprint(fp) {}
    };

    class TableAux : public AuxBase {
    public:
        PTR(daf::base::PropertyList) metadata;

        KeyPair<Flux> flux[N_FLUX_SLOTS];
        KeyPair<Centroid> centroid;
        KeyPair<Shape> shape;

        TableAux(PTR(daf::base::PropertyList) const & metadata_) : metadata(metadata_) {}
    };

#endif

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
    static Schema makeBasicSchema() { return getBasicSchema().schema; }

    //@{
    /**
     *  Get keys for standard fields shared by all sources.
     *
     *  These keys are used to implement getters and setters on SourceRecord.
     *  If performance is critical, it may be faster to get and cache these
     *  keys locally rather than use the SourceRecord getters, as looking up
     *  the keys each time requires an additional function call that
     *  cannot be inlined.
     */
    /// @brief Key for the sky background at the location of the source.
    static Key<float> getSkyKey() { return getBasicSchema().sky; }
    /// @brief Key for the sky background uncertainty at the location of the source.
    static Key<float> getSkyErrKey() { return getBasicSchema().skyErr; }
    /// @brief Key for the ra/dec of the source.
    static Key< Point<double> > getCoordKey() { return getBasicSchema().coord; }
    //@}

private:
    
    struct BasicSchema {
        Schema schema;
        Key<float> sky;
        Key<float> skyErr;
        Key< Point<double> > coord;
    };
    
    static BasicSchema & getBasicSchema();

};

/**
 *  @brief Record class that contains measurements made on a single exposure.
 */
class SourceRecord : public RecordInterface<Source> {
public:

    bool hasFootprint() const { return getAux(); }

    Footprint const & getFootprint() const {
        return boost::static_pointer_cast<Source::RecordAux>(getAux())->footprint;
    }

    void setFootprint(Footprint const & footprint) const {
        assertBit(CAN_SET_FIELD);
        getAux() = boost::make_shared<Source::RecordAux>(footprint);
    }

    //@{
    /**
     *  @brief Return canonical measurements and errors defined by "slots" in the table.
     *
     *  When performance is critical, it may be faster to obtain the keys that correspond
     *  to these slots from the SourceTable and cache these locally; key lookup involves
     *  a function call that cannot be inlined.
     */
    ADD_FLUX_GETTERS(`Psf')
    ADD_FLUX_GETTERS(`Model')
    ADD_FLUX_GETTERS(`Ap')
    ADD_FLUX_GETTERS(`Inst')
    ADD_CENTROID_GETTERS
    ADD_SHAPE_GETTERS
    //@}

    /// @brief Construct a null record, which is unusable until a valid record is assigned to it.
    SourceRecord() : RecordInterface<Source>() {}
    
private:

    PTR(Source::TableAux) getTableAux() const {
        return boost::static_pointer_cast<Source::TableAux>(RecordBase::getTableAux());
    }

    friend class detail::Access;

    SourceRecord(RecordBase const & other) : RecordInterface<Source>(other) {}
};

/**
 *  @brief Table class that contains measurements made on a single exposure.
 */
class SourceTable : public TableInterface<Source> {
public:

    /**
     *  @brief Construct a new table.
     *
     *  @param[in] schema            Schema that defines the fields, offsets, and record size for the table.
     *  @param[in] capacity          Number of records to pre-allocate space for in the first block.  This
     *                               overrides nRecordsPerBlock for the first block and the first block only.
     *  @param[in] metadata          Flexible metadata for the table.  An empty PropertyList will be used
     *                               if an empty pointer is passed.
     *  @param[in] idFactory         Factory class to generate record IDs when they are not explicitly given.
     *                               If empty, defaults to a simple counter that starts at 1.
     */
    SourceTable(
        Schema const & schema,
        int capacity = 0,
        PTR(daf::base::PropertyList) const & metadata = PTR(daf::base::PropertyList)(),
        PTR(IdFactory) const & idFactory = PTR(IdFactory)()
    ) : TableInterface<Source>(schema, capacity, idFactory) {
        TableBase::getAux() = boost::make_shared<Source::TableAux>(metadata);
    }

    /// @copydoc TableBase::TableBase()
    SourceTable() {}

    /// @brief Return the flexible metadata associated with the source table.
    PTR(daf::base::PropertyList) getMetadata() const { return getAux()->metadata; }

    /// @brief Set the flexible metadata associated with the source table.
    void setMetadata(PTR(daf::base::PropertyList) const & metadata) const { getAux()->metadata = metadata; }

    ADD_FLUX_DEFINERS(Psf, CANONICAL_PSF)
    ADD_FLUX_DEFINERS(Model, CANONICAL_MODEL)
    ADD_FLUX_DEFINERS(Ap, CANONICAL_AP)
    ADD_FLUX_DEFINERS(Inst, CANONICAL_INST)
    ADD_CENTROID_DEFINERS
    ADD_SHAPE_DEFINERS

    /// @brief Create and add a new record with an ID generated by the table's IdFactory.
    Record addRecord() const { return _addRecord(); }
    
    /// @brief Create and add a new record with an ID generated by the table's IdFactory.
    Record addRecord(Footprint const & footprint) const {
        return _addRecord(boost::make_shared<Source::RecordAux>(footprint));
    }

    /// @brief Create and add a new record with an explicit RecordId.
    Record addRecord(RecordId id) const { return _addRecord(id); }

    /// @brief Create and add a new record with an explicit RecordId.
    Record addRecord(RecordId id, Footprint const & footprint) const {
        return _addRecord(id, boost::make_shared<Source::RecordAux>(footprint));
    }

    /**
     *  @brief Write a FITS binary table.
     *
     *  @param[in]   filename        Name of the FITS file to open.  This will be passed directly
     *                               to cfitsio, so all of its extended filename syntaxes should
     *                               work here.
     */
    void writeFits(std::string const & filename);
    
    /**
     *  @brief Load a table from a FITS binary table.
     *
     *  @param[in]   filename        Name of the FITS file to open.  This will be passed directly
     *                               to cfitsio, so all of its extended filename syntaxes should
     *                               work here.
     */
    static SourceTable readFits(std::string const & filename);

private:

    template <typename Tag> friend class RecordInterface;

    PTR(Source::TableAux) getAux() const {
        return boost::static_pointer_cast<Source::TableAux>(TableBase::getAux());
    }
    
    /// @brief Create a table from a base-class table.  Required to implement RecordBase::getTable().
    explicit SourceTable(TableBase const & base) : TableInterface<Source>(base) {}

};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Source_h_INCLUDED
