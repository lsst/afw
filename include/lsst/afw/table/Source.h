// -*- lsst-c++ -*-
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

struct Photometry : public Measurement<double,double> {};
struct Astrometry : public Measurement< Point<double>, Covariance< Point<double> > > {};
struct Shape : public Measurement< Moments<double>, Covariance< Moments<double> > > {};

enum CanonicalPhotometryEnum {
    CANONICAL_PSF=0,
    CANONICAL_MODEL,
    CANONICAL_AP,
    CANONICAL_INST,
    N_CANONICAL_PHOTOMETRY_FIELDS
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

        KeyPair<Photometry> photometry[N_CANONICAL_PHOTOMETRY_FIELDS];
        KeyPair<Astrometry> astrometry;
        KeyPair<Shape> shape;

        TableAux(PTR(daf::base::PropertyList) const & metadata_) : metadata(metadata_) {}
    };

#endif

};

/**
 *  @brief A bare-bones record class intended for testing and generic tabular data.
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

#define ADD_PHOTOMETRY_GETTERS(NAME, INDEX)                    \
    Photometry::MeasValue get ## NAME ## Flux() const {        \
        return get(getTableAux()->photometry[INDEX].meas);     \
    }                                                          \
    Photometry::ErrValue get ## NAME ## FluxErr() const {      \
        return get(getTableAux()->photometry[INDEX].err);      \
    }

    ADD_PHOTOMETRY_GETTERS(Psf, CANONICAL_PSF)
    ADD_PHOTOMETRY_GETTERS(Model, CANONICAL_MODEL)
    ADD_PHOTOMETRY_GETTERS(Ap, CANONICAL_AP)
    ADD_PHOTOMETRY_GETTERS(Inst, CANONICAL_INST)

#undef ADD_PHOTOMETRY_GETTERS

    Astrometry::MeasValue getCentroid() const { return get(getTableAux()->astrometry.meas); }
    Astrometry::ErrValue getCentroidCov() const { return get(getTableAux()->astrometry.err); }

    Shape::MeasValue getShape() const { return get(getTableAux()->shape.meas); }
    Shape::ErrValue getShapeCov() const { return get(getTableAux()->shape.err); }

private:

    PTR(Source::TableAux) getTableAux() const {
        return boost::static_pointer_cast<Source::TableAux>(RecordBase::getTableAux());
    }

    friend class detail::Access;

    SourceRecord(RecordBase const & other) : RecordInterface<Source>(other) {}
};

/**
 *  @brief A bare-bones record class intended for testing and generic tabular data.
 */
class SourceTable : public TableInterface<Source> {
public:

    /**
     *  @brief Construct a new table.
     *
     *  @param[in] schema            Schema that defines the fields, offsets, and record size for the table.
     *  @param[in] capacity          Number of records to pre-allocate space for in the first block.  This
     *                               overrides nRecordsPerBlock for the first block and the first block only.
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

    /// @brief Return the flexible metadata associated with the source table.
    PTR(daf::base::PropertyList) getMetadata() const { return getAux()->metadata; }

    /// @brief Set the flexible metadata associated with the source table.
    void setMetadata(PTR(daf::base::PropertyList) const & metadata) const { getAux()->metadata = metadata; }

#define ADD_PHOTOMETRY_DEFINERS(NAME, INDEX)    \
    /** @brief Set the measurement used for a canonical photometry slot with Keys. */ \
    void define ## NAME ## Photometry(                                  \
        Photometry::MeasKey const & flux,                               \
        Photometry::ErrKey const & err                                  \
    ) const {                                                           \
        getAux()->photometry[INDEX] = KeyPair<Photometry>(flux, err);   \
    }                                                                   \
    /** @brief Set the measurement used for a canonical photometry slot with a given name. */ \
    /** This requires that the field in question adhere to the convention of having */ \
    /** '<name>' and '<name>.err' fields. */                   \
    void define ## NAME ## Photometry(std::string const & name) const { \
        Schema schema = getSchema();                                    \
        getAux()->photometry[INDEX] = KeyPair<Photometry>(schema[name], schema[name]["err"]); \
    }                                                                   \
    /* @brief Return the name of the field used for a canonical photometry slot. */ \
    std::string get ## NAME ## PhotometryDefinition() const {           \
        return getSchema().find(getAux()->photometry[INDEX].meas).field.getName(); \
    }
    ADD_PHOTOMETRY_DEFINERS(Psf, CANONICAL_PSF)
    ADD_PHOTOMETRY_DEFINERS(Model, CANONICAL_MODEL)
    ADD_PHOTOMETRY_DEFINERS(Ap, CANONICAL_AP)
    ADD_PHOTOMETRY_DEFINERS(Inst, CANONICAL_INST)
#undef ADD_PHOTOMETRY_DEFINERS

    /// @brief Set the measurement used for the canonical astrometry slot with Keys.
    void defineAstrometry(Astrometry::MeasKey const & flux, Astrometry::ErrKey const & err) const {
        getAux()->astrometry = KeyPair<Astrometry>(flux, err);
    }

    /**
     *  @brief Set the measurement used for the canonical astrometry slot with a field name.
     *
     * This requires that the field in question adhere to the convention of having
     * '<name>' and '<name>.cov' fields.
     */
    void defineAstrometry(std::string const & name) const {
        Schema schema = getSchema();
        getAux()->astrometry = KeyPair<Astrometry>(schema[name], schema[name]["cov"]);
    }

    // @brief Return the name of the field used for the canonical astrometry slot.
    std::string getAstrometryDefinition() const {
        return getSchema().find(getAux()->astrometry.meas).field.getName();
    }

    /// @brief Set the measurement used for the canonical shape slot with Keys.
    void defineShape(Shape::MeasKey const & flux, Shape::ErrKey const & err) const {
        getAux()->shape = KeyPair<Shape>(flux, err);
    }

    /**
     *  @brief Set the measurement used for the canonical shape slot with a field name.
     *
     * This requires that the field in question adhere to the convention of having
     * '<name>' and '<name>.cov' fields.
     */
    void defineShape(std::string const & name) const {
        Schema schema = getSchema();
        getAux()->shape = KeyPair<Shape>(schema[name], schema[name]["cov"]);
    }

    // @brief Return the name of the field used for the canonical shape slot.
    std::string getShapeDefinition() const {
        return getSchema().find(getAux()->shape.meas).field.getName();
    }


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
     *  @param[in]   sanitizeNames   If true, periods in names will be converted to underscores.
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
