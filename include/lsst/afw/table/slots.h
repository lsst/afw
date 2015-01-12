// -*- lsst-c++ -*-
#ifndef LSST_AFW_TABLE_slots_h_INCLUDED
#define LSST_AFW_TABLE_slots_h_INCLUDED

#include "lsst/afw/table/aggregates.h"

namespace lsst {

namespace daf { namespace base {
class PropertySet;
}} // namespace daf::base

namespace afw {

namespace fits {
class Fits;
}; // namespace fits

namespace table {


/**
 *  Base class for helper classes that define slots on SourceTable/SourceRecord.
 *
 *  Each type of slot corresponds to a subclass of Slot, and each actual
 *  slot corresponds to a particular field name prefix.  For instance, to look up
 *  the centroid slot, we look for fields named "slot_Centroid_x" and "slot_Centroid_y"
 *  (or a single compound "slot.Centroid" field in version 0).  Instead of actually
 *  naming a particular field that, however, we use Schema's alias mechanism (see AliasMap)
 *  to make these field name lookups resolve to the name of other fields.  The actual
 *  definition of the slots is thus managed by the Schema's AliasMap, though a SourceTable
 *  object will cache Keys for the various slots to make sure accessing slot values is
 *  efficient (more precisely, when you set an alias related to a slot on an AliasMap, any
 *  table it corresponds to will receive a notification that it should update its Keys).
 *  These cached Keys are actually stored within the Slot (as data members of
 *  derived classes).
 *
 *  Note that the uncertainty and failure flag components of slots are not required; a slot
 *  may have only a measurement defined, or a measurement and either one of these (but not both).
 *  A slot may not have only an uncertainty and/or a a failure flag, however.
 *
 *  A Slot instance is not just an internal object used by SourceTable; it can also be
 *  used to inspect the slots via SourceTable::getXxxSlot(), which is now the preferred way
 *  to access the Keys that slots correspond to.  Slot objects should only be
 *  constructed by SourceTable, however.
 *
 *  If a measurement algorithms uses the standard naming conventions for its output fields,
 *  an single alias that points from the slot name (e.g. "slot_Centroid") to the measurement
 *  algorithm name (e.g. "base_SdssCentroid") will be sufficient to define the measurement,
 *  uncertainty, and flag keys, because the alias replacement will be applied just to those
 *  prefixes.  Individual aliases can also be set up for each field, however, to support
 *  algorithms that cannot follow the standard naming conventions.  SourceTable's
 *  defineXxSlot methods only support the simpler, single-alias approach, though they will
 *  remove multiple old aliases when changing the definition of a slot.
 */
class Slot {
public:

    /// Construct a Slot from the name of the slot (e.g. "Centroid" or "PsfFlux")
    explicit Slot(std::string const & name) : _name(name) {}

    /// Return the name of the slot (e.g. "Centroid" or "PsfFlux")
    std::string getName() const { return _name; }

    /**
     *  Return the alias field prefix used to lookup Keys for the slot.
     *
     *  This simply prepends "slot_" to the slot name (or "slot." for version 0 tables).
     */
    std::string getAlias(int version) const {
        return (version > 0 ? "slot_" : "slot.") + _name;
    }

protected:
    std::string _name;
};

/// Slot specialization for fluxes
class FluxSlot : public Slot {
public:

    typedef double MeasValue;    ///< Type returned by accessing the slot measurement
    typedef double ErrValue;     ///< Type returned by accessing the slot uncertainty
    typedef Key<double> MeasKey; ///< Key type used to access the slot measurement
    typedef Key<double> ErrKey;  ///< Key type used to access the slot uncertainty

    /// Construct a Slot from the name of the slot (e.g. "PsfFlux")
    explicit FluxSlot(std::string const & name) : Slot(name) {}

    /// Return true if the key associated with the measurement is valid.
    bool isValid() const { return _measKey.isValid(); }

    /// Return the cached Key used to access the slot measurement
    MeasKey getMeasKey() const { return _measKey; }

    /// Return the cached Key used to access the slot uncertainty
    ErrKey getErrKey() const { return _errKey; }

    /// Return the cached Key used to access the slot failure flag
    Key<Flag> getFlagKey() const { return _flagKey; }

    /**
     *  Update the cached Keys following an change of aliases in the given Schema
     *
     *  This method is intended for internal use by SourceTable only.
     *
     *  @param[in] alias     If non-empty, abort early if this string does not start
     *                       with getAlias() (used to see if an alias change might
     *                       have affected this slot, and avoid unnecessary work if not).
     *  @param[in] schema    Schema to search for Keys.
     */
    void setKeys(std::string const & alias, Schema const & schema);

private:
    MeasKey _measKey;
    ErrKey _errKey;
    Key<Flag> _flagKey;
};

/// Slot specialization for centroids
class CentroidSlot : public Slot {
public:

    typedef geom::Point2D MeasValue;             ///< Type returned by accessing the slot measurement
    typedef Eigen::Matrix<float,2,2> ErrValue;   ///< Type returned by accessing the slot uncertainty
    typedef Point2DKey MeasKey;                  ///< Key type used to access the slot measurement
    typedef CovarianceMatrixKey<float,2> ErrKey; ///< Key type used to access the slot uncertainty

    /// Construct a Slot from the name of the slot (e.g. "Centroid")
    explicit CentroidSlot(std::string const & name) : Slot(name) {}

    /// Return true if the key associated with the measurement is valid.
    bool isValid() const { return _measKey.isValid(); }

    /// Return the cached Key used to access the slot measurement
    MeasKey getMeasKey() const { return _measKey; }

    /// Return the cached Key used to access the slot uncertainty
    ErrKey getErrKey() const { return _errKey; }

    /// Return the cached Key used to access the slot failure flag
    Key<Flag> getFlagKey() const { return _flagKey; }

    /**
     *  Update the cached Keys following an change of aliases in the given Schema
     *
     *  This method is intended for internal use by SourceTable only.
     *
     *  @param[in] alias     If non-empty, abort early if this string does not start
     *                       with getAlias() (used to see if an alias change might
     *                       have affected this slot, and avoid unnecessary work if not).
     *  @param[in] schema    Schema to search for Keys.
     */
    void setKeys(std::string const & alias, Schema const & schema);

private:
    MeasKey _measKey;
    ErrKey _errKey;
    Key<Flag> _flagKey;
};

/// Slot specialization for shapes
class ShapeSlot : public Slot {
public:

    typedef geom::ellipses::Quadrupole MeasValue; ///< Type returned by accessing the slot measurement
    typedef Eigen::Matrix<float,3,3> ErrValue;    ///< Type returned by accessing the slot uncertainty
    typedef QuadrupoleKey MeasKey;                ///< Key type used to access the slot measurement
    typedef CovarianceMatrixKey<float,3> ErrKey;  ///< Key type used to access the slot uncertainty

    /// Construct a Slot from the name of the slot (e.g. "Shape")
    explicit ShapeSlot(std::string const & name) : Slot(name) {}

    /// Return true if the key associated with the measurement is valid.
    bool isValid() const { return _measKey.isValid(); }

    /// Return the cached Key used to access the slot measurement
    MeasKey getMeasKey() const { return _measKey; }

    /// Return the cached Key used to access the slot uncertainty
    ErrKey getErrKey() const { return _errKey; }

    /// Return the cached Key used to access the slot failure flag
    Key<Flag> getFlagKey() const { return _flagKey; }

    /**
     *  Update the cached Keys following an change of aliases in the given Schema
     *
     *  This method is intended for internal use by SourceTable only.
     *
     *  @param[in] alias     If non-empty, abort early if this string does not start
     *                       with getAlias() (used to see if an alias change might
     *                       have affected this slot, and avoid unnecessary work if not).
     *  @param[in] schema    Schema to search for Keys.
     */
    void setKeys(std::string const & alias, Schema const & schema);

private:
    MeasKey _measKey;
    ErrKey _errKey;
    Key<Flag> _flagKey;
};

/**
 *  An aggregate containing all of the current slots used in SourceTable.
 *
 *  This is essentially for internal use by SourceTable only; it is defined here to keep
 *  the source code for the slot mechanism in one place as much as possible.
 */
struct SlotSuite {
    FluxSlot defPsfFlux;
    FluxSlot defApFlux;
    FluxSlot defInstFlux;
    FluxSlot defModelFlux;
    CentroidSlot defCentroid;
    ShapeSlot defShape;

    /// Handle a callback from an AliasMap informing the table that an alias has changed.
    void handleAliasChange(std::string const & alias, Schema const & schema);

    /// Initialize the slots.
    explicit SlotSuite(Schema const & schema);
};


//@{
/**
 *  Utilities for version 0 tables and measurement algorithms that fill them.
 *
 *  These are deprecated and should not be used by new code; they will be removed when
 *  the new measurement framework in meas_base is complete and the old one in meas_algorithms
 *  is retired.
 */

#ifndef SWIG

template <typename MeasTagT, typename ErrTagT>
struct Measurement {
    typedef MeasTagT MeasTag;  ///< the tag (template parameter) type used for the measurement
    typedef ErrTagT ErrTag;    ///< the tag (template parameter) type used for the uncertainty
    typedef typename Field<MeasTag>::Value MeasValue; ///< the value type used for the measurement
    typedef typename Field<ErrTag>::Value ErrValue;   ///< the value type used for the uncertainty
    typedef Key<MeasTag> MeasKey;  ///< the Key type for the actual measurement
    typedef Key<ErrTag> ErrKey;    ///< the Key type for the error on the measurement
};

/// A collection of types useful for flux measurement algorithms.
struct Flux : public Measurement<double, double> {};

/// A collection of types useful for centroid measurement algorithms.
struct Centroid : public Measurement< Point<double>, Covariance< Point<float> > > {};

/// A collection of types useful for shape measurement algorithms.
struct Shape : public Measurement< Moments<double>, Covariance< Moments<float> > > {};

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

//@

}}} // lsst::afw::table

#endif // !LSST_AFW_TABLE_slots_h_INCLUDED
