// -*- lsst-c++ -*-
#ifndef LSST_AFW_TABLE_slots_h_INCLUDED
#define LSST_AFW_TABLE_slots_h_INCLUDED

#include "lsst/afw/table/aggregates.h"

namespace lsst {

namespace daf {
namespace base {
class PropertySet;
}
}  // namespace daf

namespace afw {

namespace fits {
class Fits;
};  // namespace fits

namespace table {

/**
 *  Base class for helper classes that define slots on SourceTable/SourceRecord.
 *
 *  Each type of slot corresponds to a subclass of SlotDefinition, and each actual slot
 *  corresponds to a particular field name prefix.  For instance, to look up the centroid
 *  slot, we look for fields named "slot_Centroid_x" and "slot_Centroid_y".  Instead of actually
 *  naming a particular field that, however, we use Schema's alias mechanism (see AliasMap) to
 *  make these field name lookups resolve to the name of other fields.  The actual
 *  definition of the slots is thus managed by the Schema's AliasMap, though a SourceTable
 *  object will cache Keys for the various slots to make sure accessing slot values is
 *  efficient (more precisely, when you set an alias related to a slot on an AliasMap, any
 *  table it corresponds to will receive a notification that it should update its Keys).
 *  These cached Keys are actually stored within the SlotDefinition (as data members of
 *  derived classes).
 *
 *  Note that the uncertainty and failure flag components of slots are not required; a slot
 *  may have only a measurement defined, or a measurement and either one of these (but not both).
 *  A slot may not have only an uncertainty and/or a a failure flag, however.
 *
 *  A SlotDefinition instance is not just an internal object used by SourceTable; it can also be
 *  used to inspect the slots via SourceTable::getXxxSlot(), which is now the preferred way
 *  to access the Keys that slots correspond to.  SlotDefinition objects should only be
 *  constructed by SourceTable, however.
 */
class SlotDefinition {
public:
    /// Construct a SlotDefinition from the name of the slot (e.g. "Centroid" or "PsfFlux")
    explicit SlotDefinition(std::string const &name) : _name(name) {}

    /// Return the name of the slot (e.g. "Centroid" or "PsfFlux")
    std::string getName() const { return _name; }

    /**
     *  Return the alias field prefix used to lookup Keys for the slot.
     *
     *  This simply prepends "slot_" to the slot name.
     */
    std::string getAlias() const { return "slot_" + _name; }

    SlotDefinition(SlotDefinition const &) = default;
    SlotDefinition(SlotDefinition &&) = default;
    SlotDefinition &operator=(SlotDefinition const &) = default;
    SlotDefinition &operator=(SlotDefinition &&) = default;
    ~SlotDefinition() = default;

protected:
    std::string _name;
};

/// SlotDefinition specialization for fluxes
class FluxSlotDefinition : public SlotDefinition {
public:
    typedef double MeasValue;     ///< Type returned by accessing the slot measurement
    typedef double ErrValue;      ///< Type returned by accessing the slot uncertainty
    typedef Key<double> MeasKey;  ///< Key type used to access the slot measurement
    typedef Key<double> ErrKey;   ///< Key type used to access the slot uncertainty

    /// Construct a SlotDefinition from the name of the slot (e.g. "PsfFlux")
    explicit FluxSlotDefinition(std::string const &name) : SlotDefinition(name) {}

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
    void setKeys(std::string const &alias, Schema const &schema);

    FluxSlotDefinition(FluxSlotDefinition const &) = default;
    FluxSlotDefinition(FluxSlotDefinition &&) = default;
    FluxSlotDefinition &operator=(FluxSlotDefinition const &) = default;
    FluxSlotDefinition &operator=(FluxSlotDefinition &&) = default;
    ~FluxSlotDefinition() = default;

private:
    MeasKey _measKey;
    ErrKey _errKey;
    Key<Flag> _flagKey;
};

/// SlotDefinition specialization for centroids
class CentroidSlotDefinition : public SlotDefinition {
public:
    typedef lsst::geom::Point2D MeasValue;         ///< Type returned by accessing the slot measurement
    typedef Eigen::Matrix<float, 2, 2> ErrValue;   ///< Type returned by accessing the slot uncertainty
    typedef Point2DKey MeasKey;                    ///< Key type used to access the slot measurement
    typedef CovarianceMatrixKey<float, 2> ErrKey;  ///< Key type used to access the slot uncertainty

    /// Construct a SlotDefinition from the name of the slot (e.g. "Centroid")
    explicit CentroidSlotDefinition(std::string const &name) : SlotDefinition(name) {}

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
    void setKeys(std::string const &alias, Schema const &schema);

    CentroidSlotDefinition(CentroidSlotDefinition const &) = default;
    CentroidSlotDefinition(CentroidSlotDefinition &&) = default;
    CentroidSlotDefinition &operator=(CentroidSlotDefinition const &) = default;
    CentroidSlotDefinition &operator=(CentroidSlotDefinition &&) = default;
    ~CentroidSlotDefinition() = default;

private:
    MeasKey _measKey;
    ErrKey _errKey;
    Key<Flag> _flagKey;
};

/// SlotDefinition specialization for shapes
class ShapeSlotDefinition : public SlotDefinition {
public:
    typedef geom::ellipses::Quadrupole MeasValue;  ///< Type returned by accessing the slot measurement
    typedef Eigen::Matrix<float, 3, 3> ErrValue;   ///< Type returned by accessing the slot uncertainty
    typedef QuadrupoleKey MeasKey;                 ///< Key type used to access the slot measurement
    typedef CovarianceMatrixKey<float, 3> ErrKey;  ///< Key type used to access the slot uncertainty

    /// Construct a SlotDefinition from the name of the slot (e.g. "Shape")
    explicit ShapeSlotDefinition(std::string const &name) : SlotDefinition(name) {}

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
    void setKeys(std::string const &alias, Schema const &schema);

    ShapeSlotDefinition(ShapeSlotDefinition const &) = default;
    ShapeSlotDefinition(ShapeSlotDefinition &&) = default;
    ShapeSlotDefinition &operator=(ShapeSlotDefinition const &) = default;
    ShapeSlotDefinition &operator=(ShapeSlotDefinition &&) = default;
    ~ShapeSlotDefinition() = default;

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
    FluxSlotDefinition defPsfFlux;
    FluxSlotDefinition defApFlux;
    FluxSlotDefinition defGaussianFlux;
    FluxSlotDefinition defModelFlux;
    FluxSlotDefinition defCalibFlux;
    CentroidSlotDefinition defCentroid;
    ShapeSlotDefinition defShape;

    /// Handle a callback from an AliasMap informing the table that an alias has changed.
    void handleAliasChange(std::string const &alias, Schema const &schema);

    /// Initialize the slots.
    explicit SlotSuite(Schema const &schema);
};
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_TABLE_slots_h_INCLUDED
