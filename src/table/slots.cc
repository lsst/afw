#include "lsst/afw/table/slots.h"
#include "lsst/pex/exceptions.h"

namespace lsst { namespace afw { namespace table {

void FluxSlotDefinition::define0(std::string const & name, Schema const & schema) {
    SubSchema s = schema[_target];
    _measKey = s;
    try {
        _errKey = s["err"];
    } catch (pex::exceptions::NotFoundError &) {}
    try {
        _flagKey = s["flags"];
    } catch (pex::exceptions::NotFoundError &) {}
}

void FluxSlotDefinition::handleAliasChange(std::string const & alias, Schema const & schema) {
    if (alias.compare(0, _target.size(), _target) != 0) return;
    SubSchema s = schema[_target];
    _measKey = s["flux"];
    try {
        _errKey = s["fluxSigma"];
    } catch (pex::exceptions::NotFoundError &) {}
    try {
        _flagKey = s["flag"];
    } catch (pex::exceptions::NotFoundError &) {}
}

void CentroidSlotDefinition::define0(std::string const & name, Schema const & schema) {
    SubSchema s = schema[_target];
    _measKey = MeasKey(Key< Point<double> >(s));
    try {
        _errKey = ErrKey(Key< Covariance< Point<float> > >(s["err"]));
    } catch (pex::exceptions::NotFoundError &) {}
    try {
        _flagKey = s["flags"];
    } catch (pex::exceptions::NotFoundError &) {}
}

namespace {

CentroidSlotDefinition::ErrKey::NameArray makeCentroidNameArray() {
    CentroidSlotDefinition::ErrKey::NameArray v;
    v.push_back("x");
    v.push_back("y");
    return v;
}

} // anonymous

void CentroidSlotDefinition::handleAliasChange(std::string const & alias, Schema const & schema) {
    static ErrKey::NameArray names = makeCentroidNameArray();
    if (alias.compare(0, _target.size(), _target) != 0) return;
    SubSchema s = schema[_target];
    _measKey = s;
    try {
        _errKey = ErrKey(s, names);
    } catch (pex::exceptions::NotFoundError &) {}
    try {
        _flagKey = s["flag"];
    } catch (pex::exceptions::NotFoundError &) {}
}

void ShapeSlotDefinition::define0(std::string const & name, Schema const & schema) {
    SubSchema s = schema[_target];
    _measKey = MeasKey(Key< Moments<double> >(s));
    try {
        _errKey = ErrKey(Key< Covariance< Moments<float> > >(s["err"]));
    } catch (pex::exceptions::NotFoundError &) {}
    try {
        _flagKey = s["flags"];
    } catch (pex::exceptions::NotFoundError &) {}
}

namespace {

ShapeSlotDefinition::ErrKey::NameArray makeShapeNameArray() {
    ShapeSlotDefinition::ErrKey::NameArray v;
    v.push_back("xx");
    v.push_back("yy");
    v.push_back("xy");
    return v;
}

} // anonymous

void ShapeSlotDefinition::handleAliasChange(std::string const & alias, Schema const & schema) {
    static ErrKey::NameArray names = makeShapeNameArray();
    if (alias.compare(0, _target.size(), _target) != 0) return;
    SubSchema s = schema[_target];
    _measKey = s;
    try {
        _errKey = ErrKey(s, names);
    } catch (pex::exceptions::NotFoundError &) {}
    try {
        _flagKey = s["flag"];
    } catch (pex::exceptions::NotFoundError &) {}
}

void SlotSuite::handleAliasChange(std::string const & alias, Schema const & schema) {
    defPsfFlux.handleAliasChange(alias, schema);
    defApFlux.handleAliasChange(alias, schema);
    defInstFlux.handleAliasChange(alias, schema);
    defModelFlux.handleAliasChange(alias, schema);
    defCentroid.handleAliasChange(alias, schema);
    defShape.handleAliasChange(alias, schema);
}

void SlotSuite::writeSlots(afw::fits::Fits & fits) const {
    // TODO
}

void SlotSuite::readSlots(daf::base::PropertySet & metadata, bool strip) {
    // TODO
}

SlotSuite::SlotSuite(int version) :
    defPsfFlux(version > 0 ? "slot_PsfFlux" : ""),
    defApFlux(version > 0 ? "slot_ApFlux" : ""),
    defInstFlux(version > 0 ? "slot_InstFlux" : ""),
    defModelFlux(version > 0 ? "slot_ModelFlux" : ""),
    defCentroid(version > 0 ? "slot_Centroid" : ""),
    defShape(version > 0 ? "slot_Shape" : "")
{}

}}} // namespace lsst::afw::table
