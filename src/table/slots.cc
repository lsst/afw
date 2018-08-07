#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/slots.h"

namespace lsst {
namespace afw {
namespace table {

namespace {

// Return true if 'a' starts with 'b'
bool startsWith(std::string const &a, std::string const b) { return a.compare(0, b.size(), b) == 0; }

// helper class that resolves aliases for a slot's main measurement field, while
// distiguishing between two cases where the field can't be found:
//  - if the alias points to an invalid field, we should throw an exception (which is actually
//    done by the calling setKeys() method, by allowing a lower-level exception to propagate up).
//  - if the alias simply isn't defined, we should just reset the slot with invalid keys
class MeasFieldNameGetter {
public:
    MeasFieldNameGetter(SubSchema const &s, Schema const &schema)
            : replaced(schema[schema.getAliasMap()->apply(s.getPrefix())]),
              defined(replaced.getPrefix() !=
                      s.getPrefix())  // slot is defined if applying alias wasn't a no-op
    {}

    SubSchema replaced;  // a SubSchema that includes alias replacement
    bool defined;        // whether the slot is defined at all
};

}  // namespace

void FluxSlotDefinition::setKeys(std::string const &alias, Schema const &schema) {
    SubSchema s = schema["slot"][_name];
    if (!alias.empty() && !startsWith(alias, s.getPrefix())) return;
    _measKey = MeasKey();
    _errKey = ErrKey();
    _flagKey = Key<Flag>();
    MeasFieldNameGetter helper(s["flux"], schema);
    if (!helper.defined) {
        return;
    }
    _measKey = helper.replaced;
    try {
        _errKey = s["fluxErr"];
    } catch (pex::exceptions::NotFoundError &) {
    }
    try {
        _flagKey = s["flag"];
    } catch (pex::exceptions::NotFoundError &) {
    }
}

namespace {

CentroidSlotDefinition::ErrKey::NameArray makeCentroidNameArray() {
    CentroidSlotDefinition::ErrKey::NameArray v;
    v.push_back("x");
    v.push_back("y");
    return v;
}

}  // namespace

void CentroidSlotDefinition::setKeys(std::string const &alias, Schema const &schema) {
    SubSchema s = schema["slot"][_name];
    if (!alias.empty() && !startsWith(alias, s.getPrefix())) return;
    static ErrKey::NameArray names = makeCentroidNameArray();
    _measKey = MeasKey();
    _errKey = ErrKey();
    _flagKey = Key<Flag>();
    MeasFieldNameGetter helper(s, schema);
    if (!helper.defined) return;
    _measKey = helper.replaced;
    try {
        _errKey = ErrKey(s, names);
    } catch (pex::exceptions::NotFoundError &) {
    }
    try {
        _flagKey = s["flag"];
    } catch (pex::exceptions::NotFoundError &) {
    }
}

namespace {

ShapeSlotDefinition::ErrKey::NameArray makeShapeNameArray() {
    ShapeSlotDefinition::ErrKey::NameArray v;
    v.push_back("xx");
    v.push_back("yy");
    v.push_back("xy");
    return v;
}

}  // namespace

void ShapeSlotDefinition::setKeys(std::string const &alias, Schema const &schema) {
    SubSchema s = schema["slot"][_name];
    if (!alias.empty() && !startsWith(alias, s.getPrefix())) return;
    static ErrKey::NameArray names = makeShapeNameArray();
    _measKey = MeasKey();
    _errKey = ErrKey();
    _flagKey = Key<Flag>();
    MeasFieldNameGetter helper(s, schema);
    if (!helper.defined) return;
    _measKey = helper.replaced;
    try {
        _errKey = ErrKey(s, names);
    } catch (pex::exceptions::NotFoundError &) {
    }
    try {
        _flagKey = s["flag"];
    } catch (pex::exceptions::NotFoundError &) {
    }
}

void SlotSuite::handleAliasChange(std::string const &alias, Schema const &schema) {
    defPsfFlux.setKeys(alias, schema);
    defApFlux.setKeys(alias, schema);
    defInstFlux.setKeys(alias, schema);
    defModelFlux.setKeys(alias, schema);
    defCalibFlux.setKeys(alias, schema);
    defCentroid.setKeys(alias, schema);
    defShape.setKeys(alias, schema);
}

SlotSuite::SlotSuite(Schema const &schema)
        : defPsfFlux("PsfFlux"),
          defApFlux("ApFlux"),
          defInstFlux("InstFlux"),
          defModelFlux("ModelFlux"),
          defCalibFlux("CalibFlux"),
          defCentroid("Centroid"),
          defShape("Shape") {
    defPsfFlux.setKeys("", schema);
    defApFlux.setKeys("", schema);
    defInstFlux.setKeys("", schema);
    defModelFlux.setKeys("", schema);
    defCalibFlux.setKeys("", schema);
    defCentroid.setKeys("", schema);
    defShape.setKeys("", schema);
}
}  // namespace table
}  // namespace afw
}  // namespace lsst
