#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/slots.h"

namespace lsst { namespace afw { namespace table {

namespace {

// Return true if 'a' starts with 'b'
bool startsWith(std::string const & a, std::string const b) {
    return a.compare(0, b.size(), b) == 0;
}

} // anonymous

void SlotDefinition::setKeys(std::string const & alias, Schema const & schema) {
    SubSchema s = schema["slot"][_name];
    if (!alias.empty() && !startsWith(alias, s.getPrefix())) return;
    try {
        if (schema.getVersion() == 0) {
            _suspectKey = s["flags"];
            _failureKey = s["flags"];
        } else if (schema.getVersion() == 1) {
            _suspectKey = s["flag"];
            _failureKey = s["flag"];
        } else {
            _suspectKey = s["flag"]["suspect"];
            _failureKey = s["flag"]["failure"];
        }
    } catch (pex::exceptions::NotFoundError &) {}
}

void FluxSlotDefinition::setKeys(std::string const & alias, Schema const & schema) {
    SubSchema s = schema["slot"][_name];
    if (!alias.empty() && !startsWith(alias, s.getPrefix())) return;
    try {
        if (schema.getVersion() == 0) {
            _measKey = s;
            try {
                _errKey = s["err"];
            } catch (pex::exceptions::NotFoundError &) {}
        } else {
            _measKey = s["flux"];
            try {
                _errKey = s["fluxSigma"];
            } catch (pex::exceptions::NotFoundError &) {}
        }
    } catch (pex::exceptions::NotFoundError & err) {
        if (!alias.empty()) throw;
    }
    SlotDefinition::setKeys(alias, schema);
}

namespace {

CentroidSlotDefinition::ErrKey::NameArray makeCentroidNameArray() {
    CentroidSlotDefinition::ErrKey::NameArray v;
    v.push_back("x");
    v.push_back("y");
    return v;
}

} // anonymous

void CentroidSlotDefinition::setKeys(std::string const & alias, Schema const & schema) {
    SubSchema s = schema["slot"][_name];
    if (!alias.empty() && !startsWith(alias, s.getPrefix())) return;
    static ErrKey::NameArray names = makeCentroidNameArray();
    try {
        if (schema.getVersion() == 0) {
            _measKey = MeasKey(Key< Point<double> >(s));
            try {
                _errKey = ErrKey(Key< Covariance< Point<float> > >(s["err"]));
            } catch (pex::exceptions::NotFoundError &) {}
        } else {
            _measKey = s;
            try {
                _errKey = ErrKey(s, names);
            } catch (pex::exceptions::NotFoundError &) {}
        }
    } catch (pex::exceptions::NotFoundError & err) {
        if (!alias.empty()) throw;
    }
    SlotDefinition::setKeys(alias, schema);
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

void ShapeSlotDefinition::setKeys(std::string const & alias, Schema const & schema) {
    SubSchema s = schema["slot"][_name];
    if (!alias.empty() && !startsWith(alias, s.getPrefix())) return;
    static ErrKey::NameArray names = makeShapeNameArray();
    try {
        if (schema.getVersion() == 0) {
            _measKey = MeasKey(Key< Moments<double> >(s));
            try {
                _errKey = ErrKey(Key< Covariance< Moments<float> > >(s["err"]));
            } catch (pex::exceptions::NotFoundError &) {}
        } else {
            _measKey = s;
            try {
                _errKey = ErrKey(s, names);
            } catch (pex::exceptions::NotFoundError &) {}
        }
    } catch (pex::exceptions::NotFoundError &) {
        if (!alias.empty()) throw;
    }
    SlotDefinition::setKeys(alias, schema);
}

void SlotSuite::handleAliasChange(std::string const & alias, Schema const & schema) {
    defPsfFlux.setKeys(alias, schema);
    defApFlux.setKeys(alias, schema);
    defInstFlux.setKeys(alias, schema);
    defModelFlux.setKeys(alias, schema);
    defCentroid.setKeys(alias, schema);
    defShape.setKeys(alias, schema);
}

SlotSuite::SlotSuite(Schema const & schema) :
    defPsfFlux("PsfFlux"),
    defApFlux("ApFlux"),
    defInstFlux("InstFlux"),
    defModelFlux("ModelFlux"),
    defCentroid("Centroid"),
    defShape("Shape")
{
    defPsfFlux.setKeys("", schema);
    defApFlux.setKeys("", schema);
    defInstFlux.setKeys("", schema);
    defModelFlux.setKeys("", schema);
    defCentroid.setKeys("", schema);
    defShape.setKeys("", schema);
}

//------ deprecated functions for verison=0 measurement algorithms ------------------------------------------

KeyTuple<Centroid> addCentroidFields(
    Schema & schema,
    std::string const & name,
    std::string const & doc
) {
    KeyTuple<Centroid> keys;
    keys.meas = schema.addField<Centroid::MeasTag>(name, doc, "pixels");
    keys.err = schema.addField<Centroid::ErrTag>(
        name + ".err", "covariance matrix for " + name, "pixels^2"
    );
    keys.flag = schema.addField<Flag>(
        name + ".flags", "set if the " + name + " measurement did not fully succeed"
    );
    return keys;
}

KeyTuple<Shape> addShapeFields(
    Schema & schema,
    std::string const & name,
    std::string const & doc
) {
    KeyTuple<Shape> keys;
    keys.meas = schema.addField<Shape::MeasTag>(
        name, doc, "pixels^2"
    );
    keys.err = schema.addField<Shape::ErrTag>(
        name + ".err", "covariance matrix for " + name, "pixels^4"
    );
    keys.flag = schema.addField<Flag>(
        name + ".flags", "set if the " + name + " measurement failed"
    );
    return keys;
}

KeyTuple<Flux> addFluxFields(
    Schema & schema,
    std::string const & name,
    std::string const & doc
) {
    KeyTuple<Flux> keys;
    keys.meas = schema.addField<Flux::MeasTag>(
        name, doc, "dn"
    );
    keys.err = schema.addField<Flux::ErrTag>(
        name + ".err", "uncertainty for " + name, "dn"
    );
    keys.flag = schema.addField<Flag>(
        name + ".flags", "set if the " + name + " measurement failed"
    );
    return keys;
}

}}} // namespace lsst::afw::table
