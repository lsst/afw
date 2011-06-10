#include "lsst/afw/detection/Schema.h"

namespace afwDetect = lsst::afw::detection;

#include <boost/serialization/export.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
BOOST_CLASS_EXPORT(afwDetect::SchemaEntry)

/// Return a Schema given its name and component
afwDetect::Schema const& afwDetect::Schema::find(
        std::string const& name,        ///< The name of the desired Schema
        std::string const& component    ///< The component name, if not blank
                          ) const {
    for (std::vector<afwDetect::Schema::Ptr>::const_iterator ptr = _entries.begin();
         ptr != _entries.end(); ++ptr) {
        if ((*ptr)->_component == component) {
            afwDetect::Schema const& val = (*ptr)->find(name);
            if (val) {
                return val;
            }
        } else if ((*ptr)->_component == "") {
            afwDetect::Schema const& val = (*ptr)->find(name, component);
            if (val) {
                return val;
            }
        }
    }
        
    return afwDetect::Schema::unknown();
}

/// Print v to os, using dynamic dispatch
std::ostream &operator<<(std::ostream &os, afwDetect::Schema const& v)
{
    return v.output(os);
}

/// Print all the values to os;  note that this is a virtual function called by operator<<
std::ostream &afwDetect::Schema::output(std::ostream &os) const {
    for (std::vector<afwDetect::Schema::Ptr>::const_iterator ptr = _entries.begin();
         ptr != _entries.end(); ++ptr) {
        if (ptr != _entries.begin()) {
            os << " ";
        }
        os << "[";
        if (_component != "") {
            os << _component << ".";
        }
        os << **ptr << "]";
    }

    return os;
}

/**
 * Return the unknown object
 */
afwDetect::Schema const& afwDetect::Schema::unknown() {
    static afwDetect::Schema unknown("unknown", -1, UNKNOWN);

    return unknown;
}
