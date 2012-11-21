// -*- lsst-c++ -*-

#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/RecordGeneratorWcsFactory.h"
#include "lsst/afw/image/Wcs.h"

namespace lsst { namespace afw { namespace image {

namespace {

typedef std::map<std::string,RecordGeneratorWcsFactory*> Registry;

Registry & getRegistry() {
    static Registry registry;
    return registry;
}

} // anonymous

RecordGeneratorWcsFactory::RecordGeneratorWcsFactory(std::string const & name) {
    getRegistry()[name] = this;
}

afw::table::RecordOutputGeneratorSet Wcs::writeToRecords() const {
    throw LSST_EXCEPT(
        pex::exceptions::LogicErrorException,
        "Record persistence is not implemented for this Wcs"
    );
}

PTR(Wcs) Wcs::readFromRecords(afw::table::RecordInputGeneratorSet const & inputs) {
    Registry::iterator i = getRegistry().find(inputs.name);
    if (i == getRegistry().end()) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            boost::str(boost::format("No RecordGeneratorWcsFactory with name '%s'") % inputs.name)
        );
    }
    return (*i->second)(inputs);
}

}}} // namespace lsst::afw::image
