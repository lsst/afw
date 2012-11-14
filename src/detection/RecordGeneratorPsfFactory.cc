// -*- lsst-c++ -*-

#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection/RecordGeneratorPsfFactory.h"
#include "lsst/afw/detection/Psf.h"

namespace lsst { namespace afw { namespace detection {

namespace {

typedef std::map<std::string,RecordGeneratorPsfFactory*> Registry;

Registry & getRegistry() {
    static Registry registry;
    return registry;
}

} // anonymous

RecordGeneratorPsfFactory::RecordGeneratorPsfFactory(std::string const & name) {
    getRegistry()[name] = this;
}

afw::table::RecordOutputGeneratorSet Psf::writeToRecords() const {
    throw LSST_EXCEPT(
        pex::exceptions::LogicErrorException,
        "Record persistence is not implemented for this Psf"
    );
}

PTR(Psf) Psf::readFromRecords(afw::table::RecordInputGeneratorSet const & inputs) {
    Registry::iterator i = getRegistry().find(inputs.name);
    if (i == getRegistry().end()) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            boost::str(boost::format("No RecordGeneratorPsfFactory with name '%s'") % inputs.name)
        );
    }
    return (*i->second)(inputs);
}

}}} // namespace lsst::afw::detection
