// -*- lsst-c++ -*-

#include "lsst/afw/table/io/FitsReader.h"

namespace lsst {
namespace afw {
namespace table {
namespace io {

namespace {

typedef std::map<std::string, FitsReader const*> Registry;

Registry& getRegistry() {
    static Registry it;
    return it;
}

static FitsReader const baseFitsReader("BASE");

}  // anonymous

std::shared_ptr<BaseTable> FitsReader::makeTable(FitsSchemaInputMapper& mapper,
                                                 std::shared_ptr<daf::base::PropertyList> metadata,
                                                 int ioFlags, bool stripMetadata) const {
    std::shared_ptr<BaseTable> result = BaseTable::make(mapper.finalize());
    result->setMetadata(metadata);
    return result;
}

FitsReader::FitsReader(std::string const& name) { getRegistry()[name] = this; }

FitsReader const* FitsReader::_lookupFitsReader(daf::base::PropertyList const& metadata) {
    std::string name = metadata.get(std::string("AFW_TYPE"), std::string("BASE"));
    Registry::iterator i = getRegistry().find(name);
    if (i == getRegistry().end()) {
        throw LSST_EXCEPT(
                lsst::pex::exceptions::NotFoundError,
                (boost::format("FitsReader with name '%s' does not exist; check AFW_TYPE keyword.") %
                 name).str());
    }
    return i->second;
}

void FitsReader::_setupArchive(afw::fits::Fits& fits, FitsSchemaInputMapper& mapper,
                               std::shared_ptr<InputArchive> archive, int ioFlags) const {
    if (usesArchive(ioFlags)) {
        if (archive) {
            mapper.setArchive(archive);
        } else {
            mapper.readArchive(fits);
        }
    }
}
}
}
}
}  // namespace lsst::afw::table::io
