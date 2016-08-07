#include <iostream>
#include "lsst/daf/base/Citizen.h"
#include "lsst/afw/table.h"

using namespace lsst::afw::table;

int main(int argc, char const * argv[]) {
    bool const read = (argc > 1 && std::string(argv[1]) == "read");
    int const memId = (argc > 2) ? std::atoi(argv[2]) : 0;

    lsst::daf::base::Citizen::setNewCallbackId(memId);
    lsst::daf::base::Citizen::setDeleteCallbackId(memId);

    std::string filename = "memleak.fits";
    {
        SourceCatalog cat(SourceTable::makeMinimalSchema());
        cat.writeFits(filename);
    }
    if (read) {
        SourceCatalog readVector = SourceCatalog::readFits(filename);
    }

    lsst::daf::base::Citizen::census(std::cout);

    return 0;
}
