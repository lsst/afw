#include "lsst/afw/table.h"

using namespace lsst::afw::table;

int main(int argc, char const * argv[]) {
    std::string filename = "memleak.fits";
    SourceCatalog cat(SourceTable::makeMinimalSchema());
    cat.writeFits(filename);
    if (argc > 1 && std::string(argv[1]) == "read") {
        SourceCatalog readVector = SourceCatalog::readFits(filename);
    }
    return 0;
}
