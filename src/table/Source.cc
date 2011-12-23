// -*- lsst-c++ -*-
#include "lsst/afw/table/fits.h"
#include "lsst/afw/table/Source.h"

namespace lsst { namespace afw { namespace table {

void SourceTable::writeFits(std::string const & filename) {
    fits::Fits file = fits::Fits::createFile(filename.c_str());
    file.checkStatus();
    fits::writeFitsHeader(file, getSchema(), true);
    fits::writeFitsRecords(file, *this);
    // TODO: save footprints
    file.closeFile();
    file.checkStatus();
}

SourceTable SourceTable::readFits(std::string const & filename) {
    fits::Fits file = fits::Fits::openFile(filename.c_str(), true);
    Schema schema = fits::readFitsHeader(file, true);
    int nRecords = 0;
    file.readKey("NAXIS2", nRecords);
    SourceTable table(schema, nRecords);
    fits::readFitsRecords(file, table);
    // TODO: load footprints
    file.closeFile();
    file.checkStatus();
    return table;
}

}}} // namespace lsst::afw::table
