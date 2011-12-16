// -*- lsst-c++ -*-
#include "lsst/afw/table/fits.h"
#include "lsst/afw/table/Simple.h"

namespace lsst { namespace afw { namespace table {

void SimpleTable::writeFits(std::string const & filename, bool sanitizeNames) {
    fits::Fits file = fits::Fits::createFile(filename.c_str());
    file.checkStatus();
    fits::writeFitsHeader(file, getSchema(), sanitizeNames);
    fits::writeFitsRecords(file, *this);
    file.closeFile();
    file.checkStatus();
}

void SimpleTable::writeFits(std::string const & filename, SchemaMapper const & mapper, bool sanitizeNames) {
    fits::Fits file = fits::Fits::createFile(filename.c_str());
    file.checkStatus();
    fits::writeFitsHeader(file, getSchema(), sanitizeNames);
    fits::writeFitsRecords(file, *this, mapper);
    file.closeFile();
    file.checkStatus();
}

}}} // namespace lsst::afw::table
