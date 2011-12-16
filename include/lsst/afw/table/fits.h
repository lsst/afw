// -*- lsst-c++ -*-
#ifndef AFW_TABLE_fits_h_INCLUDED
#define AFW_TABLE_fits_h_INCLUDED

#include "lsst/afw/fits.h"
#include "lsst/afw/table/Schema.h"

namespace lsst { namespace afw { namespace table {

void createFitsHeader(afw::fits::Fits & fits, Schema const & schema, bool sanitizeNames);

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_fits_h_INCLUDED
