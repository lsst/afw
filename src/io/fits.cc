// -*- lsst-c++ -*-

#include "fits_io.h"

#include "lsst/afw/table/TableBase.h"
#include "lsts/afw/image/fits_io_private.h"
#include "lsst/daf/base/PropertySet.h"

namespace lsst { namespace afw { namespace table {

void TableBase::_writeFits(
    std::string const & name,
    CONST_PTR(daf::base::PropertySet) const & metadata,
    std::string const & mode
) const {
    
}

void TableBase::_writeFits(
    std::string const & name,
    LayoutMapper const & mapper,
    CONST_PTR(daf::base::PropertySet) const & metadata,
    std::string const & mode
) const {
    
}

}}} // namespace lsst::afw::table
