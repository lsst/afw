// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_CatalogVector_h_INCLUDED
#define AFW_TABLE_IO_CatalogVector_h_INCLUDED

#include <vector>

#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/BaseRecord.h"

namespace lsst { namespace afw { namespace table { namespace io {

/**
 *  A vector of catalogs used by Persistable.
 *
 *  This should really be thought of as just a typedef, but we can't forward-declare a typedef
 *  to a template class, so we use a trivial subclass instead.  That may seem like a dirty hack,
 *  but it has a huge benefit in keeping compilation times down: it keeps us from needing to
 *  include Catalog.h in Persistable.h,  which otherwise would pull all of the afw::table headers
 *  into the header of any class that wanted to make use of a Persistable subclass.
 *
 *  CatalogVector is also used in such limited circumstances that we don't really have to worry
 *  about the fact that std::vector doesn't have a virtual destructor and that we only have default
 *  and copy constructors for CatalogVector.
 */
class CatalogVector : public std::vector<BaseCatalog> {};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_CatalogVector_h_INCLUDED
