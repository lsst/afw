// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_ArchiveInputSchema_h_INCLUDED
#define AFW_TABLE_IO_ArchiveInputSchema_h_INCLUDED

/*
 *  Private singleton shared by the two archive classes; should only be included in their definition
 *  source files.
 */

#include "lsst/base.h"
#include "lsst/afw/table/Schema.h"

namespace lsst {
namespace afw {
namespace table {
namespace io {

/**
 *  @brief Schema for the index catalog that specifies where objects are stored in the
 *         data catalogs.
 *
 *  Each row in the index catalog corresponds to a combination of an object and a
 *  data catalog, and contains the range of rows used by that object in the catalog
 *  (in the 'row0' and 'nrows' fields).  The 'cat.archive' field indicates which
 *  catalog in the archive the index entry refers to (where 0 is the index itself,
 *  and 1 is the first data catalog), and 'cat.persistable' refers to which catalog
 *  the index entry refers to from the perspective of the saved object.
 *
 *  An object may thus have more than one row in the index, but the 'id' and 'name'
 *  fields must be the same for all index entires that correspond to a single object.
 *  These contain, respectively, the unique ID returned by OutputArchive::put and the
 *  name returned by Persistable::getPersistenceName() and used by InputArchive to
 *  look up a PersistableFactory in the registry.
 */
struct ArchiveIndexSchema {
    Schema schema;
    Key<int> id;
    Key<int> catArchive;      // 'cat.archive' in schema
    Key<int> catPersistable;  // 'cat.persistable' in schema
    Key<int> row0;
    Key<int> nRows;  // 'nrows' in schema
    Key<std::string> name;
    Key<std::string> module;

    static int const MAX_NAME_LENGTH = 64;
    static int const MAX_MODULE_LENGTH = 64;

    /**
     *  Special value used for catArchive, catPersistable, and row0 when an
     *  object with no state is saved.
     */
    static constexpr int const NO_CATALOGS_SAVED = -1;

    /// Return the singleton instance.
    static ArchiveIndexSchema const& get();

    // No copying
    ArchiveIndexSchema(const ArchiveIndexSchema&) = delete;
    ArchiveIndexSchema& operator=(const ArchiveIndexSchema&) = delete;

    // No moving
    ArchiveIndexSchema(ArchiveIndexSchema&&) = delete;
    ArchiveIndexSchema& operator=(ArchiveIndexSchema&&) = delete;

private:
    ArchiveIndexSchema();
};
}  // namespace io
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_IO_ArchiveInputSchema_h_INCLUDED
