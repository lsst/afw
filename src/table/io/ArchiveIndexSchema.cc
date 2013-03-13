// -*- lsst-c++ -*-

#include "lsst/afw/table/io/ArchiveIndexSchema.h"

namespace lsst { namespace afw { namespace table { namespace io {

int const ArchiveIndexSchema::MAX_NAME_LENGTH;

ArchiveIndexSchema const & ArchiveIndexSchema::get() {
    static ArchiveIndexSchema instance;
    return instance;
}

ArchiveIndexSchema::ArchiveIndexSchema() :
    schema(),
    id(schema.addField<int>(
           "id", "Archive ID of the persistable object that owns the records pointed at by this entry"
       )),
    catArchive(
        schema.addField<int>(
            "cat.archive", 
            "index of the catalog this entry points to, from the perspective of the archive"
        )),
    catPersistable(
        schema.addField<int>(
            "cat.persistable",
            "index of the catalog this entry points to, from the perspective of the Persistable"
        )),
    row0(schema.addField<int>(
             "row0", "first row used by the persistable object in this catalog"
         )),
    nRows(schema.addField<int>(
              "nrows", "number of rows used by the persistable object in this catalog"
          )),
    name(schema.addField<std::string>(
             "name", "unique name for the persistable object's class", MAX_NAME_LENGTH
         )),
    module(schema.addField<std::string>(
             "module", "Python module that should be imported to register the object's factory",
             MAX_MODULE_LENGTH
         ))
{
    schema.getCitizen().markPersistent();
}

}}}} // namespace lsst::afw::table::io
