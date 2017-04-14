// -*- lsst-c++ -*-

#include <map>

#include "lsst/base/ModuleImporter.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/fits.h"

namespace lsst { namespace afw { namespace table { namespace io {

// ----- Persistable ----------------------------------------------------------------------------------------

void Persistable::writeFits(fits::Fits & fitsfile) const {
    OutputArchive archive;
    archive.put(this);
    archive.writeFits(fitsfile);
}

void Persistable::writeFits(std::string const & fileName, std::string const & mode) const {
    fits::Fits fitsfile(fileName, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile);
}

void Persistable::writeFits(fits::MemFileManager & manager, std::string const & mode) const {
    fits::Fits fitsfile(manager, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile);
}

std::string Persistable::getPersistenceName() const { return std::string(); }

std::string Persistable::getPythonModule() const { return std::string(); }

void Persistable::write(OutputArchiveHandle &) const {
    assert(!isPersistable());
    throw LSST_EXCEPT(
        pex::exceptions::LogicError,
        "afw::table-based persistence is not supported for this object."
    );
}

std::shared_ptr<Persistable> Persistable::_readFits(std::string const & fileName, int hdu) {
    fits::Fits fitsfile(fileName, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    fitsfile.setHdu(hdu);
    return _readFits(fitsfile);
}

std::shared_ptr<Persistable> Persistable::_readFits(fits::MemFileManager & manager, int hdu) {
    fits::Fits fitsfile(manager, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    fitsfile.setHdu(hdu);
    return _readFits(fitsfile);
}

std::shared_ptr<Persistable> Persistable::_readFits(fits::Fits & fitsfile) {
    InputArchive archive = InputArchive::readFits(fitsfile);
    return archive.get(1); // the first object saved always has id=1
}

// ----- PersistableFactory ---------------------------------------------------------------------------------

namespace {

typedef std::map<std::string,PersistableFactory const *> RegistryMap;

RegistryMap & getRegistry() {
    static RegistryMap instance;
    return instance;
}

} // anonymous

PersistableFactory::PersistableFactory(std::string const & name) {
    getRegistry()[name] = this;
}

PersistableFactory const & PersistableFactory::lookup(std::string const & name, std::string const & module) {
    RegistryMap::const_iterator i = getRegistry().find(name);
    if (i == getRegistry().end()) {
        if (!module.empty()) {
            bool success = base::ModuleImporter::import(module);
            if (!success) {
                throw LSST_EXCEPT(
                    pex::exceptions::NotFoundError,
                    (boost::format("PersistableFactory with name '%s' not found, and import of module "
                                   "'%s' failed (possibly because Python calls were not available from C++).")
                     % name % module).str()
                );
            }
            i = getRegistry().find(name);
            if (i == getRegistry().end()) {
                throw LSST_EXCEPT(
                    pex::exceptions::LogicError,
                    (boost::format("PersistableFactory with name '%s' not found even after successful import "
                                   "of module '%s'.  Please report this as a bug in the persistence "
                                   "implementation for this object.")
                     % name % module).str()
                );
            }
        } else {
            throw LSST_EXCEPT(
                pex::exceptions::LogicError,
                (boost::format("PersistableFactory with name '%s' not found, and no Python module to import "
                               "was provided.  Please report this as a bug in the persistence implementation "
                               "for this object.")
                 % name).str()
            );
        }
    }
    return *i->second;
}

}}}} // namespace lsst::afw::table::io
