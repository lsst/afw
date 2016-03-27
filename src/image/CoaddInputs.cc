// -*- LSST-C++ -*- // fixed format comment for emacs
/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#include "lsst/afw/image/CoaddInputs.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"

namespace lsst { namespace afw { namespace image {

namespace {

class CoaddInputsFactory : public table::io::PersistableFactory {
public:

    virtual PTR(table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 2);
        PTR(CoaddInputs) result = boost::make_shared<CoaddInputs>();
        result->visits = table::ExposureCatalog::readFromArchive(archive, catalogs.front());
        result->ccds = table::ExposureCatalog::readFromArchive(archive, catalogs.back());
        return result;
    }

    CoaddInputsFactory(std::string const & name) : table::io::PersistableFactory(name) {}

};

CoaddInputsFactory registration("CoaddInputs");

} // anonymous

CoaddInputs::CoaddInputs() : visits(), ccds() {}

CoaddInputs::CoaddInputs(table::Schema const & visitSchema, table::Schema const & ccdSchema) :
    visits(visitSchema), ccds(ccdSchema)
{}

CoaddInputs::CoaddInputs(table::ExposureCatalog const & visits_, table::ExposureCatalog const & ccds_) :
    visits(visits_), ccds(ccds_)
{}

bool CoaddInputs::isPersistable() const { return true; }

std::string CoaddInputs::getPersistenceName() const { return "CoaddInputs"; }

std::string CoaddInputs::getPythonModule() const { return "lsst.afw.image"; }

void CoaddInputs::write(OutputArchiveHandle & handle) const {
    visits.writeToArchive(handle, true); // true == permissive - just ignore Psfs, Wcss that can't be saved
    ccds.writeToArchive(handle, true);
}

}}} // namespace lsst::afw::image
