// -*- LSST-C++ -*- // fixed format comment for emacs
/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#include "lsst/afw/image/CoaddInputs.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/OutputArchive.h"

namespace lsst { namespace afw { namespace image {

namespace {

// A std::for_each functor to copy BaseRecords into ExposureRecords
// (this should probably be something Catalogs should be able to do,
// but coming up with a good API for that is tricky enough that I
// don't want to tackle it right now).
struct CopyRecords {

    void operator()(afw::table::BaseRecord const & input) const {
        out->addNew()->assign(input);
    }

    static void apply(afw::table::BaseCatalog const & input, afw::table::ExposureCatalog & output) {
        CopyRecords f = { &output };
        output.reserve(input.size());
        std::for_each(input.begin(), input.end(), f);
    }

    afw::table::ExposureCatalog * out;
};

class CoaddInputsFactory : public table::io::PersistableFactory {
public:

    virtual PTR(table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 2);
        PTR(CoaddInputs) result = boost::make_shared<CoaddInputs>(
            catalogs.front().getSchema(), catalogs.back().getSchema()
        );
        CopyRecords::apply(catalogs.front(), result->visits);
        CopyRecords::apply(catalogs.back(), result->ccds);
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

void CoaddInputs::write(OutputArchiveHandle & handle) const {
    afw::table::BaseCatalog visitOut = handle.makeCatalog(visits.getSchema());
    visitOut.assign(visits.begin(), visits.end(), true); // true == deep copy
    handle.saveCatalog(visitOut, true); // true == permissive - just ignore Psfs, Wcss that can't be saved
    afw::table::BaseCatalog ccdOut = handle.makeCatalog(ccds.getSchema());
    ccdOut.assign(ccds.begin(), ccds.end(), true);
    handle.saveCatalog(ccdOut, true);
}

}}} // namespace lsst::afw::image
