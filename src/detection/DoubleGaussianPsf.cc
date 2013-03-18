// -*- LSST-C++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2013 LSST Corporation.
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

#include <cmath>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection/DoubleGaussianPsf.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"

namespace lsst { namespace afw { namespace detection {

namespace {

// Read-only singleton struct containing the schema and keys that a double-Gaussian Psf is mapped
// to in record persistence.
struct DoubleGaussianPsfPersistenceHelper : private boost::noncopyable {
    afw::table::Schema schema;
    afw::table::Key< afw::table::Point<int> > dimensions;
    afw::table::Key<double> sigma1;
    afw::table::Key<double> sigma2;
    afw::table::Key<double> b;

    static DoubleGaussianPsfPersistenceHelper const & get() {
        static DoubleGaussianPsfPersistenceHelper instance;
        return instance;
    }

private:
    DoubleGaussianPsfPersistenceHelper() :
        schema(),
        dimensions(
            schema.addField< afw::table::Point<int> >("dimensions", "width/height of kernel", "pixels")
        ),
        sigma1(schema.addField<double>("sigma1", "radius of inner Gaussian", "pixels")),
        sigma2(schema.addField<double>("sigma2", "radius of outer Gaussian", "pixels")),
        b(schema.addField<double>("b", "central amplitude of outer Gaussian (inner amplitude == 1)"))
    {
        schema.getCitizen().markPersistent();
    }
};

class DoubleGaussianPsfFactory : public table::io::PersistableFactory {
public:

    virtual PTR(table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        static DoubleGaussianPsfPersistenceHelper const & keys = DoubleGaussianPsfPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        table::BaseRecord const & record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == keys.schema);
        return boost::make_shared<DoubleGaussianPsf>(
            record.get(keys.dimensions.getX()),
            record.get(keys.dimensions.getY()),
            record.get(keys.sigma1),
            record.get(keys.sigma2),
            record.get(keys.b)
        );
    }

    DoubleGaussianPsfFactory(std::string const & name) : table::io::PersistableFactory(name) {}
};

// Helper function for ctor: need to construct the kernel to pass to KernelPsf, because we
// can't change it after construction.
PTR(math::Kernel) makeDoubleGaussianKernel(
    int width, int height, double sigma1, double & sigma2, double b
) {
    if (b == 0.0 && sigma2 == 0.0) {
        sigma2 = 1.0;                  // avoid 0/0 at centre of Psf
    }
    if (sigma1 <= 0 || sigma2 <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("sigma may not be 0: %g, %g") % sigma1 % sigma2).str());
    }
    math::DoubleGaussianFunction2<double> dg(sigma1, sigma2, b);
    PTR(math::Kernel) kernel(new math::AnalyticKernel(width, height, dg));
    return kernel;
}

std::string getDoubleGaussianPsfPersistenceName() { return "DoubleGaussianPsf"; }

DoubleGaussianPsfFactory registration(getDoubleGaussianPsfPersistenceName());

} // anonymous

DoubleGaussianPsf::DoubleGaussianPsf(int width, int height, double sigma1, double sigma2, double b) :
    KernelPsf(makeDoubleGaussianKernel(width, height, sigma1, sigma2, b)),
    _sigma1(sigma1), _sigma2(sigma2), _b(b)
{}

std::string DoubleGaussianPsf::getPersistenceName() const { return getDoubleGaussianPsfPersistenceName(); }

void DoubleGaussianPsf::write(OutputArchiveHandle & handle) const {
    static DoubleGaussianPsfPersistenceHelper const & keys = DoubleGaussianPsfPersistenceHelper::get();
    afw::table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    PTR(afw::table::BaseRecord) record = catalog.addNew();
    (*record).set(keys.dimensions.getX(), getKernel()->getWidth());
    (*record).set(keys.dimensions.getY(), getKernel()->getHeight());
    (*record).set(keys.sigma1, getSigma1());
    (*record).set(keys.sigma2, getSigma2());
    (*record).set(keys.b, getB());
    handle.saveCatalog(catalog);
}

}}} // namespace lsst::afw::detection

