// -*- LSST-C++ -*-
#include <cmath>
#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection/DoubleGaussianPsf.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"

namespace afwMath = lsst::afw::math;

namespace lsst {
namespace afw {
namespace detection {

DoubleGaussianPsf::DoubleGaussianPsf(int width, int height, double sigma1, double sigma2, double b) :
    KernelPsf(), _sigma1(sigma1), _sigma2(sigma2), _b(b)
{
    if (b == 0.0 && sigma2 == 0.0) {
        sigma2 = 1.0;                  // avoid 0/0 at centre of Psf
    }

    if (sigma1 <= 0 || sigma2 <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("sigma may not be 0: %g, %g") % sigma1 % sigma2).str());
    }
    
    if (width > 0) {
        afwMath::DoubleGaussianFunction2<double> dg(sigma1, sigma2, b);
        setKernel(afwMath::Kernel::Ptr(new afwMath::AnalyticKernel(width, height, dg)));
    }
}

namespace {

// We need to make an instance here so as to register it
volatile bool isInstance =
    Psf::registerMe<DoubleGaussianPsf, boost::tuple<int, int, double, double, double> >("DoubleGaussian");

// Read-only singleton struct containing the schema and keys that a double-Gaussian Psf is mapped
// to in record persistence.
struct DoubleGaussianPsfSchema : private boost::noncopyable {
    afw::table::Schema schema;
    afw::table::Key<int> width;
    afw::table::Key<int> height;
    afw::table::Key<double> sigma1;
    afw::table::Key<double> sigma2;
    afw::table::Key<double> b;

    static DoubleGaussianPsfSchema const & get() {
        static DoubleGaussianPsfSchema instance;
        return instance;
    }

private:
    DoubleGaussianPsfSchema() :
        schema(),
        width(schema.addField<int>("width", "number of columns in realization of Psf", "pixels")),
        height(schema.addField<int>("height", "number of rows in realization of Psf", "pixels")),
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
        static DoubleGaussianPsfSchema const & keys = DoubleGaussianPsfSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        table::BaseRecord const & record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == keys.schema);
        return boost::make_shared<DoubleGaussianPsf>(
            record.get(keys.width),
            record.get(keys.height),
            record.get(keys.sigma1),
            record.get(keys.sigma2),
            record.get(keys.b)
        );
    }

    DoubleGaussianPsfFactory(std::string const & name) : table::io::PersistableFactory(name) {}

};

DoubleGaussianPsfFactory registration("DoubleGaussianPsf");

} // anonymous

std::string DoubleGaussianPsf::getPersistenceName() const { return "DoubleGaussianPsf"; }

void DoubleGaussianPsf::write(OutputArchiveHandle & handle) const {
    static DoubleGaussianPsfSchema const & keys = DoubleGaussianPsfSchema::get();
    afw::table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    PTR(afw::table::BaseRecord) record = catalog.addNew();
    (*record)[keys.width] = getKernel()->getWidth();
    (*record)[keys.height] = getKernel()->getHeight();
    (*record)[keys.sigma1] = getSigma1();
    (*record)[keys.sigma2] = getSigma2();
    (*record)[keys.b] = getB();
    handle.saveCatalog(catalog);
}

}}} // namespace lsst::afw::detection

