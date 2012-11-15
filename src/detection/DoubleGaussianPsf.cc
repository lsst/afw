// -*- LSST-C++ -*-
#include <cmath>
#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection/DoubleGaussianPsf.h"
#include "lsst/afw/detection/RecordGeneratorPsfFactory.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/image/ImageUtils.h"

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

class DoubleGaussianPsfRecordOutputGenerator : public afw::table::RecordOutputGenerator {
public:

    virtual void fill(afw::table::BaseRecord & record) {
        DoubleGaussianPsfSchema const & keys = DoubleGaussianPsfSchema::get();
        if (record.getSchema() != keys.schema) {
            throw LSST_EXCEPT(
                pex::exceptions::LogicErrorException,
                "Incorrect schema for Psf persistence"
            );
        }
        record[keys.width] = _psf->getKernel()->getWidth();
        record[keys.height] = _psf->getKernel()->getHeight();
        record[keys.sigma1] = _psf->getSigma1();
        record[keys.sigma2] = _psf->getSigma2();
        record[keys.b] = _psf->getB();
    }

    explicit DoubleGaussianPsfRecordOutputGenerator(DoubleGaussianPsf const * psf) :
        afw::table::RecordOutputGenerator(DoubleGaussianPsfSchema::get().schema),
        _psf(psf)
        {}

private:
    DoubleGaussianPsf const * _psf;
};

class DoubleGaussianPsfRecordGeneratorPsfFactory : public RecordGeneratorPsfFactory {
public:

    virtual PTR(Psf) operator()(table::RecordInputGeneratorSet const & inputs) const {
        CONST_PTR(afw::table::BaseRecord) record = inputs.generators.front()->next();
        DoubleGaussianPsfSchema const & keys = DoubleGaussianPsfSchema::get();
        if (record->getSchema() != keys.schema) {
            throw LSST_EXCEPT(
                pex::exceptions::LogicErrorException,
                "Incorrect schema for Psf persistence"
            );
        }
        return boost::make_shared<DoubleGaussianPsf>(
            record->get(keys.width),
            record->get(keys.height),
            record->get(keys.sigma1),
            record->get(keys.sigma2),
            record->get(keys.b)
        );
    }

    DoubleGaussianPsfRecordGeneratorPsfFactory(std::string const & name) : RecordGeneratorPsfFactory(name) {}

};

DoubleGaussianPsfRecordGeneratorPsfFactory registration("DoubleGaussian");

} // anonymous

afw::table::RecordOutputGeneratorSet DoubleGaussianPsf::writeToRecords() const {
    afw::table::RecordOutputGeneratorSet result("DoubleGaussian");
    result.generators.resize(1);
    result.generators.front() = boost::make_shared<DoubleGaussianPsfRecordOutputGenerator>(this);
    return result;
}

}}} // namespace lsst::afw::detection

