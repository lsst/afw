#if !defined(LSST_AFW_DETECTION_PHOTOMETRY_H)
#define LSST_AFW_DETECTION_PHOTOMETRY_H 1

#include <boost/serialization/export.hpp>

#include "lsst/afw/detection/Measurement.h"

namespace lsst { namespace afw { namespace detection {
/**
 * A version of Measurement designed to support Photometry
 */
class Photometry : public Measurement<Photometry> {
protected:
    /// The quantities that the base-class Photometry knows how to measure
    /// These values will be used as an index into lsst::afw::detection::Measurement::_data
    ///
    /// NVALUE is used by subclasses to add more quantities that they care about
    enum { FLUX=0, FLUX_ERR, NVALUE };
public:
    typedef boost::shared_ptr<Photometry> Ptr;
    typedef boost::shared_ptr<Photometry const> ConstPtr;

    /// Ctor
    Photometry() : Measurement<Photometry>()
    {
        init();                         // This allocates space for fields added by defineSchema
    }
    /// Ctor
    Photometry(double flux, double fluxErr=std::numeric_limits<double>::quiet_NaN()) : Measurement<Photometry>()
    {
        init();                         // This allocates space for everything in the schema

        set<FLUX>(flux);                // ... if you don't, these set calls will fail an assertion
        set<FLUX_ERR>(fluxErr);         // the type of the value must match the schema
    }

    /// Add desired members to the schema
    virtual void defineSchema(lsst::afw::detection::Schema::Ptr schema) {
        schema->add(lsst::afw::detection::SchemaEntry("flux", FLUX,
                                                      lsst::afw::detection::Schema::DOUBLE));
        schema->add(lsst::afw::detection::SchemaEntry("fluxErr", FLUX_ERR,
                                                      lsst::afw::detection::Schema::DOUBLE, 1));
    }

    virtual Ptr clone() const {
        if (empty()) {
            return boost::make_shared<Photometry>(getFlux(), getFluxErr());
        }
        return Measurement<Photometry>::clone();
    }

    /// Return the number of fluxes available (> 1 iff an array)
    virtual int getNFlux() const {
        return 1;
    }

    /// Return any flag from the measurement algorithm
    virtual boost::int64_t getFlag() const {
        return 0;
    }

    /// Return the flux
    virtual double getFlux() const {
        return lsst::afw::detection::Measurement<Photometry>::get<FLUX, double>();
    }
    /// Return the flux (if an array)
    virtual double getFlux(int i) const {
        return lsst::afw::detection::Measurement<Photometry>::get<FLUX, double>(i);
    }
    /// Return the error in the flux
    virtual double getFluxErr() const {
        return lsst::afw::detection::Measurement<Photometry>::get<FLUX_ERR, double>();
    }
    /// Return some extra parameter of the measurement
    virtual double getParameter(int=0   ///< Desired parameter
                               ) const {
        return std::numeric_limits<double>::quiet_NaN();
    }
    /// Return the error in the flux (if an array)
    virtual double getFluxErr(int i) const {
        return lsst::afw::detection::Measurement<Photometry>::get<FLUX_ERR, double>(i);
    }

    virtual ::std::ostream &output(std::ostream &os) const;
    virtual Photometry::Ptr average() const;

private:
    LSST_SERIALIZE_PARENT(lsst::afw::detection::Measurement<Photometry>)
};
}}}

LSST_REGISTER_SERIALIZER(lsst::afw::detection::Photometry)

#endif
