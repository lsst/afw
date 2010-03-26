#if !defined(LSST_AFW_DETECTION_PHOTOMETRY_H)
#define LSST_AFW_DETECTION_PHOTOMETRY_H 1

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

    /// Add desired members to the schema
    virtual void defineSchema(lsst::afw::detection::Schema::Ptr schema) {
        schema->add(lsst::afw::detection::SchemaEntry("flux", FLUX,
                                                      lsst::afw::detection::Schema::DOUBLE));
        schema->add(lsst::afw::detection::SchemaEntry("fluxErr", FLUX_ERR,
                                                      lsst::afw::detection::Schema::FLOAT, 1));
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
    virtual float getFluxErr() const {
        return lsst::afw::detection::Measurement<Photometry>::get<FLUX_ERR, float>();
    }
    /// Return the error in the flux (if an array)
    virtual float getFluxErr(int i) const {
        return lsst::afw::detection::Measurement<Photometry>::get<FLUX_ERR, float>(i);
    }

    virtual ::std::ostream &output(std::ostream &os) const;
};
}}}
#endif
