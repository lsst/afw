#if !defined(LSST_AFW_DETECTION_APERTUREPHOTOMETRY_H)
#define LSST_AFW_DETECTION_APERTUREPHOTOMETRY_H 1

#include <boost/serialization/export.hpp>

#include "lsst/pex/exceptions/Runtime.h"
#include "lsst/afw/detection/Photometry.h"

namespace lsst { namespace afw { namespace detection {

struct ApertureFlux {
    ApertureFlux(double radius_,
                 double flux_=std::numeric_limits<double>::quiet_NaN(),
                 double fluxErr_=std::numeric_limits<double>::quiet_NaN()) :
        radius(radius_), flux(flux_), fluxErr(fluxErr_) {}
    double radius, flux, fluxErr;    // type must match defineSchema below
};


class AperturePhotometry : public Photometry
{
    enum { FLUX=Photometry::FLUX,
           FLUX_ERR,
           RADIUS,
           NVALUE };
public:
    typedef boost::shared_ptr<AperturePhotometry> Ptr;
    typedef boost::shared_ptr<AperturePhotometry const> ConstPtr;

    /// Ctor
    AperturePhotometry(ApertureFlux const& fluxes) : Photometry() {
        init();                         // This allocates space for everything in the schema
        set<RADIUS>(fluxes.radius);
        set<FLUX>(fluxes.flux);
        set<FLUX_ERR>(fluxes.fluxErr);
    }
    AperturePhotometry(double radius, double flux, double fluxErr) : Photometry() {
        init();                         // This allocates space for everything in the schema
        set<RADIUS>(radius);
        set<FLUX>(flux);
        set<FLUX_ERR>(fluxErr);
    }
    AperturePhotometry() : Photometry() {
        init();
    }

    /// Add desired fields to the schema
    virtual void defineSchema(Schema::Ptr schema ///< our schema; == _mySchema
                     ) {
        schema->clear();
        schema->add(SchemaEntry("flux",    FLUX,     Schema::DOUBLE, 1));
        schema->add(SchemaEntry("fluxErr", FLUX_ERR, Schema::DOUBLE, 1));
        schema->add(SchemaEntry("radius",  RADIUS,   Schema::DOUBLE, 1, "pixels"));
    }

    virtual PTR(Photometry) clone() const {
        if (empty()) {
            return boost::make_shared<AperturePhotometry>(getRadius(), getFlux(), getFluxErr());
        }
        return Measurement<Photometry>::clone();
    }

    static Ptr null() {
        double const NaN = std::numeric_limits<double>::quiet_NaN();
        return boost::make_shared<AperturePhotometry>(NaN, NaN, NaN);
    }

    virtual double getRadius() const { return get<RADIUS, double>(); }

    virtual PTR(Photometry) average(void) {
        if (empty()) {
            return clone();
        }

        double const NaN = std::numeric_limits<double>::quiet_NaN();

        double sumFlux = 0;
        double sumWeight = 0;
        double radius = NaN;

        for (iterator iter = begin(); iter != end(); ++iter) {
            PTR(AperturePhotometry) phot = boost::dynamic_pointer_cast<AperturePhotometry, Photometry>(*iter);
            if (lsst::utils::isnan(getRadius())) {
                continue;
            }
            if (lsst::utils::isnan(radius)) {
                radius = getRadius();
            } else if (getRadius() != radius) {
                throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                                  (boost::format("Radius doesn't match: %f vs %f") % 
                                   radius % getRadius()).str());
            }
            double const weight = 1.0 / (getFluxErr() * getFluxErr());
            sumFlux += getFlux() * weight;
            sumWeight += weight;
        }
        
        return boost::make_shared<AperturePhotometry>(radius, sumFlux / sumWeight, ::sqrt(1.0 / sumWeight));
    }


private:
    LSST_SERIALIZE_PARENT(lsst::afw::detection::Photometry)
};


/// XXX Template on NRADIUS?
class MultipleAperturePhotometry : public Photometry
{
    enum { NRADIUS = 3 };               // dimension of RADIUS array
    enum { FLUX=Photometry::FLUX,
           FLUX_ERR = FLUX     + NRADIUS,
           RADIUS   = FLUX_ERR + NRADIUS,
           NVALUE   = RADIUS   + NRADIUS };
public:
    typedef boost::shared_ptr<MultipleAperturePhotometry> Ptr;
    typedef boost::shared_ptr<MultipleAperturePhotometry const> ConstPtr;

    /// Ctor
    MultipleAperturePhotometry(std::vector<ApertureFlux> const& fluxes) : Photometry() {
        init();                         // This allocates space for everything in the schema

        int const nflux = fluxes.size();
        assert (nflux <= NRADIUS);      // XXX be nice
        for (int i = 0; i != nflux; ++i) {
            set<RADIUS>(i, fluxes[i].radius);
            set<FLUX>(i, fluxes[i].flux);
            set<FLUX_ERR>(i, fluxes[i].fluxErr);
        }
        // Ensure everything has a value; lest we get boost::any cast problems....
        double const NaN = std::numeric_limits<double>::quiet_NaN();
        for (int i = nflux; i != NRADIUS; ++i) {
            set<RADIUS>(i, NaN);
            set<FLUX>(i, NaN);
            set<FLUX_ERR>(i, NaN);
        }
    }

    MultipleAperturePhotometry() : Photometry() {
        init();
    }

    /// Add desired fields to the schema
    virtual void defineSchema(Schema::Ptr schema ///< our schema; == _mySchema
                     ) {
        schema->clear();
        schema->add(SchemaEntry("flux",    FLUX,     Schema::DOUBLE, NRADIUS));
        schema->add(SchemaEntry("fluxErr", FLUX_ERR, Schema::DOUBLE, NRADIUS));
        schema->add(SchemaEntry("radius",  RADIUS,   Schema::DOUBLE, NRADIUS, "pixels"));
    }

    virtual PTR(Photometry) clone() const {
        if (empty()) {
            return boost::make_shared<MultipleAperturePhotometry>(getFluxes());
        }
        return Measurement<Photometry>::clone();
    }

    static Ptr null() {
        double const NaN = std::numeric_limits<double>::quiet_NaN();
        std::vector<ApertureFlux> nulls;
        for (size_t i = 0; i < NRADIUS; ++i) {
            nulls.push_back(ApertureFlux(NaN, NaN, NaN));
        }
        return boost::make_shared<MultipleAperturePhotometry>(nulls);
    }

    virtual std::vector<ApertureFlux> getFluxes() const {
        std::vector<ApertureFlux> fluxes;
        for (size_t i = 0; i != NRADIUS; ++i) {
            fluxes.push_back(ApertureFlux(get<RADIUS, double>(i), get<FLUX, double>(i), 
                                          get<FLUX_ERR, double>(i)));
        }
        return fluxes;
    }

    virtual PTR(Photometry) average(void) {
        if (empty()) {
            return clone();
        }
        std::vector<double> sumFlux(NRADIUS);
        std::vector<double> sumWeight(NRADIUS);
        std::vector<double> radius(NRADIUS);

        double const NaN = std::numeric_limits<double>::quiet_NaN();

        for (size_t i = 0; i != NRADIUS; ++i) {
            sumFlux[i] = 0;
            sumWeight[i] = 0;
            radius[i] = NaN;
        }

        for (iterator iter = begin(); iter != end(); ++iter) {
            PTR(MultipleAperturePhotometry) phot = 
                boost::dynamic_pointer_cast<MultipleAperturePhotometry, Photometry>(*iter);
            std::vector<ApertureFlux> const& apFlux = getFluxes();
            for (size_t i = 0; i != NRADIUS; ++i) {
                if (lsst::utils::isnan(apFlux[i].radius)) {
                    continue;
                }
                if (lsst::utils::isnan(radius[i])) {
                    radius[i] = apFlux[i].radius;
                } else if (apFlux[i].radius != radius[i]) {
                    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                                      (boost::format("Radius %d doesn't match: %f vs %f") % 
                                       i % radius[i] % apFlux[i].radius).str());
                }
                double const weight = 1.0 / (apFlux[i].fluxErr * apFlux[i].fluxErr);
                sumFlux[i] += apFlux[i].flux * weight;
                sumWeight[i] += weight;
            }
        }

        std::vector<ApertureFlux> apFlux;
        for (size_t i = 0; i != NRADIUS; ++i) {
            apFlux.push_back(ApertureFlux(radius[i], sumFlux[i] / sumWeight[i], ::sqrt(1.0 / sumWeight[i])));
        }

        return boost::make_shared<MultipleAperturePhotometry>(apFlux);
    }

private:
    LSST_SERIALIZE_PARENT(lsst::afw::detection::Photometry)
};

}}}

LSST_REGISTER_SERIALIZER(lsst::afw::detection::AperturePhotometry)
LSST_REGISTER_SERIALIZER(lsst::afw::detection::MultipleAperturePhotometry)

#endif
