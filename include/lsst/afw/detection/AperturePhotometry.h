#if !defined(LSST_AFW_DETECTION_APERTUREPHOTOMETRY_H)
#define LSST_AFW_DETECTION_APERTUREPHOTOMETRY_H 1

#include <boost/serialization/export.hpp>

#include "lsst/afw/detection/Photometry.h"

namespace lsst { namespace afw { namespace detection {

class AperturePhotometry : public Photometry
{
    enum { RADIUS = Photometry::NVALUE,
           NVALUE };
public:
    typedef boost::shared_ptr<AperturePhotometry> Ptr;
    typedef boost::shared_ptr<AperturePhotometry const> ConstPtr;

    /// Ctor
    AperturePhotometry(double flux, double fluxErr, double radius) : Photometry() {
        // XXX Photometry() and Measurement() have called init() too, but they don't know they right type,
        // and hence we have to call init() over again....  Wish there was a simple way not to have to do this.
        // Oh, this is just ticket #1675.  Definitely leave that to the Great Cleanup.
        init();
        set<FLUX>(flux);
        set<FLUX_ERR>(fluxErr);
        set<RADIUS>(radius);
    }
    AperturePhotometry() : Photometry() {
        init();
    }

    /// Add desired fields to the schema
    virtual void defineSchema(lsst::afw::detection::Schema::Ptr schema) {
        // XXX Can I avoid clearing the schema, and just extend what's in Photometry?
        schema->clear();
        schema->add(lsst::afw::detection::SchemaEntry("flux",    FLUX,    
                                                      lsst::afw::detection::Schema::DOUBLE, 1));
        schema->add(lsst::afw::detection::SchemaEntry("fluxErr", FLUX_ERR, 
                                                      lsst::afw::detection::Schema::DOUBLE, 1));
        schema->add(lsst::afw::detection::SchemaEntry("radius",  RADIUS,   
                                                      lsst::afw::detection::Schema::DOUBLE, 1, "pixels"));
    }

    virtual PTR(Photometry) clone() const {
        if (empty()) {
            return boost::make_shared<AperturePhotometry>(getFlux(), getFluxErr(), getRadius());
        }
        return Measurement<Photometry>::clone();
    }

    static Ptr null() {
        double const NaN = std::numeric_limits<double>::quiet_NaN();
        return boost::make_shared<AperturePhotometry>(NaN, NaN, NaN);
    }

    virtual std::vector<double> getFluxes() const {
        std::vector<double> fluxes;
        if (empty()) {
            fluxes.push_back(get<FLUX, double>());
        } else {
            for (const_iterator i = begin(); i != end(); ++i) {
                PTR(AperturePhotometry) phot = boost::dynamic_pointer_cast<AperturePhotometry, Photometry>(*i);
                if (phot->empty()) {
                    fluxes.push_back(phot->get<FLUX, double>());
                } else {
                    std::vector<double> const& more = phot->getFluxes();
                    for (std::vector<double>::const_iterator k = more.begin(); k != more.end(); ++k) {
                        fluxes.push_back(*k);
                    }
                }
            }
        }
        return fluxes;
    }

    virtual double getRadius() const { return get<RADIUS, double>(); }

    virtual PTR(Photometry) average(void) {
        if (empty()) {
            return clone();
        }
        typedef std::vector<ConstPtr> Group;
        typedef std::map<double, Group> GroupMap;
        std::map<double, Group> groups;
        for (iterator iter = begin(); iter != end(); ++iter) {
            PTR(AperturePhotometry) phot = boost::dynamic_pointer_cast<AperturePhotometry, Photometry>(*iter);
            double const radius = phot->getRadius();
            GroupMap::const_iterator mapIter = groups.find(radius);
            if (mapIter == groups.end()) {
                groups[radius] = Group();
            }
            groups[radius].push_back(phot);
        }
        PTR(AperturePhotometry) averages = boost::make_shared<AperturePhotometry>();
        for (GroupMap::iterator groupIter = groups.begin(); groupIter != groups.end(); ++groupIter) {
            double const radius = groupIter->first;
            Group const group = groupIter->second;
            double sum = 0.0, sumWeight = 0.0;
            for (Group::const_iterator grpIter = group.begin(); grpIter != group.end(); ++grpIter) {
                CONST_PTR(AperturePhotometry) phot = *grpIter;
                double flux = phot->getFlux();
                double fluxErr = phot->getFluxErr();
                double weight = 1.0 / (fluxErr * fluxErr);
                sum += flux * weight;
                sumWeight += weight;
            }
            double const flux = sum / sumWeight;
            double const fluxErr = ::sqrt(1.0 / sumWeight);
            averages->add(boost::make_shared<AperturePhotometry>(flux, fluxErr, radius));
        }
        return averages;
    }

private:
    LSST_SERIALIZE_PARENT(lsst::afw::detection::Photometry)
};

}}}

LSST_REGISTER_SERIALIZER(lsst::afw::detection::AperturePhotometry)

#endif
