#if !defined(LSST_AFW_DETECTION_ASTROMETRY_H)
#define LSST_AFW_DETECTION_ASTROMETRY_H 1

#include "lsst/afw/detection/Measurement.h"

namespace lsst { namespace afw { namespace detection {
/**
 * A version of Measurement designed to support Astrometry
 */
class Astrometry;

class Astrometry : public Measurement<Astrometry> {
protected:
    /// The quantities that the base-class Astrometry knows how to measure
    /// These values will be used as an index into Measurement::_data
    ///
    /// NVALUE is used by subclasses to add more quantities that they care about
    enum { X=0, X_ERR, Y, Y_ERR, NVALUE };
public:
    typedef boost::shared_ptr<Astrometry> Ptr;
    typedef boost::shared_ptr<Astrometry const> ConstPtr;

    /// Add desired members to the schema
    virtual void defineSchema(Schema::Ptr schema) {
        schema->add(SchemaEntry("x", X, Schema::DOUBLE, 1, "pixel"));
        schema->add(SchemaEntry("xErr", X_ERR, Schema::DOUBLE, 1, "pixel"));
        schema->add(SchemaEntry("y", Y, Schema::DOUBLE, 1, "pixel"));
        schema->add(SchemaEntry("yErr", Y_ERR, Schema::DOUBLE, 1, "pixel"));
    }
    
    /// Return the x-centroid
    double getX() const {
        return Measurement<Astrometry>::get<Astrometry::X, double>();
    }
    /// Return the error in the x-centroid
    double getXErr() const {
        return Measurement<Astrometry>::get<Astrometry::X_ERR, double>();
    }
    /// Return the y-centroid
    double getY() const {
        return Measurement<Astrometry>::get<Astrometry::Y, double>();
    }
    /// Return the error in the y-centroid
    double getYErr() const {
        return Measurement<Astrometry>::get<Astrometry::Y_ERR, double>();
    }

    virtual ::std::ostream &output(std::ostream &os) const {
        return os << "(" << getX() << "+-" << getXErr() << ", " << getY() << "+-" << getYErr() << ")";
    }
};
}}}

#endif
