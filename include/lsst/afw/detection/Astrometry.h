#if !defined(LSST_AFW_DETECTION_ASTROMETRY_H)
#define LSST_AFW_DETECTION_ASTROMETRY_H 1

#include <boost/serialization/export.hpp>

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

    /// Ctor
    Astrometry() : Measurement<Astrometry>()
    {
        init();                         // This allocates space for fields added by defineSchema
    }
    /// Ctor
    Astrometry(double x, double xErr, double y, double yErr) : Measurement<Astrometry>()
    {
        init();                         // This allocates space for fields added by defineSchema
        set<X>(x);                      // ... if you don't, these set calls will fail an assertion
        set<X_ERR>(xErr);               // the type of the value must match the schema
        set<Y>(y);
        set<Y_ERR>(yErr);
    }

    /// Add desired members to the schema
    virtual void defineSchema(Schema::Ptr schema) {
        schema->add(SchemaEntry("x", X, Schema::DOUBLE, 1, "pixel"));
        schema->add(SchemaEntry("xErr", X_ERR, Schema::DOUBLE, 1, "pixel"));
        schema->add(SchemaEntry("y", Y, Schema::DOUBLE, 1, "pixel"));
        schema->add(SchemaEntry("yErr", Y_ERR, Schema::DOUBLE, 1, "pixel"));
    }
    
    virtual Ptr clone() const {
        if (empty()) {
            return boost::make_shared<Astrometry>(getX(), getXErr(), getY(), getYErr());
        }
        return Measurement<Astrometry>::clone();
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

    virtual Astrometry::Ptr average() const {
        double xSum = 0.0, xSumWeight = 0.0, ySum = 0.0, ySumWeight = 0.0;
        for (const_iterator iter = begin(); iter != end(); ++iter) {
            ConstPtr astrom = *iter;
            if (!astrom->empty()) {
                astrom = astrom->average();
            }
            double xWeight = 1.0 / (astrom->getXErr() * astrom->getXErr());
            double yWeight = 1.0 / (astrom->getYErr() * astrom->getYErr());
            xSum += astrom->getX() * xWeight;
            ySum += astrom->getY() * yWeight;
            xSumWeight += xWeight;
            ySumWeight += yWeight;
        }
        return Ptr(new Astrometry(xSum / xSumWeight, 1.0 / xSumWeight, ySum / ySumWeight, 1.0 / ySumWeight));
    }
private:
    LSST_SERIALIZE_PARENT(lsst::afw::detection::Measurement<Astrometry>)
};

}}}

LSST_REGISTER_SERIALIZER(lsst::afw::detection::Astrometry)

#endif
