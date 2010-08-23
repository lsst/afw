#if !defined(LSST_AFW_DETECTION_SHAPE_H)
#define LSST_AFW_DETECTION_SHAPE_H 1

#include "lsst/afw/detection/Measurement.h"

namespace lsst { namespace afw { namespace detection {
/**
 * A version of Measurement designed to support shape measurements
 */
class Shape;

class Shape : public Measurement<Shape> {
protected:
    /// The quantities that the base-class Shape knows how to measure
    /// These values will be used as an index into Measurement::_data
    ///
    /// NVALUE is used by subclasses to add more quantities that they care about
    enum { X=0, X_ERR, Y, Y_ERR,
           IXX, IXX_ERR, IXY, IXY_ERR, IYY, IYY_ERR,
           NVALUE };
public:
    typedef boost::shared_ptr<Shape> Ptr;
    typedef boost::shared_ptr<Shape const> ConstPtr;

    /// Ctor
    Shape() : Measurement<Shape>()
    {
        init();                         // This allocates space for fields added by defineSchema
    }
    /// Ctor
    Shape(double x, double xErr, double y, double yErr,
          double ixx, double ixxErr, double ixy, double ixyErr, double iyy, double iyyErr)
    {
        init();                         // This allocates space for fields added by defineSchema
        set<X>(x);                      // ... if you don't, these set calls will fail an assertion
        set<X_ERR>(xErr);               // the type of the value must match the schema
        set<Y>(y);
        set<Y_ERR>(yErr);

        set<IXX>(ixx);
        set<IXX_ERR>(ixxErr);
        set<IXY>(ixy);
        set<IXY_ERR>(ixyErr);
        set<IYY>(iyy);
        set<IYY_ERR>(iyyErr);
    }

    /// Add desired members to the schema
    virtual void defineSchema(Schema::Ptr schema) {
        schema->add(SchemaEntry("x",       X,         Schema::DOUBLE, 1, "pixel"));
        schema->add(SchemaEntry("xErr",    X_ERR,     Schema::DOUBLE, 1, "pixel"));
        schema->add(SchemaEntry("y",       Y,         Schema::DOUBLE, 1, "pixel"));
        schema->add(SchemaEntry("yErr",    Y_ERR,     Schema::DOUBLE, 1, "pixel"));

        schema->add(SchemaEntry("ixx",     IXX,       Schema::DOUBLE, 1, "pixel^2"));
        schema->add(SchemaEntry("ixxErr",  IXX_ERR,   Schema::DOUBLE, 1, "pixel^2"));
        schema->add(SchemaEntry("ixy",     IXY,       Schema::DOUBLE, 1, "pixel^2"));
        schema->add(SchemaEntry("ixyErr",  IXY_ERR,   Schema::DOUBLE, 1, "pixel^2"));
        schema->add(SchemaEntry("iyy",     IYY,       Schema::DOUBLE, 1, "pixel^2"));
        schema->add(SchemaEntry("iyyErr",  IYY_ERR,   Schema::DOUBLE, 1, "pixel^2"));
    }
    
    /// Return the x-moment
    double getX() const {
        return Measurement<Shape>::get<Shape::X, double>();
    }
    /// Return the error in the x-moment
    double getXErr() const {
        return Measurement<Shape>::get<Shape::X_ERR, double>();
    }
    /// Return the y-moment
    double getY() const {
        return Measurement<Shape>::get<Shape::Y, double>();
    }
    /// Return the error in the y-moment
    double getYErr() const {
        return Measurement<Shape>::get<Shape::Y_ERR, double>();
    }
    /// Return the xx-moment
    double getIxx() const {
        return Measurement<Shape>::get<Shape::IXX, double>();
    }
    /// Return the error in the xx-moment
    double getIxxErr() const {
        return Measurement<Shape>::get<Shape::IXX_ERR, double>();
    }
    /// Return the xx-moment
    double getIxy() const {
        return Measurement<Shape>::get<Shape::IXY, double>();
    }
    /// Return the error in the xy-moment
    double getIxyErr() const {
        return Measurement<Shape>::get<Shape::IXY_ERR, double>();
    }
    /// Return the yy-moment
    double getIyy() const {
        return Measurement<Shape>::get<Shape::IYY, double>();
    }
    /// Return the error in the yy-moment
    double getIyyErr() const {
        return Measurement<Shape>::get<Shape::IYY_ERR, double>();
    }

    virtual ::std::ostream &output(std::ostream &os) const {
        return os << "(" << getX() << "+-" << getXErr() << ", " << getY() << "+-" << getYErr() << ")";
    }
};
}}}

#endif
