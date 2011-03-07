#if !defined(LSST_AFW_DETECTION_SHAPE_H)
#define LSST_AFW_DETECTION_SHAPE_H 1

#include <boost/serialization/export.hpp>

#include "lsst/afw/detection/Measurement.h"

namespace lsst { namespace afw { namespace detection {

namespace {
    double const NaN = std::numeric_limits<double>::quiet_NaN();
}
            
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
           E1, E1_ERR, E2, E2_ERR, SIGMA, SIGMA_ERR,
           PSF_IXX, PSF_IXX_ERR, PSF_IXY, PSF_IXY_ERR, PSF_IYY, PSF_IYY_ERR,
           SHEAR1, SHEAR1_ERR, SHEAR2, SHEAR2_ERR, RESOLUTION,
           STATUS,
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
            
            set<SIGMA>(NaN);
            set<SIGMA_ERR>(NaN);
            
            set<E1>(NaN);
            set<E1_ERR>(NaN);
            set<E2>(NaN);
            set<E2_ERR>(NaN);
            
            set<SHEAR1>(NaN);
            set<SHEAR2>(NaN);
            set<SHEAR1_ERR>(NaN);
            set<SHEAR2_ERR>(NaN);
            
            set<RESOLUTION>(NaN);
            
            set<PSF_IXX>(NaN);
            set<PSF_IXX_ERR>(NaN);
            set<PSF_IXY>(NaN);
            set<PSF_IXY_ERR>(NaN);
            set<PSF_IYY>(NaN);
            set<PSF_IYY_ERR>(NaN);
            set<STATUS>(-1);
            
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
        
        schema->add(SchemaEntry("e1",       E1,        Schema::DOUBLE, 1, "unitless"));
        schema->add(SchemaEntry("e1Err",    E1_ERR,    Schema::DOUBLE, 1, "unitless"));
        schema->add(SchemaEntry("e2",       E2,        Schema::DOUBLE, 1, "unitless"));
        schema->add(SchemaEntry("e2Err",    E2_ERR,    Schema::DOUBLE, 1, "unitless"));

        schema->add(SchemaEntry("shear1",       SHEAR1,        Schema::DOUBLE, 1, "unitless"));
        schema->add(SchemaEntry("shear1Err",    SHEAR1_ERR,    Schema::DOUBLE, 1, "unitless"));
        schema->add(SchemaEntry("shear2",       SHEAR2,        Schema::DOUBLE, 1, "unitless"));
        schema->add(SchemaEntry("shear2Err",    SHEAR2_ERR,    Schema::DOUBLE, 1, "unitless"));

        schema->add(SchemaEntry("resolution",  RESOLUTION,    Schema::DOUBLE, 1, "unitless"));
        
        schema->add(SchemaEntry("sigma",    SIGMA,     Schema::DOUBLE, 1, "pixel"));
        schema->add(SchemaEntry("sigmaErr", SIGMA_ERR, Schema::DOUBLE, 1, "pixel"));

        schema->add(SchemaEntry("psfIxx",     PSF_IXX,       Schema::DOUBLE, 1, "pixel^2"));
        schema->add(SchemaEntry("psfIxxErr",  PSF_IXX_ERR,   Schema::DOUBLE, 1, "pixel^2"));
        schema->add(SchemaEntry("psfIxy",     PSF_IXY,       Schema::DOUBLE, 1, "pixel^2"));
        schema->add(SchemaEntry("psfIxyErr",  PSF_IXY_ERR,   Schema::DOUBLE, 1, "pixel^2"));
        schema->add(SchemaEntry("psfIyy",     PSF_IYY,       Schema::DOUBLE, 1, "pixel^2"));
        schema->add(SchemaEntry("psfIyyErr",  PSF_IYY_ERR,   Schema::DOUBLE, 1, "pixel^2"));

        // this should be a boost::int16_t, which was used in BaseSourceAttributes
        schema->add(SchemaEntry("status",  STATUS,   Schema::INT, 1, "unitless"));
        
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

    /// Return the e1 ellipticity
    double getE1() const {
        return Measurement<Shape>::get<Shape::E1, double>();
    }
    /// Return the error in the e1 ellipticity
    double getE1Err() const {
        return Measurement<Shape>::get<Shape::E1_ERR, double>();
    }
    /// Return the e2 ellipticity
    double getE2() const {
        return Measurement<Shape>::get<Shape::E2, double>();
    }
    /// Return the error in the e2 ellipticity
    double getE2Err() const {
        return Measurement<Shape>::get<Shape::E2_ERR, double>();
    }


    /// Return the shear1
    double getShear1() const {
        return Measurement<Shape>::get<Shape::SHEAR1, double>();
    }
    /// Return the error in the shear1
    double getShear1Err() const {
        return Measurement<Shape>::get<Shape::SHEAR1_ERR, double>();
    }
    /// Return the shear2
    double getShear2() const {
        return Measurement<Shape>::get<Shape::SHEAR2, double>();
    }
    /// Return the error in the shear2
    double getShear2Err() const {
        return Measurement<Shape>::get<Shape::SHEAR2_ERR, double>();
    }


    /// get the resolution
    double getResolution() const {
        return Measurement<Shape>::get<Shape::RESOLUTION, double>();
    }
    
    /// Return the width
    double getSigma() const {
        return Measurement<Shape>::get<Shape::SIGMA, double>();
    }
    /// Return the error in the width
    double getSigmaErr() const {
        return Measurement<Shape>::get<Shape::SIGMA_ERR, double>();
    }


    /// Return the xx-moment for the PSF
    double getPsfIxx() const {
        return Measurement<Shape>::get<Shape::PSF_IXX, double>();
    }
    /// Return the error in the xx-moment for the PSF
    double getPsfIxxErr() const {
        return Measurement<Shape>::get<Shape::PSF_IXX_ERR, double>();
    }
    /// Return the xx-moment for the PSF
    double getPsfIxy() const {
        return Measurement<Shape>::get<Shape::PSF_IXY, double>();
    }
    /// Return the error in the xy-moment for the PSF
    double getPsfIxyErr() const {
        return Measurement<Shape>::get<Shape::PSF_IXY_ERR, double>();
    }
    /// Return the yy-moment for the PSF
    double getPsfIyy() const {
        return Measurement<Shape>::get<Shape::PSF_IYY, double>();
    }
    /// Return the error in the yy-moment for the PSF
    double getPsfIyyErr() const {
        return Measurement<Shape>::get<Shape::PSF_IYY_ERR, double>();
    }

    /// Return the status of the routine which performed the calculation
    // This should be a boost::int16_t, but Schema doesn't support that.
    // ... thus requires a cast in meas-algorithms Measure.cc where it get passed
    //     through to Source
    int getStatus() const {
        return Measurement<Shape>::get<Shape::STATUS, int>();
    }
    
    virtual ::std::ostream &output(std::ostream &os) const {
        return os << "(" << getX() << "+-" << getXErr() << ", " << getY() << "+-" << getYErr() << ")";
    }

private:
    LSST_SERIALIZE_PARENT(lsst::afw::detection::Measurement<Shape>)
};
}}}

LSST_REGISTER_SERIALIZER(lsst::afw::detection::Shape)

#endif
