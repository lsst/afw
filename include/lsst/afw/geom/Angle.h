#if !defined(ANGLE_H)
#define ANGLE_H

#include <limits>
#include <iostream>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <boost/static_assert.hpp>

namespace lsst { namespace afw { namespace geom {

/************************************************************************************************************/
/*
 * None of C99, C++98, and C++0x define M_PI, so we'll do it ourselves
 */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
double const PI = boost::math::constants::pi<double>(); ///< The ratio of a circle's circumference to diameter
double const TWOPI = boost::math::constants::pi<double>() * 2.0;
double const HALFPI = boost::math::constants::pi<double>() * 0.5;
double const ONE_OVER_PI = 1.0 / boost::math::constants::pi<double>();
double const SQRTPI = sqrt(boost::math::constants::pi<double>());
double const INVSQRTPI = 1.0/sqrt(boost::math::constants::pi<double>());
double const ROOT2 = boost::math::constants::root_two<double>(); // sqrt(2)
#pragma clang diagnostic pop

// These shouldn't be necessary if the Angle class is used, but sometimes you just need
// them.  Better to define them once here than have *180/PI throughout the code...
inline double degToRad(double x) {
	return x * PI / 180.;
}
inline double radToDeg(double x) {
	return x * 180. / PI;
}
inline double radToArcsec(double x) {
	return x * 3600. * 180. / PI;
}
inline double radToMas(double x) {
	return x * 1000. * 3600. * 180. / PI;
}
inline double arcsecToRad(double x) {
	return (x / 3600.) * PI / 180.;
}
inline double masToRad(double x) {
	return (x / (1000. * 3600.)) * PI / 180.;
}

// NOTE, if you add things here, you must also add them to
//    python/lsst/afw/geom/__init__.py
// (if you want them to accessible from python)

#if 0 && !defined(M_PI)                 // a good idea, but with ramifications
#   define M_PI ::lsst::afw::geom::PI
#endif

/************************************************************************************************************/

class Angle;
/**
 * \brief A class used to convert scalar POD types such as double to Angle
 *
 * For example:
 * \code
 *    Angle pi = 180*degrees;
 * \endcode
 * is equivalent to 
 * \code
 *    Angle pi(180, degrees);
 * \endcode
 */
class AngleUnit {
    friend class Angle;
    template<typename T> friend const Angle operator *(T lhs, AngleUnit const rhs);
public:
    explicit AngleUnit(double val) : _val(val) {}

	bool operator==(AngleUnit const &rhs) const;
private:
    double _val;
};

inline bool lsst::afw::geom::AngleUnit::operator==(lsst::afw::geom::AngleUnit const &rhs) const {
	return (_val == rhs._val);
}

// NOTE, if you add things here, remember to also add them to
//    python/lsst/afw/geom/__init__.py

// swig likes this way of initialising the constant, so don't mess with it;
// N.b. swig 1.3 doesn't like PI/(60*180)
AngleUnit const radians =    AngleUnit(1.0); ///< constant with units of radians
AngleUnit const degrees =    AngleUnit(PI/180.0); // constant with units of degrees
AngleUnit const hours   =    AngleUnit(PI*15.0/180.0); // constant with units of hours
AngleUnit const arcminutes = AngleUnit(PI/60/180.0); // constant with units of arcminutes
AngleUnit const arcseconds = AngleUnit(PI/180.0/3600.0); // constant with units of arcseconds

/************************************************************************************************************/
/**
 * A class representing an Angle
 *
 * Angles may be manipulated like doubles, and automatically converted to doubles, but they may not be
 * constructed from doubles without calling a constructor or multiplying by an AngleUnit
 */
class Angle {
    friend class AngleUnit;
public:
    /** Construct an Angle with the specified value (interpreted in the given units) */
    explicit Angle(double val, AngleUnit units=radians) : _val(val*units._val) {}
	Angle() : _val(0) {}
    /** Copy constructor. */
    Angle(Angle const& other) : _val(other._val) {}
    /** Convert an Angle to a double in radians*/
    operator double() const { return _val; }
    /** Convert an Angle to a float in radians*/
    //operator float() const { return _val; }

    /** Return an Angle's value as a double in the specified units (i.e. afwGeom::degrees) */
    double asAngularUnits(AngleUnit const& units) const {
        return _val/units._val;
    }
    /** Return an Angle's value as a double in radians */
    double asRadians() const { return asAngularUnits(radians); }
    /** Return an Angle's value as a double in degrees */
    double asDegrees() const { return asAngularUnits(degrees); }
    /** Return an Angle's value as a double in hours */
    double asHours() const { return asAngularUnits(hours); }
    /** Return an Angle's value as a double in arcminutes */
    double asArcminutes() const { return asAngularUnits(arcminutes); }
    /** Return an Angle's value as a double in arcseconds */
    double asArcseconds() const { return asAngularUnits(arcseconds); }

	double toUnitSphereDistanceSquared() const { return 2. * (1. - std::cos(asRadians())); }
	// == 4.0 * pow(std::sin(0.5 * asRadians()), 2.0)
	static Angle fromUnitSphereDistanceSquared(double d2) {
		return (std::acos(1. - d2/2.)) * radians;
		// == 2.0 * asin(0.5 * sqrt(d2))
	}

	/** Wraps this angle to the range [0, 2 pi)

	@warning The upper limit is only guaranteed for radians;
	the upper limit may be slightly squishy for other units, due to roundoff errors.
    Whether there are any violations is unknown; please update this comment if you can prove
    that the limits are or are not valid for all supported units.
    */
	void wrap() {
		_val = std::fmod(_val, TWOPI);
		// _val is now in the range (-TWOPI, TWOPI)
		if (_val < 0.0)
			_val += TWOPI;
		// if _val is small enough, adding 2 pi gives 2 pi
		if (_val >= TWOPI)
			_val = 0.0;
	}
	
	/** Wrap this angle to the range [-pi, pi)
	
	@warning Exact limits are only guaranteed for radians; limits for other units
	may be slightly squishy, due to roundoff errors. Whether there are any violations is unknown;
	please update this comment if you can prove that the limits are or are not valid for all supported units.
	*/
	void wrapCtr() {
		_val = std::fmod(_val, TWOPI);
		// _val is now in the range [-TWOPI, TWOPI]
        if (_val < -PI) {
            _val += TWOPI;
            if (_val >= PI) {
                // handle roundoff error, however unlikely
                _val = -PI;
            }
        } else if (_val >= PI) {
            _val -= TWOPI;
            if (_val < -PI) {
                // handle roundoff error, however unlikely
                _val = -PI;
            }
        }
	}
	
	/** Wrap this angle such that pi <= this - refAng < pi
	
	@warning Exact limits are only guaranteed for radians; limits for other units
	may be slightly squishy due to roundoff errors. There are known violations
	that are demonstrated in testWrap in tests/angle.py.
	*/
	void wrapNear(
	    Angle const & refAng ///< reference angle to match
	) {
	    // compute this = (this - refAng).wrapCtr() + refAng
	    // which is correct except for roundoff error at the edges
	    double refAngRad = refAng.asRadians();
	    *this -= refAng;
	    wrapCtr();
        _val += refAngRad;

        // roundoff can cause slightly out-of-range values; fix those
        if (_val - refAngRad >= PI) {
            _val -= TWOPI;
        }
        // maximum relative roundoff error for subtraction is 2 epsilon
        if (_val - refAngRad < -PI) {
            _val -= _val * 2.0 * std::numeric_limits<double>::epsilon();
        }
	}
	

#define ANGLE_OPUP_TYPE(OP, TYPE)                             \
    Angle& operator OP(TYPE const& d) {						  \
		_val OP d;											  \
        return *this;										  \
    }

ANGLE_OPUP_TYPE(*=, double)
ANGLE_OPUP_TYPE(*=, int)
ANGLE_OPUP_TYPE(+=, double)
ANGLE_OPUP_TYPE(+=, int)
ANGLE_OPUP_TYPE(-=, double)
ANGLE_OPUP_TYPE(-=, int)

#undef ANGLE_OPUP_TYPE

#define ANGLE_COMP(OP)                          \
    bool operator OP ( const Angle& rhs ) {     \
        return _val OP rhs._val;                \
    }

ANGLE_COMP(==)
ANGLE_COMP(!=)
ANGLE_COMP(<=)
ANGLE_COMP(>=)
ANGLE_COMP(<)
ANGLE_COMP(>)

#undef ANGLE_COMP

private:
    double _val;
};

Angle const NullAngle = Angle(-1000000., degrees);


/************************************************************************************************************/
/*
 * Operators for Angles.
 *
 * N.b. We need both int and double versions to avoid ambiguous overloading due to implicit conversion of
 * Angle to double
 */
#define ANGLE_OP(OP)													\
    inline const Angle operator OP(Angle const a, Angle const d) {		\
        return Angle(static_cast<double>(a) OP static_cast<double>(d));	\
    }

#define ANGLE_OP_TYPE(OP, TYPE)                             \
    inline const Angle operator OP(Angle const a, TYPE d) {	\
        return Angle(static_cast<double>(a) OP d);          \
    }                                                       \
															\
    inline const Angle operator OP(TYPE d, Angle const a) {	\
        return Angle(d OP static_cast<double>(a));          \
    }

ANGLE_OP(+)
ANGLE_OP(-)
ANGLE_OP(*)
ANGLE_OP_TYPE(*, double)
ANGLE_OP_TYPE(*, int)

#undef ANGLE_OP
#undef ANGLE_OP_TYPE

// Division is different.  Don't allow division by an Angle
inline const Angle operator /(Angle const a, int d) {
    return Angle(static_cast<double>(a)/d);
}

inline const Angle operator /(Angle const a, double d) {
    return Angle(static_cast<double>(a)/d);
}

template<typename T>
	double operator /(T const lhs, Angle const rhs);
            
/************************************************************************************************************/
/**
 * \brief Allow a user to check if they have an angle (yes; they could do this themselves via trivial TMP)
 */
template<typename T>
inline bool isAngle(T) {
    return false;
};

inline bool isAngle(Angle const&) {
    return true;
};

/************************************************************************************************************/
/**
 * \brief Use AngleUnit to convert a POD (e.g. int, double) to an Angle; e.g. 180*afwGeom::degrees
 */
template<typename T>
inline
const Angle operator *(T lhs,              ///< the value to convert
                       AngleUnit const rhs ///< the conversion coefficient
                      ) {
    BOOST_STATIC_ASSERT_MSG(std::numeric_limits<T>::is_specialized,
                            "Only numeric types may be converted to Angles using degrees/radians!");
    return Angle(lhs*rhs._val);
}
/**
 * Output operator for an Angle
 */
std::ostream& operator<<(std::ostream &s, ///< The output stream
                         Angle const a    ///< The angle
						 );

}}}
#endif
