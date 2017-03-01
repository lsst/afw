#if !defined(LSST_AFW_GEOM_ANGLE_H)
#define LSST_AFW_GEOM_ANGLE_H

#include <limits>
#include <iostream>
#include <boost/math/constants/constants.hpp>
#include <cmath>

namespace lsst {
namespace afw {
namespace geom {

/************************************************************************************************************/
/*
 * None of C99, C++98, and C++0x define M_PI, so we'll do it ourselves
 */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
/// The ratio of a circle's circumference to diameter
double const PI = boost::math::constants::pi<double>();
double const TWOPI = boost::math::constants::pi<double>() * 2.0;
double const HALFPI = boost::math::constants::pi<double>() * 0.5;
double const ONE_OVER_PI = 1.0 / boost::math::constants::pi<double>();
double const SQRTPI = sqrt(boost::math::constants::pi<double>());
double const INVSQRTPI = 1.0 / sqrt(boost::math::constants::pi<double>());
double const ROOT2 = boost::math::constants::root_two<double>();  // sqrt(2)
#pragma clang diagnostic pop

// These shouldn't be necessary if the Angle class is used, but sometimes you just need
// them.  Better to define them once here than have *180/PI throughout the code...
inline double degToRad(double x) { return x * PI / 180.; }
inline double radToDeg(double x) { return x * 180. / PI; }
inline double radToArcsec(double x) { return x * 3600. * 180. / PI; }
inline double radToMas(double x) { return x * 1000. * 3600. * 180. / PI; }
inline double arcsecToRad(double x) { return (x / 3600.) * PI / 180.; }
inline double masToRad(double x) { return (x / (1000. * 3600.)) * PI / 180.; }

/************************************************************************************************************/

class Angle;
/**
 * A class used to convert scalar POD types such as double to Angle
 *
 * For example:
 *
 *     Angle pi = 180*degrees;
 *
 * is equivalent to
 *
 *     Angle pi(180, degrees);
 */
class AngleUnit {
    friend class Angle;
    template <typename T>
    friend Angle operator*(T lhs, AngleUnit rhs);

public:
    explicit AngleUnit(double val) : _val(val) {}

    bool operator==(AngleUnit const& rhs) const;

private:
    double _val;
};

inline bool AngleUnit::operator==(AngleUnit const& rhs) const { return (_val == rhs._val); }

AngleUnit const radians = AngleUnit(1.0);                     ///< constant with units of radians
AngleUnit const degrees = AngleUnit(PI / 180.0);              ///< constant with units of degrees
AngleUnit const hours = AngleUnit(PI * 15.0 / 180.0);         ///< constant with units of hours
AngleUnit const arcminutes = AngleUnit(PI / 60 / 180.0);      ///< constant with units of arcminutes
AngleUnit const arcseconds = AngleUnit(PI / 180.0 / 3600.0);  ///< constant with units of arcseconds

/************************************************************************************************************/
/**
 * A class representing an Angle
 *
 * Angles may be manipulated like doubles, and automatically converted to doubles, but they may not be
 * constructed from doubles without calling a constructor or multiplying by an AngleUnit. Angles can be
 * modified only by assignment; all other operations that transform an Angle return a new Angle instead.
 */
class Angle {
    friend class AngleUnit;

public:
    /** Construct an Angle with the specified value (interpreted in the given units) */
    explicit Angle(double val, AngleUnit units = radians) : _val(val * units._val) {}

    Angle() : _val(0) {}

    /** Copy constructor. */
    Angle(Angle const& other) : _val(other._val) {}

    /** Convert an Angle to a double in radians*/
    operator double() const { return _val; }

    /** Return an Angle's value as a double in the specified units (e.g.\ ::degrees) */
    double asAngularUnits(AngleUnit const& units) const { return _val / units._val; }

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

    double toUnitSphereDistanceSquared() const;

    static Angle fromUnitSphereDistanceSquared(double d2);

    /**
     * Wrap this angle to the range [0, 2&pi;).
     *
     * @returns an angle in the normalized interval.
     *
     * @exceptsafe Shall not throw exceptions.
     *
     * @warning The upper limit is only guaranteed for radians; the upper limit
     * may be slightly squishy for other units, due to roundoff errors. Whether
     * there are any violations is unknown; please update this comment if you
     * can prove that the limits are or are not valid for all supported units.
     */
    Angle wrap() const noexcept;

    /**
     * Wrap this angle to the range [-&pi;, &pi;).
     *
     * @returns an angle in the normalized interval.
     *
     * @exceptsafe Shall not throw exceptions.
     *
     * @warning Exact limits are only guaranteed for radians; limits for other
     * units may be slightly squishy, due to roundoff errors. Whether there are
     * any violations is unknown; please update this comment if you can prove
     * that the limits are or are not valid for all supported units.
     */
    Angle wrapCtr() const noexcept;

    /**
     * Wrap this angle to a value `x` such that -&pi; &le; `x - refAng` < &pi;.
     *
     * @param refAng reference angle to match
     *
     * @returns an angle in the custom normalized interval.
     *
     * @exceptsafe Shall not throw exceptions.
     *
     * @warning Exact limits are only guaranteed for radians; limits for other
     * units may be slightly squishy due to roundoff errors. There are known
     * violations that are demonstrated in testWrap in tests/angle.py.
     */
    Angle wrapNear(Angle const& refAng) const noexcept;

    /**
     * The signed difference between two Angles.
     *
     * @param other the angle to which this angle will be compared
     * @return `*this - other`, wrapped to the range [-&pi;, &pi;)
     *
     * @exceptsafe Shall not throw exceptions.
     */
    Angle separation(Angle const& other) const noexcept;

#define ANGLE_OPUP_TYPE(OP, TYPE)       \
    Angle& operator OP(TYPE const& d) { \
        _val OP d;                      \
        return *this;                   \
    }

    ANGLE_OPUP_TYPE(*=, double)
    ANGLE_OPUP_TYPE(*=, int)
    ANGLE_OPUP_TYPE(+=, double)
    ANGLE_OPUP_TYPE(+=, int)
    ANGLE_OPUP_TYPE(-=, double)
    ANGLE_OPUP_TYPE(-=, int)

#undef ANGLE_OPUP_TYPE

#define ANGLE_COMP(OP) \
    bool operator OP(const Angle& rhs) { return _val OP rhs._val; }

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
 */
#define ANGLE_OP(OP)                                                    \
    inline Angle operator OP(Angle a, Angle d) {                        \
        return Angle(static_cast<double>(a) OP static_cast<double>(d)); \
    }

// We need both int and double versions to avoid ambiguous overloading due to
// implicit conversion of Angle to double
#define ANGLE_OP_TYPE(OP, TYPE)                                                              \
    inline Angle operator OP(Angle a, TYPE d) { return Angle(static_cast<double>(a) OP d); } \
                                                                                             \
    inline Angle operator OP(TYPE d, Angle a) { return Angle(d OP static_cast<double>(a)); }

ANGLE_OP(+)
ANGLE_OP(-)
ANGLE_OP(*)
ANGLE_OP_TYPE(*, double)
ANGLE_OP_TYPE(*, int)

#undef ANGLE_OP
#undef ANGLE_OP_TYPE

inline const Angle operator-(Angle const angle) { return Angle(-static_cast<double>(angle)); }

// Division is different.  Don't allow division by an Angle
inline Angle operator/(Angle a, int d) { return Angle(static_cast<double>(a) / d); }

inline Angle operator/(Angle a, double d) { return Angle(static_cast<double>(a) / d); }

template <typename T>
double operator/(T const lhs, Angle rhs);

/**
 * Output operator for an Angle
 */
std::ostream& operator<<(std::ostream& s,  ///< The output stream
                         Angle a           ///< The angle
                         );

/************************************************************************************************************/
/**
 * Allow a user to check if they have an angle (yes; they could do this themselves via trivial TMP)
 */
template <typename T>
inline bool isAngle(T) {
    return false;
};

inline bool isAngle(Angle const&) { return true; };

/************************************************************************************************************/
/**
 * Use AngleUnit to convert a POD (e.g.\ int, double) to an Angle; e.g.\ 180*::degrees.
 */
template <typename T>
inline Angle operator*(T lhs,         ///< the value to convert
                       AngleUnit rhs  ///< the conversion coefficient
                       ) {
    static_assert(std::numeric_limits<T>::is_specialized,
                  "Only numeric types may be converted to Angles using degrees/radians!");
    return Angle(lhs * rhs._val);
}

/************************************************************************************************************/
// Inline method definitions, placed last in order to benefit from Angle's full API

inline double Angle::toUnitSphereDistanceSquared() const {
    return 2. * (1. - std::cos(asRadians()));
    // == 4.0 * pow(std::sin(0.5 * asRadians()), 2.0)
}

inline Angle Angle::fromUnitSphereDistanceSquared(double d2) {
    return (std::acos(1. - d2 / 2.)) * radians;
    // == 2.0 * asin(0.5 * sqrt(d2))
}

inline Angle Angle::wrap() const noexcept {
    double wrapped = std::fmod(_val, TWOPI);
    // wrapped is in the range (-TWOPI, TWOPI)
    if (wrapped < 0.0) wrapped += TWOPI;
    // if wrapped is small enough, adding 2 pi gives 2 pi
    if (wrapped >= TWOPI) wrapped = 0.0;
    return wrapped * radians;
}

inline Angle Angle::wrapCtr() const noexcept {
    double wrapped = std::fmod(_val, TWOPI);
    // wrapped is in the range [-TWOPI, TWOPI]
    if (wrapped < -PI) {
        wrapped += TWOPI;
        if (wrapped >= PI) {
            // handle roundoff error, however unlikely
            wrapped = -PI;
        }
    } else if (wrapped >= PI) {
        wrapped -= TWOPI;
        if (wrapped < -PI) {
            // handle roundoff error, however unlikely
            wrapped = -PI;
        }
    }
    return wrapped * radians;
}

inline Angle Angle::wrapNear(Angle const& refAng) const noexcept {
    // compute (this - refAng).wrapCtr() + refAng
    // which is correct except for roundoff error at the edges
    double const refAngRad = refAng.asRadians();
    double wrapped = (*this - refAng).wrapCtr().asRadians() + refAngRad;

    // roundoff can cause slightly out-of-range values; fix those
    if (wrapped - refAngRad >= PI) {
        wrapped -= TWOPI;
    }
    // maximum relative roundoff error for subtraction is 2 epsilon
    if (wrapped - refAngRad < -PI) {
        wrapped -= wrapped * 2.0 * std::numeric_limits<double>::epsilon();
    }
    return wrapped * radians;
}

inline Angle Angle::separation(Angle const& other) const noexcept { return (*this - other).wrapCtr(); }
}
}
}
#endif  // if !defined(LSST_AFW_GEOM_ANGLE_H)
