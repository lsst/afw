#if !defined(LSST_AFW_GEOM_ANGLE_H)
#define LSST_AFW_GEOM_ANGLE_H

#include <iostream>
#include <type_traits>

#include <cmath>

#include "boost/math/constants/constants.hpp"

namespace lsst {
namespace afw {
namespace geom {

/*
 * None of C99, C++98, and C++11 define M_PI, so we'll do it ourselves
 */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
/// The ratio of a circle's circumference to diameter
double constexpr PI = boost::math::constants::pi<double>();
double constexpr TWOPI = boost::math::constants::pi<double>() * 2.0;
double constexpr HALFPI = boost::math::constants::pi<double>() * 0.5;
double constexpr ONE_OVER_PI = 1.0 / boost::math::constants::pi<double>();
// sqrt is not a constexpr on OS X
double const SQRTPI = sqrt(boost::math::constants::pi<double>());
double const INVSQRTPI = 1.0 / sqrt(boost::math::constants::pi<double>());
double constexpr ROOT2 = boost::math::constants::root_two<double>();  // sqrt(2)
#pragma clang diagnostic pop

// These shouldn't be necessary if the Angle class is used, but sometimes you just need
// them.  Better to define them once here than have *180/PI throughout the code...
inline constexpr double degToRad(double x) noexcept { return x * PI / 180.; }
inline constexpr double radToDeg(double x) noexcept { return x * 180. / PI; }
inline constexpr double radToArcsec(double x) noexcept { return x * 3600. * 180. / PI; }
inline constexpr double radToMas(double x) noexcept { return x * 1000. * 3600. * 180. / PI; }
inline constexpr double arcsecToRad(double x) noexcept { return (x / 3600.) * PI / 180.; }
inline constexpr double masToRad(double x) noexcept { return (x / (1000. * 3600.)) * PI / 180.; }

class Angle;
/**
 * A class used to convert scalar POD types such as `double` to Angle.
 *
 * For example, given the predefined %AngleUnit ::degrees:
 *
 *     Angle pi = 180*degrees;
 *
 * is equivalent to
 *
 *     Angle pi(180, degrees);
 */
class AngleUnit final {
    friend class Angle;
    template <typename T>
    friend constexpr Angle operator*(T lhs, AngleUnit rhs) noexcept;

public:
    /**
     * Define a new angle unit.
     *
     * @param val the number of radians in one unit. See ::degrees for an example.
     *
     * @exceptsafe Provides strong exception safety.
     */
    // Current implementation does not throw exceptions, but somebody may want
    // to add input validation later
    explicit constexpr AngleUnit(double val) : _val(val) {}

    /**
     * Test if two units are the same.
     *
     * @param rhs the unit to compare `this` to
     * @returns `true` if the two units have the same size, `false` otherwise.
     *
     * @exceptsafe Shall not throw exceptions.
     */
    constexpr bool operator==(AngleUnit const& rhs) const noexcept;

private:
    double _val;
};

inline constexpr bool AngleUnit::operator==(AngleUnit const& rhs) const noexcept {
    return (_val == rhs._val);
}

AngleUnit constexpr radians = AngleUnit(1.0);                     ///< constant with units of radians
AngleUnit constexpr degrees = AngleUnit(PI / 180.0);              ///< constant with units of degrees
AngleUnit constexpr hours = AngleUnit(PI * 15.0 / 180.0);         ///< constant with units of hours
AngleUnit constexpr arcminutes = AngleUnit(PI / 60 / 180.0);      ///< constant with units of arcminutes
AngleUnit constexpr arcseconds = AngleUnit(PI / 180.0 / 3600.0);  ///< constant with units of arcseconds

/**
 * A class representing an angle.
 *
 * Angles may be manipulated like doubles, and automatically converted to doubles, but they may not be
 * constructed from doubles without calling a constructor or multiplying by an AngleUnit. Angles can be
 * modified only by assignment; all other operations that transform an Angle return a new Angle instead.
 *
 * Unless otherwise specified, all methods and associated operators shall not throw exceptions.
 */
class Angle final {
    friend class AngleUnit;

public:
    /** Construct an Angle with the specified value (interpreted in the given units).
     *
     * @param val the size of the angle
     * @param units the units in which `val` is measured
     */
    explicit constexpr Angle(double val, AngleUnit units = radians) noexcept : _val(val* units._val) {}

    /// Construct the zero angle.
    constexpr Angle() noexcept : _val(0) {}

    /// Copy constructor.
    constexpr Angle(Angle const& other) noexcept = default;

    /// Move constructor.
    constexpr Angle(Angle&& other) noexcept = default;

    /// Copy assignment.
    Angle& operator=(Angle const& other) noexcept = default;

    /// Move assignment.
    Angle& operator=(Angle&& other) noexcept = default;

    ~Angle() = default;

    /// Convert an Angle to a double in radians.
    constexpr operator double() const noexcept { return _val; }

    /**
     * Return an Angle's value in the specified units.
     *
     * @param units the units in which the angle's value is desired (e.g.\ ::degrees).
     */
    constexpr double asAngularUnits(AngleUnit const& units) const noexcept { return _val / units._val; }

    /// Return an Angle's value in radians.
    constexpr double asRadians() const noexcept { return asAngularUnits(radians); }

    /// Return an Angle's value in degrees.
    constexpr double asDegrees() const noexcept { return asAngularUnits(degrees); }

    /// Return an Angle's value in hours.
    constexpr double asHours() const noexcept { return asAngularUnits(hours); }

    /// Return an Angle's value in arcminutes.
    constexpr double asArcminutes() const noexcept { return asAngularUnits(arcminutes); }

    /// Return an Angle's value in arcseconds.
    constexpr double asArcseconds() const noexcept { return asAngularUnits(arcseconds); }

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
     * Wrap this angle to a value `x` such that -&pi; &le; `x - refAng` &le; &pi;, approximately.
     *
     * @param refAng reference angle to match
     *
     * @returns an angle in the custom normalized interval.
     *
     * @exceptsafe Shall not throw exceptions.
     *
     * @warning The only firm limit is -&pi; &le; `x - refAng` in radians. The upper limit in radians
     * and both limits in other units are somewhat squishy, due to roundoff error.
     */
    Angle wrapNear(Angle const& refAng) const noexcept;

    /**
     * The signed difference between two Angles.
     *
     * @param other the angle to which this angle will be compared
     * @returns `*this - other`, wrapped to the range [-&pi;, &pi;)
     *
     * @exceptsafe Shall not throw exceptions.
     */
    Angle separation(Angle const& other) const noexcept;

#define ANGLE_OPUP_TYPE(OP, TYPE)                \
    Angle& operator OP(TYPE const& d) noexcept { \
        _val OP d;                               \
        return *this;                            \
    }

    //@{
    /// Multiply this angle by the given factor.
    ANGLE_OPUP_TYPE(*=, double)
    ANGLE_OPUP_TYPE(*=, int)
    //@}
    //@{
    /// Increase this angle by the given number of radians.
    ANGLE_OPUP_TYPE(+=, double)
    ANGLE_OPUP_TYPE(+=, int)
    //@}
    //@{
    /// Decrease this angle by the given number of radians.
    ANGLE_OPUP_TYPE(-=, double)
    ANGLE_OPUP_TYPE(-=, int)
//@}

#undef ANGLE_OPUP_TYPE

#define ANGLE_COMP(OP) \
    constexpr bool operator OP(const Angle& rhs) const noexcept { return _val OP rhs._val; }

    //@{
    /// Test if two Angles represent the same angle (without wrapping).
    ANGLE_COMP(==)
    ANGLE_COMP(!=)
    //@}
    //@{
    /// Compare the sizes of two Angles (without wrapping).
    ANGLE_COMP(<=)
    ANGLE_COMP(>=)
    ANGLE_COMP(<)
    ANGLE_COMP(>)
//@}

#undef ANGLE_COMP

private:
    double _val;
};

/*
 * Operators for Angles.
 */
#define ANGLE_OP(OP)                                                    \
    inline constexpr Angle operator OP(Angle a, Angle d) noexcept {     \
        return Angle(static_cast<double>(a) OP static_cast<double>(d)); \
    }

// We need both int and double versions to avoid ambiguous overloading due to
// implicit conversion of Angle to double
#define ANGLE_OP_TYPE(OP, TYPE)                                    \
    inline constexpr Angle operator OP(Angle a, TYPE d) noexcept { \
        return Angle(static_cast<double>(a) OP d);                 \
    }                                                              \
                                                                   \
    inline constexpr Angle operator OP(TYPE d, Angle a) noexcept { \
        return Angle(d OP static_cast<double>(a));                 \
    }

/**
 * Sum of two angles.
 *
 * @relatesalso Angle
 */
ANGLE_OP(+)
/**
 * Difference of two angles.
 *
 * @relatesalso Angle
 */
ANGLE_OP(-)
/**
 * Product of two angles.
 *
 * @warning The result will be treated like an planar angle, not a solid angle.
 *
 * @relatesalso Angle
 */
ANGLE_OP(*)
//@{
/**
 * Product of an angle and a scalar.
 *
 * @relatesalso Angle
 */
ANGLE_OP_TYPE(*, double)
ANGLE_OP_TYPE(*, int)
//@}

#undef ANGLE_OP
#undef ANGLE_OP_TYPE

/**
 * An angle in the opposite sense.
 *
 * @relatesalso Angle
 */
inline constexpr Angle operator-(Angle angle) { return Angle(-static_cast<double>(angle)); }

// Apparently @relatesalso doesn't work with grouping
/**
 * Ratio of an angle and a scalar.
 *
 * @relatesalso Angle
 */
inline constexpr Angle operator/(Angle a, int d) noexcept { return Angle(static_cast<double>(a) / d); }

/**
 * Ratio of an angle and a scalar.
 *
 * @relatesalso Angle
 */
inline constexpr Angle operator/(Angle a, double d) noexcept { return Angle(static_cast<double>(a) / d); }

// Division is different.  Don't allow division by an Angle
template <typename T>
constexpr double operator/(T const lhs, Angle rhs) noexcept = delete;

/**
 * Print an Angle to a stream.
 *
 * The exact details of the string representation are unspecified and
 * subject to change, but the following may be regarded as typical:
 * `"0.567 rad"`.
 *
 * @param s The output stream.
 * @param a The angle.
 *
 * @exceptsafe Provides basic exception guarantee.
 *
 * @relatesalso Angle
 */
std::ostream& operator<<(std::ostream& s, Angle a);

/// Allow a user to check if they have an angle.
template <typename T>
inline constexpr bool isAngle(T) noexcept {
    return std::is_base_of<Angle, T>::value;
};

/**
 * Use AngleUnit to convert a POD (e.g.\ int, double) to an Angle; e.g.\ 180*::degrees.
 *
 * @param lhs the value to convert
 * @param rhs the conversion coefficient
 *
 * @exceptsafe Shall not throw exceptions.
 *
 * @relatesalso Angle
 */
template <typename T>
inline constexpr Angle operator*(T lhs, AngleUnit rhs) noexcept {
    static_assert(std::is_arithmetic<T>::value,
                  "Only numeric types may be multiplied by an AngleUnit to create an Angle!");
    return Angle(lhs * rhs._val);
}

// Inline method definitions, placed last in order to benefit from Angle's full API

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

    // roundoff can cause slightly out-of-range values; fix those as bast we can;
    // note that both conditionals are wanted, since either or both may execute
    // (though if/else could be used if the lower limit was squishy for radians)
    if (wrapped - refAngRad >= PI) {
        wrapped -= TWOPI;
    }
    if (wrapped - refAngRad < -PI) {
        wrapped += TWOPI;
    }
    return wrapped * radians;
}

inline Angle Angle::separation(Angle const& other) const noexcept { return (*this - other).wrapCtr(); }
}
}
}
#endif  // if !defined(LSST_AFW_GEOM_ANGLE_H)
