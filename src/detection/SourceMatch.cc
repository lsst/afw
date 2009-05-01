// -*- lsst-c++ -*-
/** @file
  * @ingroup afw
  */
#include <algorithm>
#include <cmath>

#include "boost/scoped_array.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection/SourceMatch.h"


namespace ex = lsst::pex::exceptions;
namespace det = lsst::afw::detection;

namespace lsst { namespace afw { namespace detection { namespace {

    double const DEGREES_PER_RADIAN = 57.2957795130823208767981548141;
    double const RADIANS_PER_DEGREE = 0.0174532925199432957692369076849;
    double const DEC_SCALE = 5.96523235555555555555555555e06;   //  536870912.0/90.0
    double const RA_SCALE = 1.19304647111111111111111111111e07; // 1073741824.0/90.0
    static int const LUT_SIZE = 256;  // must be a power of 2
    // shift that reduces an integer in [0, 2^30] to [0, LUT_SIZE]
    static int const LUT_SHIFT = 22;

    struct SourcePos {
        int dec;
        unsigned int ra;
        double x;
        double y;
        double z;
        Source::Ptr const *src;
    };

    bool operator<(SourcePos const &s1, SourcePos const &s2) {
        return (s1.dec < s2.dec);
    }

    struct CmpSourcePtr {
        bool operator()(Source::Ptr const *s1, Source::Ptr const *s2) {
            return (*s1)->getYAstrom() < (*s2)->getYAstrom();
        }
    };

    /**
      * Extract source positions from @a set, convert them to cartesian coordinates
      * (for faster distance checks) and sort the resulting array of @c SourcePos
      * instances by declination.
      *
      * @param[in] set          set of sources to process
      * @param[out] positions   pointer to an array of at least @c set.size()
      *                         SourcePos instances
      */
    void makeSourcePositions(SourceSet const &set, SourcePos *positions) {
        size_t n = 0;
        for (SourceSet::const_iterator i(set.begin()), e(set.end()); i != e; ++i, ++n) {
            double ra = (*i)->getRa();
            double dec = (*i)->getDec();
            if (ra < 0.0 || ra >= 360.0) {
                throw LSST_EXCEPT(ex::RangeErrorException, "right ascension out of range");
            }
            if (dec < -90.0 || dec > 90.0) {
                throw LSST_EXCEPT(ex::RangeErrorException, "declination out of range");
            }
            double cosDec    = std::cos(RADIANS_PER_DEGREE*dec);
            // map declination to an integer in range [-2^29, 2^29]
            positions[n].dec = static_cast<int>(std::floor(dec*DEC_SCALE));
            // map right ascension to an integer in range [0, 2^32)
            positions[n].ra  = static_cast<unsigned int>(std::floor(ra*RA_SCALE));
            positions[n].x   = std::cos(RADIANS_PER_DEGREE*ra)*cosDec;
            positions[n].y   = std::sin(RADIANS_PER_DEGREE*ra)*cosDec;
            positions[n].z   = std::sin(RADIANS_PER_DEGREE*dec);
            positions[n].src = &(*i);
        }
        std::sort(positions, positions + n);
    }

    /**
      * Compute the extent in right ascension [-alpha, alpha] of the circle
      * with radius @a theta and center (0, @a centerDec) on the unit sphere.
      *
      * @pre    @code theta > 0.0 @endcode
      * @pre    @code centerDec >= -90.0 && centerDec <= 90.0 @endcode
      *
      * @param[in] theta     the radius of the circle to find ra extents for (degrees)
      * @param[in] centerDec the declination of the circle center (degrees)
      *
      * @return  the largest right ascension of any point on the input circle
      */
    double maxAlpha(double theta, double centerDec) {
        if (std::fabs(centerDec) + theta > 89.9) {
            return 180.0;
        }
        double y = std::sin(RADIANS_PER_DEGREE*theta);
        double x = std::sqrt(std::fabs(cos(RADIANS_PER_DEGREE*(centerDec - theta))*
                                       cos(RADIANS_PER_DEGREE*(centerDec + theta))));
        return std::fabs(std::atan(y/x))*DEGREES_PER_RADIAN;
    }

    /** Construct a lookup table which, given a declination D, supplies an ra delta &alpha;
      * such than a ra span of 2&alpha; is guaranteed to encompass a circle of the given
      * radius centered at D.
      *
      * @param[in] radius   radius of circle to compute &alpha; LUT for (degrees)
      * @param[in] lut      pointer to an array of at least LUT_SIZE + 1 unsigned integers.
      */
    void makeAlphaLut(double radius, unsigned int *lut) {
        int i = 0;
        for (; i < LUT_SIZE; ++i) {
            double absDecMin = std::fabs((i - LUT_SIZE/2)*(180.0/LUT_SIZE));
            double absDecMax = std::fabs((i + 1 - LUT_SIZE/2)*(180.0/LUT_SIZE));
            double centerDec = std::max(absDecMin, absDecMax);
            lut[i] = static_cast<unsigned int>(std::ceil(maxAlpha(radius, centerDec)*RA_SCALE));
        }
        lut[i] = lut[i-1]; // for dec == 90.0
    }


}}}} // namespace lsst::afw::detection::<anonymous>


/** Compute all tuples (s1,s2,d) where s1 belings to @a set1, s2 belongs to @a set2 and
  * d, the distance between s1 and s2 in arcseconds, is at most @a radius. If set1 and
  * set2 are identical, then this call is equivalent to @c matchRaDec(set1,radius,true).
  * The match is performed in ra, dec space.
  *
  * @param[in] set1     first set of sources
  * @param[in] set2     second set of sources
  * @param[in] radius   match radius (arcsec)
  */
std::vector<det::SourceMatch> det::matchRaDec(det::SourceSet const &set1,
                                              det::SourceSet const &set2,
                                              double radius) {
    if (&set1 == &set2) {
        return matchRaDec(set1, radius, true);
    }
    if (radius < 0.0 || radius > 45.0*3600.0) {
        throw LSST_EXCEPT(ex::RangeErrorException, "match radius out of range");
    }
    if (set1.size() == 0 || set2.size() == 0) {
        return std::vector<SourceMatch>();
    }
    // setup match parameters
    unsigned int alphaLut[LUT_SIZE + 1];
    double const radiusDeg = radius/3600.0;
    makeAlphaLut(radiusDeg, alphaLut);

    int const deltaDec = std::ceil(radiusDeg*DEC_SCALE);
    double const shr = std::sin(RADIANS_PER_DEGREE*0.5*radiusDeg);
    double const d2Limit = 4.0*shr*shr;

    // Build position lists
    size_t const len1 = set1.size();
    size_t const len2 = set2.size();
    boost::scoped_array<SourcePos> pos1(new SourcePos[len1]);
    boost::scoped_array<SourcePos> pos2(new SourcePos[len2]);
    makeSourcePositions(set1, pos1.get());
    makeSourcePositions(set2, pos2.get());

    std::vector<SourceMatch> matches;
    for (size_t i = 0, start = 0; i < len1; ++i) {
        int minDec = pos1[i].dec - deltaDec;
        while (start < len2 && pos2[start].dec < minDec) { ++start; }
        if (start == len2) {
            break;
        }
        int maxDec = pos1[i].dec + deltaDec;
        unsigned int alpha = alphaLut[(pos1[i].dec + 536870912) >> LUT_SHIFT];
        unsigned int minRa = (pos1[i].ra - alpha) & 0xFFFFFFFFu;
        unsigned int maxRa = (pos1[i].ra + alpha) & 0xFFFFFFFFu;
        if (minRa < maxRa) {
            for (size_t j = start; j < len2 && pos2[j].dec <= maxDec; ++j) {
                if ((pos2[j].ra >= minRa) && (pos2[j].ra <= maxRa)) {
                    double dx = pos1[i].x - pos2[j].x;
                    double dy = pos1[i].y - pos2[j].y;
                    double dz = pos1[i].z - pos2[j].z;
                    double d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 < d2Limit) {
                        matches.push_back(SourceMatch(*pos1[i].src, *pos2[j].src,
                            DEGREES_PER_RADIAN*3600.0*2.0*std::asin(0.5*std::sqrt(d2))));
                    }
                }
            }
        } else {
            // ra wrap around
            for (size_t j = start; j < len2 && pos2[j].dec <= maxDec; ++j) {
                if ((pos2[j].ra >= minRa) || (pos2[j].ra <= maxRa)) {
                    double dx = pos1[i].x - pos2[j].x;
                    double dy = pos1[i].y - pos2[j].y;
                    double dz = pos1[i].z - pos2[j].z;
                    double d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 < d2Limit) {
                        matches.push_back(SourceMatch(*pos1[i].src, *pos2[j].src,
                            DEGREES_PER_RADIAN*3600.0*2.0*std::asin(0.5*std::sqrt(d2))));
                    }
                }
            }
        }
    }
    return matches;
}


/** Compute all tuples (s1,s2,d) where s1 != s2, s1 and s2 both belong to @a set,
  * and d, the distance between s1 and s2 in arcseconds, is at most @a radius. The
  * match is performed in ra, dec space.
  *
  * @param[in] set          the set of sources to self-match
  * @param[in] radius       match radius (arcsec)
  * @param[in] symmetric    if set to @c true symmetric matches are produced: i.e.
  *                         if (s1, s2, d) is reported, then so is (s2, s1, d).
  */
std::vector<det::SourceMatch> det::matchRaDec(det::SourceSet const &set,
                                              double radius,
                                              bool symmetric) {
    if (radius < 0.0 || radius > 45.0*3600.0) {
        throw LSST_EXCEPT(ex::RangeErrorException, "match radius out of range");
    }
    if (set.size() == 0) {
        return std::vector<SourceMatch>();
    }
    // setup match parameters
    unsigned int alphaLut[LUT_SIZE + 1];
    double const radiusDeg = radius/3600.0;
    makeAlphaLut(radiusDeg, alphaLut);

    int const deltaDec = std::ceil(radiusDeg*DEC_SCALE);
    double const shr = std::sin(RADIANS_PER_DEGREE*0.5*radiusDeg);
    double const d2Limit = 4.0*shr*shr;

    // Build position list
    size_t const len = set.size();
    boost::scoped_array<SourcePos> pos(new SourcePos[len]);
    makeSourcePositions(set, pos.get());

    std::vector<SourceMatch> matches;
    for (size_t i = 0; i < len; ++i) {
        int dec = pos[i].dec;
        int maxDec = pos[i].dec + deltaDec;
        unsigned int alpha = alphaLut[(dec + 536870912) >> LUT_SHIFT];
        unsigned int minRa = (pos[i].ra - alpha) & 0xFFFFFFFFu;
        unsigned int maxRa = (pos[i].ra + alpha) & 0xFFFFFFFFu;
        if (minRa < maxRa) {
            for (size_t j = i + 1; j < len && pos[j].dec <= maxDec; ++j) {
                if (pos[j].ra >= minRa && pos[j].ra <= maxRa) {
                    double dx = pos[i].x - pos[j].x;
                    double dy = pos[i].y - pos[j].y;
                    double dz = pos[i].z - pos[j].z;
                    double d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 < d2Limit) {
                        double d = DEGREES_PER_RADIAN*3600.0*2.0*std::asin(0.5*std::sqrt(d2));
                        matches.push_back(SourceMatch(*pos[i].src, *pos[j].src, d));
                        if (symmetric) {
                            matches.push_back(SourceMatch(*pos[j].src, *pos[i].src, d));
                        }
                    }
                }
            }
        } else {
            // ra wrap around
            for (size_t j = i + 1; j < len && pos[j].dec <= maxDec; ++j) {
                if (pos[j].ra >= minRa || pos[j].ra <= maxRa) {
                    double dx = pos[i].x - pos[j].x;
                    double dy = pos[i].y - pos[j].y;
                    double dz = pos[i].z - pos[j].z;
                    double d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 < d2Limit) {
                        double d = DEGREES_PER_RADIAN*3600.0*2.0*std::asin(0.5*std::sqrt(d2));
                        matches.push_back(SourceMatch(*pos[i].src, *pos[j].src, d));
                        if (symmetric) {
                            matches.push_back(SourceMatch(*pos[j].src, *pos[i].src, d));
                        } 
                    }
                }
            }
        }
    }
    return matches;
}


/** Compute all tuples (s1,s2,d) where s1 belings to @a set1, s2 belongs to @a set2 and
  * d, the distance between s1 and s2 in arcseconds, is at most @a radius. If set1 and
  * set2 are identical, then this call is equivalent to @c matchRaDec(set1,radius,true).
  * The match is performed in pixel space (2d cartesian).
  *
  * @param[in] set1     first set of sources
  * @param[in] set2     second set of sources
  * @param[in] radius   match radius (pixels)
  */
std::vector<det::SourceMatch> det::matchXy(det::SourceSet const &set1,
                                           det::SourceSet const &set2,
                                           double radius) {
    if (&set1 == &set2) {
       return matchXy(set1, radius);
    }
    // setup match parameters
    double const r2 = radius*radius;

    // copy and sort array of pointers on y
    size_t const len1 = set1.size();
    size_t const len2 = set2.size();
    boost::scoped_array<Source::Ptr const *> pos1(new Source::Ptr const *[len1]);
    boost::scoped_array<Source::Ptr const *> pos2(new Source::Ptr const *[len2]);
    size_t n = 0;
    for (SourceSet::const_iterator i(set1.begin()), e(set1.end()); i != e; ++i, ++n) {
        pos1[n] = &(*i);
    }
    n = 0;
    for (SourceSet::const_iterator i(set2.begin()), e(set2.end()); i != e; ++i, ++n) {
        pos2[n] = &(*i);
    }
    std::sort(pos1.get(), pos1.get() + len1, CmpSourcePtr());
    std::sort(pos2.get(), pos2.get() + len2, CmpSourcePtr());

    std::vector<SourceMatch> matches;
    for (size_t i = 0, start = 0; i < len1; ++i) {
        double y = (*pos1[i])->getYAstrom();
        double minY = y - radius;
        while (start < len2 && (*pos2[start])->getYAstrom() < minY) { ++start; }
        if (start == len2) {
            break;
        }
        double x = (*pos1[i])->getXAstrom();
        double maxY = y + radius;
        double y2;
        for (size_t j = start; j < len2 && (y2 = (*pos2[j])->getYAstrom()) <= maxY; ++j) {
            double dx = x - (*pos2[j])->getXAstrom();
            double dy = y - y2;
            double d2 = dx*dx + dy*dy;
            if (d2 < r2) {
                matches.push_back(SourceMatch(*pos1[i], *pos2[j], std::sqrt(d2)));
            }
        }
    }
    return matches;
}


/** Compute all tuples (s1,s2,d) where s1 != s2, s1 and s2 both belong to @a set,
  * and d, the distance between s1 and s2 in arcseconds, is at most @a radius. The
  * match is performed in pixel space (2d cartesian).
  *
  * @param[in] set          the set of sources to self-match
  * @param[in] radius       match radius (pixels)
  * @param[in] symmetric    if set to @c true symmetric matches are produced: i.e.
  *                         if (s1, s2, d) is reported, then so is (s2, s1, d).
  */
std::vector<det::SourceMatch> det::matchXy(det::SourceSet const &set,
                                           double radius,
                                           bool symmetric) {
    // setup match parameters
    double const r2 = radius*radius;

    // copy and sort array of pointers on y
    size_t const len = set.size();
    boost::scoped_array<Source::Ptr const *> pos(new Source::Ptr const *[len]);
    size_t n = 0;
    for (SourceSet::const_iterator i(set.begin()), e(set.end()); i != e; ++i, ++n) {
        pos[n] = &(*i);
    }
    std::sort(pos.get(), pos.get() + len, CmpSourcePtr());

    std::vector<SourceMatch> matches;
    for (size_t i = 0; i < len; ++i) {
        double x = (*pos[i])->getXAstrom();
        double y = (*pos[i])->getYAstrom();
        double maxY = y + radius;
        double y2;
        for (size_t j = i + 1; j < len && (y2 = (*pos[j])->getYAstrom()) <= maxY; ++j) {
            double dx = x - (*pos[j])->getXAstrom();
            double dy = y - y2;
            double d2 = dx*dx + dy*dy;
            if (d2 < r2) {
                double d = std::sqrt(d2);
                matches.push_back(SourceMatch(*pos[i], *pos[j], d));
                if (symmetric) {
                    matches.push_back(SourceMatch(*pos[j], *pos[i], d));
                }
            }
        }
    }
    return matches;
}

