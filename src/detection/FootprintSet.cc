// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

/*
 * Utilities to detect sets of Footprint%s
 *
 * Create and use an lsst::afw::detection::FootprintSet, a collection of pixels above (or below) a threshold
 * in an Image
 *
 * The "collections of pixels" are represented as lsst::afw::detection::Footprint%s, so an example application
 * would be:
 *
       namespace image = lsst::afw::image; namespace detection = lsst::afw::detection;

       image::MaskedImage<float> img(10,20);
       *img.getImage() = 100;

       detection::FootprintSet<float> sources(img, 10);
       cout << "Found " << sources.getFootprints()->size() << " sources" << std::endl;
 */
#include <cstdint>
#include <memory>
#include <algorithm>
#include <cassert>
#include <set>
#include <string>
#include <typeinfo>
#include "boost/format.hpp"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Statistics.h"
#include "lsst/afw/detection/Peak.h"
#include "lsst/afw/detection/FootprintSet.h"
#include "lsst/afw/detection/FootprintCtrl.h"
#include "lsst/afw/detection/HeavyFootprint.h"

namespace lsst {
namespace afw {
namespace detection {

namespace {
/// Don't let doxygen see this block  @cond

typedef std::uint64_t IdPixelT;  // Type of temporary Images used in merging Footprints

struct Threshold_traits {};
struct ThresholdLevel_traits : public Threshold_traits {  // Threshold is a single number
};
struct ThresholdPixelLevel_traits : public Threshold_traits {  // Threshold varies from pixel to pixel
};
struct ThresholdBitmask_traits : public Threshold_traits {  // Threshold ORs with a bitmask
};

template <typename PixelT>
class setIdImage {
public:
    explicit setIdImage(std::uint64_t const id, bool overwriteId = false, long const idMask = 0x0)
            : _id(id),
              _idMask(idMask),
              _withSetReplace(false),
              _overwriteId(overwriteId),
              _oldIds(NULL),
              _pos() {
        if (_id & _idMask) {
            throw LSST_EXCEPT(
                    pex::exceptions::InvalidParameterError,
                    str(boost::format("Id 0x%x sets bits in the protected mask 0x%x") % _id % _idMask));
        }
    }

    setIdImage(std::uint64_t const id, std::set<std::uint64_t> *oldIds, bool overwriteId = false,
               long const idMask = 0x0)
            : _id(id),
              _idMask(idMask),
              _withSetReplace(true),
              _overwriteId(overwriteId),
              _oldIds(oldIds),
              _pos(oldIds->begin()) {
        if (_id & _idMask) {
            throw LSST_EXCEPT(
                    pex::exceptions::InvalidParameterError,
                    str(boost::format("Id 0x%x sets bits in the protected mask 0x%x") % _id % _idMask));
        }
    }

    void operator()(lsst::geom::Point2I const &point, PixelT &input) {
        if (_overwriteId) {
            auto val = input & ~_idMask;

            if (val != 0 && _withSetReplace) {
                _pos = _oldIds->insert(_pos, val);
            }

            input = (input & _idMask) + _id;
        } else {
            input += _id;
        }
    }

private:
    std::uint64_t const _id;
    long const _idMask;
    bool _withSetReplace;
    bool _overwriteId;
    std::set<std::uint64_t> *_oldIds;
    std::set<std::uint64_t>::const_iterator _pos;
};

//
// Define our own functions to handle NaN tests;  this gives us the
// option to define a value for e.g. image::MaskPixel or int
//
template <typename T>
inline bool isBadPixel(T) {
    return false;
}

template <>
inline bool isBadPixel(float val) {
    return std::isnan(val);
}

template <>
inline bool isBadPixel(double val) {
    return std::isnan(val);
}

/*
 * Return the number of bits required to represent a unsigned long
 */
int nbit(unsigned long i) {
    int n = 0;
    while (i > 0) {
        ++n;
        i >>= 1;
    }

    return n;
}
/*
 * Find the list of pixel values that lie in a Footprint
 *
 * Used when the Footprints are constructed from an Image containing Footprint indices
 */
template <typename T>
class FindIdsInFootprint {
public:
    explicit FindIdsInFootprint() : _ids(), _old(0) {}

    // Reset everything for a new Footprint
    void reset() {
        _ids.clear();
        _old = 0;
    }

    // Take by copy and not be reference on purpose
    void operator()(lsst::geom::Point2I const &point, T val) {
        if (val != _old) {
            _ids.insert(val);
            _old = val;
        }
    }

    std::set<T> const &getIds() const { return _ids; }

private:
    std::set<T> _ids;
    T _old;
};

/*
 * Sort peaks by decreasing pixel value.  N.b. -ve peaks are sorted the same way as +ve ones
 */
struct SortPeaks {
    bool operator()(std::shared_ptr<PeakRecord const> a, std::shared_ptr<PeakRecord const> b) {
        if (a->getPeakValue() != b->getPeakValue()) {
            return (a->getPeakValue() > b->getPeakValue());
        }

        if (a->getIx() != b->getIx()) {
            return (a->getIx() < b->getIx());
        }

        return (a->getIy() < b->getIy());
    }
};
/*
 * Worker routine for merging two FootprintSets, possibly growing them as we proceed
 */
FootprintSet mergeFootprintSets(FootprintSet const &lhs,      // the FootprintSet to be merged to
                                int rLhs,                     // Grow lhs Footprints by this many pixels
                                FootprintSet const &rhs,      // the FootprintSet to be merged into lhs
                                int rRhs,                     // Grow rhs Footprints by this many pixels
                                FootprintControl const &ctrl  // Control how the grow is done
) {
    typedef FootprintSet::FootprintList FootprintList;
    // The isXXX routines return <isset, value>
    bool const circular = ctrl.isCircular().first && ctrl.isCircular().second;
    bool const isotropic = ctrl.isIsotropic().second;  // isotropic grow as opposed to a Manhattan metric
                                                       // n.b. Isotropic grows are significantly slower
    bool const left = ctrl.isLeft().first && ctrl.isLeft().second;
    bool const right = ctrl.isRight().first && ctrl.isRight().second;
    bool const up = ctrl.isUp().first && ctrl.isUp().second;
    bool const down = ctrl.isDown().first && ctrl.isDown().second;

    lsst::geom::Box2I const region = lhs.getRegion();
    if (region != rhs.getRegion()) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                          boost::format("The two FootprintSets must have the same region").str());
    }

    auto idImage = std::make_shared<image::Image<IdPixelT>>(region);
    idImage->setXY0(region.getMinX(), region.getMinY());
    *idImage = 0;

    FootprintList const &lhsFootprints = *lhs.getFootprints();
    FootprintList const &rhsFootprints = *rhs.getFootprints();
    int const nLhs = lhsFootprints.size();
    int const nRhs = rhsFootprints.size();
    /*
     * In general the lists of Footprints overlap, so we need to make sure that the IDs can be
     * uniquely recovered from the idImage.  We do this by allocating a range of bits to the lhs IDs
     */
    int const lhsIdNbit = nbit(nLhs);
    int const lhsIdMask = (lhsIdNbit == 0) ? 0x0 : (1 << lhsIdNbit) - 1;

    if (std::size_t(nRhs << lhsIdNbit) > std::numeric_limits<IdPixelT>::max() - 1) {
        throw LSST_EXCEPT(pex::exceptions::OverflowError,
                          (boost::format("%d + %d footprints need too many bits; change IdPixelT typedef") %
                           nLhs % nRhs)
                                  .str());
    }
    /*
     * When we insert grown Footprints into the idImage we can potentially overwrite an entire Footprint,
     * losing any peaks that it might contain.  We'll preserve the overwritten Ids in case we need to
     * get them back (n.b. Footprints that overlap, but both if which survive, will appear in this list)
     */
    typedef std::map<int, std::set<std::uint64_t>> OldIdMap;
    OldIdMap overwrittenIds;  // here's a map from id -> overwritten IDs

    auto grower = [&circular, &up, &down, &left, &right, &isotropic](
                          std::shared_ptr<Footprint> const &foot, int amount) -> std::shared_ptr<Footprint> {
        if (circular) {
            auto element = isotropic ? geom::Stencil::CIRCLE : geom::Stencil::MANHATTAN;
            auto tmpFoot = std::make_shared<Footprint>(foot->getSpans()->dilated(amount, element),
                                                       foot->getRegion());
            return tmpFoot;
        } else {
            int top = up ? amount : 0;
            int bottom = down ? amount : 0;
            int lLimit = left ? amount : 0;
            int rLimit = right ? amount : 0;

            auto yRange = top + bottom + 1;
            std::vector<geom::Span> spanList;
            spanList.reserve(yRange);

            for (auto dy = 1; dy <= top; ++dy) {
                spanList.push_back(geom::Span(dy, 0, 0));
            }
            for (auto dy = -1; dy >= -bottom; --dy) {
                spanList.push_back(geom::Span(dy, 0, 0));
            }
            spanList.push_back(geom::Span(0, -lLimit, rLimit));
            geom::SpanSet structure(std::move(spanList));
            auto tmpFoot =
                    std::make_shared<Footprint>(foot->getSpans()->dilated(structure), foot->getRegion());
            return tmpFoot;
        }
    };

    IdPixelT id = 1;  // the ID inserted into the image
    for (FootprintList::const_iterator ptr = lhsFootprints.begin(), end = lhsFootprints.end(); ptr != end;
         ++ptr, ++id) {
        std::shared_ptr<Footprint> foot = *ptr;

        if (rLhs > 0 && foot->getArea() > 0) {
            foot = grower(foot, rLhs);
        }

        std::set<std::uint64_t> overwritten;
        foot->getSpans()
                ->clippedTo(idImage->getBBox())
                ->applyFunctor(setIdImage<IdPixelT>(id, &overwritten, true), *idImage);

        if (!overwritten.empty()) {
            overwrittenIds.insert(overwrittenIds.end(), std::make_pair(id, overwritten));
        }
    }

    assert(id <= std::size_t(1 << lhsIdNbit));
    id = (1 << lhsIdNbit);
    for (FootprintList::const_iterator ptr = rhsFootprints.begin(), end = rhsFootprints.end(); ptr != end;
         ++ptr, id += (1 << lhsIdNbit)) {
        std::shared_ptr<Footprint> foot = *ptr;

        if (rRhs > 0 && foot->getArea() > 0) {
            foot = grower(foot, rRhs);
        }

        std::set<std::uint64_t> overwritten;
        foot->getSpans()
                ->clippedTo(idImage->getBBox())
                ->applyFunctor(setIdImage<IdPixelT>(id, &overwritten, true, lhsIdMask), *idImage);

        if (!overwritten.empty()) {
            overwrittenIds.insert(overwrittenIds.end(), std::make_pair(id, overwritten));
        }
    }

    FootprintSet fs(*idImage, Threshold(1), 1, false);  // detect all pixels in rhs + lhs
                                                        /*
                                                         * Now go through the new Footprints looking up and remembering their progenitor's IDs; we'll use
                                                         * these IDs to merge the peaks in a moment
                                                         *
                                                         * We can't do this as we go through the idFinder as the IDs it returns are
                                                         *   (lhsId + 1) | ((rhsId + 1) << nbit)
                                                         * and, depending on the geometry, values of lhsId and/or rhsId can appear multiple times
                                                         * (e.g. if nbit is 2, idFinder IDs 0x5 and 0x6 both contain lhsId = 0) so we get duplicates
                                                         * of peaks.  This is not too bad, but it's a bit of a pain to make the lists unique again,
                                                         * and we avoid this by this two-step process.
                                                         */
    FindIdsInFootprint<IdPixelT> idFinder;
    for (FootprintList::iterator ptr = fs.getFootprints()->begin(), end = fs.getFootprints()->end();
         ptr != end; ++ptr) {
        std::shared_ptr<Footprint> foot = *ptr;

        // find the (mangled) [lr]hsFootprint IDs that contribute to foot
        foot->getSpans()->applyFunctor(idFinder, *idImage);

        std::set<std::uint64_t> lhsFootprintIndxs, rhsFootprintIndxs;  // indexes into [lr]hsFootprints

        for (std::set<IdPixelT>::iterator idptr = idFinder.getIds().begin(), idend = idFinder.getIds().end();
             idptr != idend; ++idptr) {
            unsigned int indx = *idptr;
            if ((indx & lhsIdMask) > 0) {
                std::uint64_t i = (indx & lhsIdMask) - 1;
                lhsFootprintIndxs.insert(i);
                /*
                 * Now allow for Footprints that vanished beneath this one
                 */
                OldIdMap::iterator mapPtr = overwrittenIds.find(indx);
                if (mapPtr != overwrittenIds.end()) {
                    std::set<std::uint64_t> &overwritten = mapPtr->second;

                    for (std::set<std::uint64_t>::iterator ptr = overwritten.begin(), end = overwritten.end();
                         ptr != end; ++ptr) {
                        lhsFootprintIndxs.insert((*ptr & lhsIdMask) - 1);
                    }
                }
            }
            indx >>= lhsIdNbit;

            if (indx > 0) {
                std::uint64_t i = indx - 1;
                rhsFootprintIndxs.insert(i);
                /*
                 * Now allow for Footprints that vanished beneath this one
                 */
                OldIdMap::iterator mapPtr = overwrittenIds.find(indx);
                if (mapPtr != overwrittenIds.end()) {
                    std::set<std::uint64_t> &overwritten = mapPtr->second;

                    for (std::set<std::uint64_t>::iterator ptr = overwritten.begin(), end = overwritten.end();
                         ptr != end; ++ptr) {
                        rhsFootprintIndxs.insert(*ptr - 1);
                    }
                }
            }
        }
        /*
         * We now have a complete set of Footprints that contributed to this one, so merge
         * all their Peaks into the new one
         */
        PeakCatalog &peaks = foot->getPeaks();

        for (std::set<std::uint64_t>::iterator ptr = lhsFootprintIndxs.begin(), end = lhsFootprintIndxs.end();
             ptr != end; ++ptr) {
            std::uint64_t i = *ptr;
            assert(i < lhsFootprints.size());
            PeakCatalog const &oldPeaks = lhsFootprints[i]->getPeaks();

            int const nold = peaks.size();
            peaks.insert(peaks.end(), oldPeaks.begin(), oldPeaks.end());
            // We use getInternal() here to get the vector of shared_ptr that Catalog uses internally,
            // which causes the STL algorithm to copy pointers instead of PeakRecords (which is what
            // it'd try to do if we passed Catalog's own iterators).
            std::inplace_merge(peaks.getInternal().begin(), peaks.getInternal().begin() + nold,
                               peaks.getInternal().end(), SortPeaks());
        }

        for (std::set<std::uint64_t>::iterator ptr = rhsFootprintIndxs.begin(), end = rhsFootprintIndxs.end();
             ptr != end; ++ptr) {
            std::uint64_t i = *ptr;
            assert(i < rhsFootprints.size());
            PeakCatalog const &oldPeaks = rhsFootprints[i]->getPeaks();

            int const nold = peaks.size();
            peaks.insert(peaks.end(), oldPeaks.begin(), oldPeaks.end());
            // See note above on why we're using getInternal() here.
            std::inplace_merge(peaks.getInternal().begin(), peaks.getInternal().begin() + nold,
                               peaks.getInternal().end(), SortPeaks());
        }
        idFinder.reset();
    }

    return fs;
}
/*
 * run-length code for part of object
 */
class IdSpan {
public:
    explicit IdSpan(int id, int y, int x0, int x1, double good) : id(id), y(y), x0(x0), x1(x1), good(good) {}
    int id;     /* ID for object */
    int y;      /* Row wherein IdSpan dwells */
    int x0, x1; /* inclusive range of columns */
    bool good;  /* includes a value over the desired threshold? */
};
/*
 * comparison functor; sort by ID then row
 */
struct IdSpanCompare {
    bool operator()(IdSpan const &a, IdSpan const &b) const {
        if (a.id < b.id) {
            return true;
        } else if (a.id > b.id) {
            return false;
        } else {
            return (a.y < b.y) ? true : false;
        }
    }
};
/*
 * Follow a chain of aliases, returning the final resolved value.
 */
int resolve_alias(std::vector<int> const &aliases, /* list of aliases */
                  int id) {                        /* alias to look up */
    int resolved = id;                             /* resolved alias */

    while (id != aliases[id]) {
        resolved = id = aliases[id];
    }

    return (resolved);
}
/// @endcond
}  // namespace

namespace {
template <typename ImageT>
void findPeaksInFootprint(ImageT const &image, bool polarity, PeakCatalog &peaks, Footprint &foot,
                          std::size_t const margin = 0) {
    auto spanSet = foot.getSpans();
    if (spanSet->size() == 0) {
        return;
    }
    auto bbox = image.getBBox();
    for (auto const &spanIter : *spanSet) {
        auto y = spanIter.getY() - image.getY0();
        if (static_cast<std::size_t>(y + image.getY0()) < bbox.getMinY() + margin ||
            static_cast<std::size_t>(y + image.getY0()) > bbox.getMaxY() - margin) {
            continue;
        }
        for (auto x = spanIter.getMinX() - image.getX0(); x <= spanIter.getMaxX() - image.getX0(); ++x) {
            if (static_cast<std::size_t>(x + image.getX0()) < (bbox.getMinX() + margin) ||
                static_cast<std::size_t>(x + image.getX0()) > (bbox.getMaxX() - margin)) {
                continue;
            }
            auto val = image(x, y);
            if (polarity) {  // look for +ve peaks
                if (image(x - 1, y + 1) > val || image(x, y + 1) > val || image(x + 1, y + 1) > val ||
                    image(x - 1, y) > val || image(x + 1, y) > val || image(x - 1, y - 1) > val ||
                    image(x, y - 1) > val || image(x + 1, y - 1) > val) {
                    continue;
                }
            } else {  // look for -ve "peaks" (pits)
                if (image(x - 1, y + 1) < val || image(x, y + 1) < val || image(x + 1, y + 1) < val ||
                    image(x - 1, y) < val || image(x + 1, y) < val || image(x - 1, y - 1) < val ||
                    image(x, y - 1) < val || image(x + 1, y - 1) < val) {
                    continue;
                }
            }

            foot.addPeak(x + image.getX0(), y + image.getY0(), val);
        }
    }
}

template <typename ImageT>
class FindMaxInFootprint {
public:
    explicit FindMaxInFootprint(bool polarity)
            : _polarity(polarity),
              _x(0),
              _y(0),
              _min(std::numeric_limits<double>::max()),
              _max(-std::numeric_limits<double>::max()) {}

    void operator()(lsst::geom::Point2I const &point, ImageT const &val) {
        if (_polarity) {
            if (val > _max) {
                _max = val;
                _x = point.getX();
                _y = point.getY();
            }
        } else {
            if (val < _min) {
                _min = val;
                _x = point.getX();
                _y = point.getY();
            }
        }
    }

    void addRecord(Footprint &foot) const { foot.addPeak(_x, _y, _polarity ? _max : _min); }

private:
    bool _polarity;
    int _x, _y;
    double _min, _max;
};

template <typename ImageT, typename ThresholdT>
void findPeaks(std::shared_ptr<Footprint> foot, ImageT const &img, bool polarity, ThresholdT) {
    findPeaksInFootprint(img, polarity, foot->getPeaks(), *foot, 1);

    // We use getInternal() here to get the vector of shared_ptr that Catalog uses internally,
    // which causes the STL algorithm to copy pointers instead of PeakRecords (which is what
    // it'd try to do if we passed Catalog's own iterators).
    std::stable_sort(foot->getPeaks().getInternal().begin(), foot->getPeaks().getInternal().end(),
                     SortPeaks());

    if (foot->getPeaks().empty()) {
        FindMaxInFootprint<typename ImageT::Pixel> maxFinder(polarity);
        foot->getSpans()->applyFunctor(maxFinder, ndarray::ndImage(img.getArray(), img.getXY0()));
        maxFinder.addRecord(*foot);
    }
}

// No need to search for peaks when processing a Mask
template <typename ImageT>
void findPeaks(std::shared_ptr<Footprint>, ImageT const &, bool, ThresholdBitmask_traits) {
    ;
}
}  // namespace

/*
 * Functions to determine if a pixel's in a Footprint
 */
template <typename ImagePixelT, typename IterT>
static inline bool inFootprint(ImagePixelT pixVal, IterT, bool polarity, double thresholdVal,
                               ThresholdLevel_traits) {
    return (polarity ? pixVal : -pixVal) >= thresholdVal;
}

template <typename ImagePixelT, typename IterT>
static inline bool inFootprint(ImagePixelT pixVal, IterT var, bool polarity, double thresholdVal,
                               ThresholdPixelLevel_traits) {
    return (polarity ? pixVal : -pixVal) >= thresholdVal * ::sqrt(*var);
}

template <typename ImagePixelT, typename IterT>
static inline bool inFootprint(ImagePixelT pixVal, IterT, bool, double thresholdVal,
                               ThresholdBitmask_traits) {
    return (pixVal & static_cast<long>(thresholdVal));
}

/*
 * Advance the x_iterator to the variance image, when relevant (it may be NULL otherwise)
 */
template <typename IterT>
static inline IterT advancePtr(IterT varPtr, Threshold_traits) {
    return varPtr;
}

template <typename IterT>
static inline IterT advancePtr(IterT varPtr, ThresholdPixelLevel_traits) {
    return varPtr + 1;
}

/*
 * Here's the working routine for the FootprintSet constructors; see documentation
 * of the constructors themselves
 */
template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename ThresholdTraitT>
static void findFootprints(
        typename FootprintSet::FootprintList *_footprints,  // Footprints
        lsst::geom::Box2I const &_region,                   // BBox of pixels that are being searched
        image::ImageBase<ImagePixelT> const &img,           // Image to search for objects
        image::Image<VariancePixelT> const *var,            // img's variance
        double const footprintThreshold,                    // threshold value for footprint
        double const includeThresholdMultiplier,  // threshold (relative to footprintThreshold) for inclusion
        bool const polarity,                      // if false, search _below_ thresholdVal
        int const npixMin,                        // minimum number of pixels in an object
        bool const setPeaks                       // should I set the Peaks list?
) {
    int id;       /* object ID */
    int in_span;  /* object ID of current IdSpan */
    int nobj = 0; /* number of objects found */
    int x0 = 0;   /* unpacked from a IdSpan */

    double includeThreshold = footprintThreshold * includeThresholdMultiplier;  // Threshold for inclusion

    int const row0 = img.getY0();
    int const col0 = img.getX0();
    int const height = img.getHeight();
    int const width = img.getWidth();
    /*
     * Storage for arrays that identify objects by ID. We want to be able to
     * refer to idp[-1] and idp[width], hence the (width + 2)
     */
    std::vector<int> id1(width + 2);
    std::fill(id1.begin(), id1.end(), 0);
    std::vector<int> id2(width + 2);
    std::fill(id2.begin(), id2.end(), 0);
    std::vector<int>::iterator idc = id1.begin() + 1;  // object IDs in current/
    std::vector<int>::iterator idp = id2.begin() + 1;  //                       previous row

    std::vector<int> aliases;          // aliases for initially disjoint parts of Footprints
    aliases.reserve(1 + height / 20);  // initial size of aliases

    std::vector<IdSpan> spans;          // y:x0,x1 for objects
    spans.reserve(aliases.capacity());  // initial size of spans

    aliases.push_back(0);  // 0 --> 0
                           /*
                            * Go through image identifying objects
                            */
    typedef typename image::Image<ImagePixelT>::x_iterator x_iterator;
    typedef typename image::Image<VariancePixelT>::x_iterator x_var_iterator;

    in_span = 0;  // not in a span
    for (int y = 0; y != height; ++y) {
        if (idc == id1.begin() + 1) {
            idc = id2.begin() + 1;
            idp = id1.begin() + 1;
        } else {
            idc = id1.begin() + 1;
            idp = id2.begin() + 1;
        }
        std::fill_n(idc - 1, width + 2, 0);

        in_span = 0;                                     /* not in a span */
        bool good = (includeThresholdMultiplier == 1.0); /* Span exceeds the threshold? */

        x_iterator pixPtr = img.row_begin(y);
        x_var_iterator varPtr = (var == NULL) ? NULL : var->row_begin(y);
        for (int x = 0; x < width; ++x, ++pixPtr, varPtr = advancePtr(varPtr, ThresholdTraitT())) {
            ImagePixelT const pixVal = *pixPtr;

            if (isBadPixel(pixVal) ||
                !inFootprint(pixVal, varPtr, polarity, footprintThreshold, ThresholdTraitT())) {
                if (in_span) {
                    spans.emplace_back(in_span, y, x0, x - 1, good);

                    in_span = 0;
                    good = false;
                }
            } else { /* a pixel to fix */
                if (idc[x - 1] != 0) {
                    id = idc[x - 1];
                } else if (idp[x - 1] != 0) {
                    id = idp[x - 1];
                } else if (idp[x] != 0) {
                    id = idp[x];
                } else if (idp[x + 1] != 0) {
                    id = idp[x + 1];
                } else {
                    id = ++nobj;
                    aliases.push_back(id);
                }

                idc[x] = id;
                if (!in_span) {
                    x0 = x;
                    in_span = id;
                }
                /*
                 * Do we need to merge ID numbers? If so, make suitable entries in aliases[]
                 */
                if (idp[x + 1] != 0 && idp[x + 1] != id) {
                    aliases[resolve_alias(aliases, idp[x + 1])] = resolve_alias(aliases, id);

                    idc[x] = id = idp[x + 1];
                }

                if (!good && inFootprint(pixVal, varPtr, polarity, includeThreshold, ThresholdTraitT())) {
                    good = true;
                }
            }
        }

        if (in_span) {
            spans.emplace_back(in_span, y, x0, width - 1, good);
        }
    }
    /*
     * Resolve aliases; first alias chains, then the IDs in the spans
     */
    for (unsigned int i = 0; i < spans.size(); i++) {
        spans[i].id = resolve_alias(aliases, spans[i].id);
    }
    /*
     * Sort spans by ID, so we can sweep through them once
     */
    if (spans.size() > 0) {
        std::sort(spans.begin(), spans.end(), IdSpanCompare());
    }
    /*
     * Build Footprints from spans
     */
    unsigned int i0;  // initial value of i
    if (spans.size() > 0) {
        id = spans[0].id;
        i0 = 0;
        for (unsigned int i = 0; i <= spans.size(); i++) {  // <= size to catch the last object
            if (i == spans.size() || spans[i].id != id) {
                bool good = false;  // Span includes pixel sufficient to include footprint in set?
                std::vector<geom::Span> tempSpanList;
                for (; i0 < i; i0++) {
                    good |= spans[i0].good;
                    tempSpanList.push_back(
                            geom::Span(spans[i0].y + row0, spans[i0].x0 + col0, spans[i0].x1 + col0));
                }
                auto tempSpanSet = std::make_shared<geom::SpanSet>(std::move(tempSpanList));
                auto fp = std::make_shared<Footprint>(tempSpanSet, _region);

                if (good && fp->getArea() >= static_cast<std::size_t>(npixMin)) {
                    _footprints->push_back(fp);
                }
            }

            if (i < spans.size()) {
                id = spans[i].id;
            }
        }
    }
    /*
     * Find all peaks within those Footprints
     */
    if (setPeaks) {
        typedef FootprintSet::FootprintList::iterator fiterator;
        for (fiterator ptr = _footprints->begin(), end = _footprints->end(); ptr != end; ++ptr) {
            findPeaks(*ptr, img, polarity, ThresholdTraitT());
        }
    }
}

template <typename ImagePixelT>
FootprintSet::FootprintSet(image::Image<ImagePixelT> const &img, Threshold const &threshold,
                           int const npixMin, bool const setPeaks)
        : _footprints(new FootprintList()), _region(img.getBBox()) {
    typedef float VariancePixelT;

    findFootprints<ImagePixelT, image::MaskPixel, VariancePixelT, ThresholdLevel_traits>(
            _footprints.get(), _region, img, NULL, threshold.getValue(img), threshold.getIncludeMultiplier(),
            threshold.getPolarity(), npixMin, setPeaks);
}

// NOTE: not a template to appease swig (see note by instantiations at bottom)

template <typename MaskPixelT>
FootprintSet::FootprintSet(image::Mask<MaskPixelT> const &msk, Threshold const &threshold, int const npixMin)
        : _footprints(new FootprintList()), _region(msk.getBBox()) {
    switch (threshold.getType()) {
        case Threshold::BITMASK:
            findFootprints<MaskPixelT, MaskPixelT, float, ThresholdBitmask_traits>(
                    _footprints.get(), _region, msk, NULL, threshold.getValue(),
                    threshold.getIncludeMultiplier(), threshold.getPolarity(), npixMin, false);
            break;

        case Threshold::VALUE:
            findFootprints<MaskPixelT, MaskPixelT, float, ThresholdLevel_traits>(
                    _footprints.get(), _region, msk, NULL, threshold.getValue(),
                    threshold.getIncludeMultiplier(), threshold.getPolarity(), npixMin, false);
            break;

        default:
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                              "You must specify a numerical threshold value with a Mask");
    }
}

template <typename ImagePixelT, typename MaskPixelT>
FootprintSet::FootprintSet(const image::MaskedImage<ImagePixelT, MaskPixelT> &maskedImg,
                           Threshold const &threshold, std::string const &planeName, int const npixMin,
                           bool const setPeaks)
        : _footprints(new FootprintList()),
          _region(lsst::geom::Point2I(maskedImg.getX0(), maskedImg.getY0()),
                  lsst::geom::Extent2I(maskedImg.getWidth(), maskedImg.getHeight())) {
    typedef typename image::MaskedImage<ImagePixelT, MaskPixelT>::Variance::Pixel VariancePixelT;
    // Find the Footprints
    switch (threshold.getType()) {
        case Threshold::PIXEL_STDEV:
            findFootprints<ImagePixelT, MaskPixelT, VariancePixelT, ThresholdPixelLevel_traits>(
                    _footprints.get(), _region, *maskedImg.getImage(), maskedImg.getVariance().get(),
                    threshold.getValue(maskedImg), threshold.getIncludeMultiplier(), threshold.getPolarity(),
                    npixMin, setPeaks);
            break;
        default:
            findFootprints<ImagePixelT, MaskPixelT, VariancePixelT, ThresholdLevel_traits>(
                    _footprints.get(), _region, *maskedImg.getImage(), maskedImg.getVariance().get(),
                    threshold.getValue(maskedImg), threshold.getIncludeMultiplier(), threshold.getPolarity(),
                    npixMin, setPeaks);
            break;
    }
    // Set Mask if requested
    if (planeName == "") {
        return;
    }
    //
    // Define the maskPlane
    //
    const std::shared_ptr<image::Mask<MaskPixelT>> mask = maskedImg.getMask();
    mask->addMaskPlane(planeName);

    MaskPixelT const bitPlane = mask->getPlaneBitMask(planeName);
    //
    // Set the bits where objects are detected
    //
    for (auto const &fIter : *_footprints) {
        fIter->getSpans()->setMask(*(maskedImg.getMask()), bitPlane);
    }
}

FootprintSet::FootprintSet(lsst::geom::Box2I region)
        : _footprints(std::make_shared<FootprintList>()), _region(region) {}

FootprintSet::FootprintSet(FootprintSet const &rhs) : _footprints(new FootprintList), _region(rhs._region) {
    _footprints->reserve(rhs._footprints->size());
    for (FootprintSet::FootprintList::const_iterator ptr = rhs._footprints->begin(),
                                                     end = rhs._footprints->end();
         ptr != end; ++ptr) {
        _footprints->push_back(std::make_shared<Footprint>(**ptr));
    }
}

// Delegate to copy-constructor for backward-compatibility
FootprintSet::FootprintSet(FootprintSet &&rhs) : FootprintSet(rhs) {}

FootprintSet &FootprintSet::operator=(FootprintSet const &rhs) {
    FootprintSet tmp(rhs);
    swap(tmp);  // See Meyers, Effective C++, Item 11
    return *this;
}

// Delegate to copy-assignment for backward-compatibility
FootprintSet &FootprintSet::operator=(FootprintSet &&rhs) { return *this = rhs; }

FootprintSet::~FootprintSet() = default;

void FootprintSet::merge(FootprintSet const &rhs, int tGrow, int rGrow, bool isotropic) {
    FootprintControl const ctrl(true, isotropic);
    FootprintSet fs = mergeFootprintSets(*this, tGrow, rhs, rGrow, ctrl);
    swap(fs);  // Swap the new FootprintSet into place
}

void FootprintSet::setRegion(lsst::geom::Box2I const &region) {
    _region = region;

    for (FootprintSet::FootprintList::iterator ptr = _footprints->begin(), end = _footprints->end();
         ptr != end; ++ptr) {
        (*ptr)->setRegion(region);
    }
}

FootprintSet::FootprintSet(FootprintSet const &rhs, int r, bool isotropic)
        : _footprints(new FootprintList), _region(rhs._region) {
    if (r == 0) {
        FootprintSet fs = rhs;
        swap(fs);  // Swap the new FootprintSet into place
        return;
    } else if (r < 0) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                          (boost::format("I cannot grow by negative numbers: %d") % r).str());
    }

    FootprintControl const ctrl(true, isotropic);
    FootprintSet fs = mergeFootprintSets(FootprintSet(rhs.getRegion()), 0, rhs, r, ctrl);
    swap(fs);  // Swap the new FootprintSet into place
}

FootprintSet::FootprintSet(FootprintSet const &rhs, int ngrow, FootprintControl const &ctrl)
        : _footprints(new FootprintList), _region(rhs._region) {
    if (ngrow == 0) {
        FootprintSet fs = rhs;
        swap(fs);  // Swap the new FootprintSet into place
        return;
    } else if (ngrow < 0) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                          str(boost::format("I cannot grow by negative numbers: %d") % ngrow));
    }

    FootprintSet fs = mergeFootprintSets(FootprintSet(rhs.getRegion()), 0, rhs, ngrow, ctrl);
    swap(fs);  // Swap the new FootprintSet into place
}

FootprintSet::FootprintSet(FootprintSet const &fs1, FootprintSet const &fs2, bool const)
        : _footprints(new FootprintList()), _region(fs1._region) {
    _region.include(fs2._region);
    throw LSST_EXCEPT(pex::exceptions::LogicError, "NOT IMPLEMENTED");
}

std::shared_ptr<image::Image<FootprintIdPixel>> FootprintSet::insertIntoImage() const {
    auto im = std::make_shared<image::Image<FootprintIdPixel>>(_region);
    *im = 0;

    FootprintIdPixel id = 0;
    for (auto const &fIter : *_footprints) {
        id++;
        fIter->getSpans()->applyFunctor(setIdImage<FootprintIdPixel>(id), *im);
    }

    return im;
}

template <typename ImagePixelT, typename MaskPixelT>
void FootprintSet::makeHeavy(image::MaskedImage<ImagePixelT, MaskPixelT> const &mimg,
                             HeavyFootprintCtrl const *ctrl) {
    HeavyFootprintCtrl ctrl_s = HeavyFootprintCtrl();

    if (!ctrl) {
        ctrl = &ctrl_s;
    }

    for (FootprintList::iterator ptr = _footprints->begin(), end = _footprints->end(); ptr != end; ++ptr) {
        ptr->reset(new HeavyFootprint<ImagePixelT, MaskPixelT>(**ptr, mimg, ctrl));
    }
}

void FootprintSet::makeSources(afw::table::SourceCatalog &cat) const {
    for (FootprintList::const_iterator i = _footprints->begin(); i != _footprints->end(); ++i) {
        std::shared_ptr<afw::table::SourceRecord> r = cat.addNew();
        r->setFootprint(*i);
    }
}

    //
    // Explicit instantiations
    //

#ifndef DOXYGEN

#define INSTANTIATE(PIXEL)                                                                              \
    template FootprintSet::FootprintSet(image::Image<PIXEL> const &, Threshold const &, int const,      \
                                        bool const);                                                    \
    template FootprintSet::FootprintSet(image::MaskedImage<PIXEL, image::MaskPixel> const &,            \
                                        Threshold const &, std::string const &, int const, bool const); \
    template void FootprintSet::makeHeavy(image::MaskedImage<PIXEL, image::MaskPixel> const &,          \
                                          HeavyFootprintCtrl const *)

template FootprintSet::FootprintSet(image::Mask<image::MaskPixel> const &, Threshold const &, int const);

template void FootprintSet::setMask(image::Mask<image::MaskPixel> *, std::string const &);
template void FootprintSet::setMask(std::shared_ptr<image::Mask<image::MaskPixel>>, std::string const &);

INSTANTIATE(std::uint16_t);
INSTANTIATE(int);
INSTANTIATE(float);
INSTANTIATE(double);
}  // namespace detection
}  // namespace afw
}  // namespace lsst

#endif  // !DOXYGEN
