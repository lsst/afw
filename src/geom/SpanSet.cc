/*
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */

#include <algorithm>
#include <iterator>
#include "lsst/afw/geom/SpanSet.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/geom/ellipses/PixelRegion.h"
#include "lsst/afw/image/LsstImageTypes.h"

namespace lsst {
namespace afw {
namespace geom {
namespace {

/* These classes are used in the erode operator to quickly calculate the
* contents of the shrunken SpanSet
*/

struct PrimaryRun {
    int m, y, xmin, xmax;
};

bool comparePrimaryRun(PrimaryRun const& first, PrimaryRun const& second) {
    if (first.y != second.y) {
        return first.y < second.y;
    } else if (first.m != second.m) {
        return first.m < second.m;
    } else {
        return first.xmin < second.xmin;
    }
}

class ComparePrimaryRunY {
public:
    bool operator()(PrimaryRun const& pr, int yval) { return pr.y < yval; }
    bool operator()(int yval, PrimaryRun const& pr) { return yval < pr.y; }
};

class ComparePrimaryRunM {
public:
    bool operator()(PrimaryRun const& pr, int mval) { return pr.m < mval; }
    bool operator()(int mval, PrimaryRun const& pr) { return mval < pr.m; }
};

/* Determine if two spans overlap
 *
 * a First Span in comparison
 * b Second Span in comparison
 * compareY a boolean to control if the comparison takes into account the y position of the spans
 */
bool spansOverlap(Span const& a, Span const& b, bool compareY = true) {
    bool yTruth = true;
    if (compareY) {
        yTruth = a.getY() == b.getY();
    }
    return (yTruth && ((a.getMaxX() >= b.getMinX() && a.getMinX() <= b.getMinX()) ||
                       (b.getMaxX() >= a.getMinX() && b.getMinX() <= a.getMinX())))
                   ? true
                   : false;
}

/* Determine if two spans are contiguous, that is they can overlap or the end of one span is
 * one pixel before the beginning of the next
 *
 * a First Span in comparison
 * b Second Span in comparison
 * compareY a boolean to control if the comparison takes into account the y position of the spans
 */
bool spansContiguous(Span const& a, Span const& b, bool compareY = true) {
    bool yTruth(true);
    if (compareY) {
        yTruth = a.getY() == b.getY();
    }
    return (yTruth && ((a.getMaxX() + 1 >= b.getMinX() && a.getMinX() <= b.getMinX()) ||
                       (b.getMaxX() + 1 >= a.getMinX() && b.getMinX() <= a.getMinX())))
                   ? true
                   : false;
}

}  // namespace

// Default constructor, creates a null SpanSet which may be useful for
// comparisons
SpanSet::SpanSet() : _spanVector(), _bbox(), _area(0) {}

// Construct a SpanSet from a Box2I object
SpanSet::SpanSet(Box2I const& box) : _bbox(box), _area(box.getArea()) {
    int beginY = box.getMinY();

    int beginX = box.getMinX();
    int maxX = box.getMaxX();

    for (int i = beginY; i < _bbox.getEndY(); ++i) {
        _spanVector.push_back(Span(i, beginX, maxX));
    }
}

// Construct a SpanSet from a std vector by copying
SpanSet::SpanSet(std::vector<Span> const& vec, bool normalize) : _spanVector(vec) {
    // If the incoming vector is zero, should create an empty spanSet
    if (_spanVector.empty()) {
        _bbox = Box2I();
        _area = 0;
    } else {
        if (normalize) {
            _runNormalize();
        }
        _initialize();
    }
}

// Construct a SpanSet from a std vector by moving
SpanSet::SpanSet(std::vector<Span>&& vec, bool normalize) : _spanVector(std::move(vec)) {
    // If the incoming vector is zero, should create an empty SpanSet
    if (_spanVector.size() == 0) {
        _bbox = Box2I();
        _area = 0;
    } else {
        if (normalize) {
            _runNormalize();
        }
        _initialize();
    }
}

void SpanSet::_runNormalize() {
    // This bit of code will only be executed with a non-empty _spanVector internally
    // and is not accessable from outside the class

    // Ensure the span set is sorted according to Span < operator
    std::sort(_spanVector.begin(), _spanVector.end());

    // Now that the span is sorted, overlapping spans should be combined
    // Start from 1 as the comparison will always be with the previous element
    for (auto iter = ++(_spanVector.begin()); iter != _spanVector.end(); iter++) {
        if (spansContiguous(*std::prev(iter, 1), *iter)) {
            // These spans overlap, create one new one, covering the whole expanse
            *(iter - 1) = Span((*(iter - 1)).getY(), std::min((*(iter - 1)).getMinX(), (*iter).getMinX()),
                               std::max((*(iter - 1)).getMaxX(), (*iter).getMaxX()));
            // Erase the current Span, as it is now contained in the previous element
            iter = _spanVector.erase(iter);
            // Move the iterator back one to make sure the element gets run in the loop
            --iter;
        }
    }
}

void SpanSet::_initialize() {
    /* This function exists to handle common functionality for most of the constructors. It  calculates the
     * bounding box for the SpanSet, and the area covered by the SpanSet
     */

    /* Because the array is sorted, the minimum and maximum values for Y will
       be in the first and last elements, only need to find the min and max X
       values */

    // This bit of code will only be executed with a non-empty _spanVector internally
    // and is not accessable from outside the class

    int minX = _spanVector[0].getMinX();
    int maxX = _spanVector[0].getMaxX();
    _area = 0;

    for (const auto& span : _spanVector) {
        if (span.getMinX() < minX) {
            minX = span.getMinX();
        }
        if (span.getMaxX() > maxX) {
            maxX = span.getMaxX();
        }
        // Plus one, because end point is inclusive
        _area += span.getMaxX() - span.getMinX() + 1;
    }
    _bbox = Box2I(Point2I(minX, _spanVector.front().getY()), Point2I(maxX, _spanVector.back().getY()));
}

// Getter for the area property
std::size_t SpanSet::getArea() const { return _area; }

// Getter for the bounding box of the SpanSet
Box2I SpanSet::getBBox() const { return _bbox; }

/* Here is a description of how the _makeLabels and _label function works. In the
   _makeLabels function, a vector is created with the same number of elements as
   spans in the SpanSet. This vector will contain the label for each Span in the
   SpanSet. These labels correspond to which connected region the span falls in.
   If the whole SpanSet is connected than there will only be one region and every
   Span will be labeled with a 1. If there are two regions (i.e. there are points
   in between not contained in the SpanSet) then Some of the Spans will be labeled
   1, and the rest labeled 2.

   The function loops over all the Spans in the Spanset. If the Span has not been
   labeled, the _label function is called, with the current span, the vector
   containing the labels, and the label for the current region under consideration.

   The _label function will then loop over all Spans in the SpanSet looking for
   rows adjacent to the currently being labeled Span which overlap in the x
   dimension and have not yet been labeled, and mark them with the current label,
   and recursively call the _label function with the adjacent row as the new Span
   under consideration. This results in all Spans which overlap each other in x
   being labeled with the current label.

   Once _label reaches the end of Spans which meet it's criteria, control falls
   back to the _makeLabels function and the current label is incremented. The
   function then moves onto the next Span in the SpanSet which has not been
   labeled. If all spans were labeled in the recursive call the loop falls to the
   end, and the label vector and one past the last label number (number of labels +1)
   are returned to the caller. If _makeLabels finds a Span that has not been labeled,
   the next label is assigned and the process is repeated until all Spans have
   been labeled.
 */

void SpanSet::_label(Span const& spn, std::vector<std::size_t>& labelVector, std::size_t currentLabel) const {
    std::size_t index = 0;
    // Loop over all Spans and if it is before or after the current span then consider it
    for (auto const& currentSpan : _spanVector) {
        if (!labelVector[index] &&
            (currentSpan.getY() == spn.getY() - 1 || currentSpan.getY() == spn.getY() + 1)) {
            if (spansOverlap(spn, currentSpan, /* compareY= */ false)) {
                labelVector[index] = currentLabel;
                _label(currentSpan, labelVector, currentLabel);
            }
        }
        ++index;
    }
}

std::pair<std::vector<std::size_t>, std::size_t> SpanSet::_makeLabels() const {
    std::vector<std::size_t> labelVector(_spanVector.size(), 0);
    std::size_t currentLabel = 1;
    std::size_t index = 0;
    for (auto const& currentSpan : _spanVector) {
        if (!labelVector[index]) {
            labelVector[index] = currentLabel;
            _label(currentSpan, labelVector, currentLabel);
            /* At this point we have recursed enough to reach all of the connected
             * region, and should increment the label such that any spans not
             * labeled in the first loop will get a new value. If all of the spans
             * were connected in the first pass, the rest of this loop will just
             * fall though.
             */
            ++currentLabel;
        }
        ++index;
    }
    return std::pair<std::vector<std::size_t>, std::size_t>(labelVector, currentLabel);
}

bool SpanSet::isContiguous() const {
    auto labeledPair = _makeLabels();
    // Here we want to check if there is only one label. Since _makeLabels always increments
    // at the end of each loop, we need to compare against the number 2. I.e. if everything
    // gets labeled 1, the label counter will increment to 2, and the if statement will always
    // be false and _makeLabels will fall through to the end and return to here.
    return labeledPair.second <= 2;
}

std::vector<std::shared_ptr<SpanSet>> SpanSet::split() const {
    auto labeledPair = _makeLabels();
    auto labels = labeledPair.first;
    auto numberOfLabels = labeledPair.second;
    std::vector<std::shared_ptr<SpanSet>> subRegions;

    // if numberOfLabels is 1, that means a null SpanSet is being operated on,
    // and we should return like
    if (numberOfLabels == 1) {
        subRegions.push_back(std::make_shared<SpanSet>());
        return subRegions;
    }
    subRegions.reserve(numberOfLabels - 1);
    for (std::size_t i = 1; i < numberOfLabels; ++i) {
        std::vector<Span> tempVec;
        for (std::size_t j = 0; j < _spanVector.size(); ++j) {
            if (labels[j] == i) {
                tempVec.push_back(_spanVector[j]);
            }
        }
        subRegions.push_back(std::make_shared<SpanSet>(std::move(tempVec)));
    }
    return subRegions;
}

std::shared_ptr<SpanSet> SpanSet::findEdgePixels() const {
    image::Mask<image::MaskPixel> tempMask(getBBox());
    setMask(tempMask, static_cast<image::MaskPixel>(1));
    auto erodedSpanSet = eroded(1, Stencil::CIRCLE);
    erodedSpanSet->clearMask(tempMask, static_cast<image::MaskPixel>(1));
    return SpanSet::fromMask(tempMask);
}

std::shared_ptr<SpanSet> SpanSet::shiftedBy(int x, int y) const {
    // Function to create a new SpanSet which is a copy of this, shifted by x and y
    return makeShift(x, y);
}

std::shared_ptr<SpanSet> SpanSet::shiftedBy(Extent2I const& offset) const {
    // Function to create a new SpanSet which is a copy of this, shifted by an extent object
    return makeShift(offset.getX(), offset.getY());
}

std::shared_ptr<SpanSet> SpanSet::makeShift(int x, int y) const {
    // Implementation method common to all overloads of the shiftedBy method
    std::vector<Span> tempVec;
    tempVec.reserve(_spanVector.size());
    for (auto const& spn : _spanVector) {
        tempVec.push_back(Span(spn.getY() + y, spn.getMinX() + x, spn.getMaxX() + x));
    }
    return std::make_shared<SpanSet>(std::move(tempVec), false);
}

std::shared_ptr<SpanSet> SpanSet::clippedTo(Box2I const& box) const {
    /* Return a copy of the current SpanSet but only with values which are contained within
     * the supplied box
     */
    std::vector<Span> tempVec;
    for (auto const& spn : _spanVector) {
        if (spn.getY() >= box.getMinY() && spn.getY() <= box.getMaxY() &&
            spansOverlap(spn, Span(spn.getY(), box.getMinX(), box.getMaxX()))) {
            tempVec.push_back(Span(spn.getY(), std::max(box.getMinX(), spn.getMinX()),
                                   std::min(box.getMaxX(), spn.getMaxX())));
        }
    }
    return std::make_shared<SpanSet>(std::move(tempVec), false);
}

bool SpanSet::overlaps(SpanSet const& other) const {
    // Function to check if two SpanSets overlap
    for (auto const& otherSpan : other) {
        for (auto const& spn : _spanVector) {
            if (spansOverlap(otherSpan, spn)) {
                return true;
            }
        }
    }
    return false;
}

bool SpanSet::contains(SpanSet const& other) const {
    // Handle null SpanSet passed as other
    if (other.empty()) {
        return false;
    }
    // Function to check if a SpanSet is entirely contained within this
    for (auto const& otherSpn : other) {
        std::size_t counter = 0;
        for (auto const& spn : _spanVector) {
            // Check that the end points of the span from other are contained in the
            // span from this
            if (spn.contains(Point2I(otherSpn.getMinX(), otherSpn.getY())) &&
                spn.contains(Point2I(otherSpn.getMaxX(), otherSpn.getY()))) {
                ++counter;
            }
        }
        // if counter is equal to zero, then the current span from other is not
        // contained in any span in this, and the function should return false
        // short circuiting any other spans from other
        if (counter == 0) {
            return false;
        }
    }
    return true;
}

bool SpanSet::contains(Point2I const& point) const {
    // Check to see if a given point is found within any spans in this
    for (auto& spn : _spanVector) {
        if (spn.contains(point)) {
            return true;
        }
    }
    return false;
}

Point2D SpanSet::computeCentroid() const {
    // Find the centroid of the SpanSet
    std::size_t n = 0;
    double xc = 0, yc = 0;
    for (auto const& spn : _spanVector) {
        int const y = spn.getY();
        int const x0 = spn.getMinX();
        int const x1 = spn.getMaxX();
        int const npix = x1 - x0 + 1;

        n += npix;
        xc += npix * 0.5 * (x1 + x0);
        yc += npix * y;
    }
    assert(n == _area);

    return Point2D(xc / _area, yc / _area);
}

ellipses::Quadrupole SpanSet::computeShape() const {
    // Compute the shape of the SpanSet
    Point2D cen = computeCentroid();
    double const xc = cen.getX();
    double const yc = cen.getY();

    double sumxx = 0, sumxy = 0, sumyy = 0;
    for (auto const& spn : _spanVector) {
        int const y = spn.getY();
        int const x0 = spn.getX0();
        int const x1 = spn.getX1();
        int const npix = x1 - x0 + 1;

        for (int x = x0; x <= x1; ++x) {
            sumxx += (x - xc) * (x - xc);
        }
        sumxy += npix * (0.5 * (x1 + x0) - xc) * (y - yc);
        sumyy += npix * (y - yc) * (y - yc);
    }

    return ellipses::Quadrupole(sumxx / _area, sumyy / _area, sumxy / _area);
}

std::shared_ptr<SpanSet> SpanSet::dilated(int r, Stencil s) const {
    // Return a dilated SpanSet made with the given stencil, by creating a SpanSet
    // from the stencil and forwarding to the appropriate overloaded method
    std::shared_ptr<SpanSet> stencilToSpanSet = fromShape(r, s);
    return dilated(*stencilToSpanSet);
}

std::shared_ptr<SpanSet> SpanSet::dilated(SpanSet const& other) const {
    // Handle a null SpanSet nothing should be dilated
    if (other.size() == 0) {
        return std::make_shared<SpanSet>(_spanVector.begin(), _spanVector.end(), false);
    }

    // Return a dilated Spanset by the given SpanSet
    std::vector<Span> tempVec;

    for (auto const& spn : _spanVector) {
        for (auto const& otherSpn : other) {
            int const xmin = spn.getMinX() + otherSpn.getMinX();
            int const xmax = spn.getMaxX() + otherSpn.getMaxX();
            int const yval = spn.getY() + otherSpn.getY();
            tempVec.push_back(Span(yval, xmin, xmax));
        }
    }
    // Allow constructor to handle merging adjacent and overlapping spans
    return std::make_shared<SpanSet>(std::move(tempVec));
}

std::shared_ptr<SpanSet> SpanSet::eroded(int r, Stencil s) const {
    // Return an eroded SpanSet made with the given stencil, by creating a SpanSet
    // from the stencil and forwarding to the appropriate overloaded method
    std::shared_ptr<SpanSet> stencilToSpanSet = fromShape(r, s);
    return eroded(*stencilToSpanSet);
}

std::shared_ptr<SpanSet> SpanSet::eroded(SpanSet const& other) const {
    // Handle a null SpanSet nothing should be eroded
    if (other.size() == 0 || this->size() == 0) {
        return std::make_shared<SpanSet>(_spanVector.begin(), _spanVector.end(), false);
    }

    // Return a SpanSet eroded by the given SpanSet
    std::vector<Span> tempVec;

    // Calculate all possible primary runs.
    std::vector<PrimaryRun> primaryRuns;
    for (auto const& spn : _spanVector) {
        int m = 0;
        for (auto const& otherSpn : other) {
            if ((otherSpn.getMaxX() - otherSpn.getMinX()) <= (spn.getMaxX() - spn.getMinX())) {
                int xmin = spn.getMinX() - otherSpn.getMinX();
                int xmax = spn.getMaxX() - otherSpn.getMaxX();
                int y = spn.getY() - otherSpn.getY();
                primaryRuns.push_back(PrimaryRun({m, y, xmin, xmax}));
            }
            ++m;
        }
    }

    // Iterate over the primary runs in such a way that we consider all values of m
    // for a given y, then all m for y+1 etc.
    std::sort(primaryRuns.begin(), primaryRuns.end(), comparePrimaryRun);

    for (int y = primaryRuns.front().y; y <= primaryRuns.back().y; ++y) {
        auto yRange = std::equal_range(primaryRuns.begin(), primaryRuns.end(), y, ComparePrimaryRunY());

        /* Discard runs for any value of y for which we find fewer groups than M,
         * the total Y range of the structuring element. This is step 3.1 of the
         * Kim et al. algorithm.
         */
        // Plus one because end points are inclusive
        auto otherYRange = other.back().getY() - other.front().getY() + 1;
        if (std::distance(yRange.first, yRange.second) < otherYRange) {
            continue;
        }

        /* "good" runs are those which are covered by each value of m, ie by each
         * row in the structuring element. Our algorithm will consider each value
         * of m in turn, gradually whittling down the list of good runs, then
         * finally convert the remainder into Spans.
         */
        std::list<PrimaryRun> goodRuns;

        for (int m = 0; m < otherYRange; ++m) {
            auto mRange = std::equal_range(yRange.first, yRange.second, m, ComparePrimaryRunM());
            if ((mRange.first == mRange.second)) {
                // If a particular m is missing, we known that this y contains
                // no good runs; this is equivalent to Kim et al. step 3.2.
                goodRuns.clear();
            } else {
                // Consolidate all primary runs at this m so that they don't overlap.
                std::list<PrimaryRun> candidateRuns;
                int startX = mRange.first->xmin;
                int endX = mRange.first->xmax;
                for (auto run = mRange.first + 1; run != mRange.second; ++run) {
                    if (run->xmin > endX) {
                        // Start of a new run
                        candidateRuns.push_back(PrimaryRun{m, y, startX, endX});
                        startX = run->xmin;
                        endX = run->xmax;
                    } else {
                        // Continuation of an existing run
                        endX = run->xmax;
                    }
                }
                candidateRuns.push_back(PrimaryRun{m, y, startX, endX});

                // Otherwise, calculate the intersection of candidate runs at
                // this m with good runs from all previous m.
                if (m == 0) {
                    // For m = 0 we have nothing to compare to; all runs are accepted
                    std::swap(goodRuns, candidateRuns);
                } else {
                    std::list<PrimaryRun> newlist;
                    for (auto& good : goodRuns) {
                        for (auto& cand : candidateRuns) {
                            int start = std::max(good.xmin, cand.xmin);
                            int end = std::min(good.xmax, cand.xmax);
                            if (end >= start) {
                                newlist.push_back(PrimaryRun({m, y, start, end}));
                            }
                        }
                    }
                    std::swap(newlist, goodRuns);
                }
            }
        }
        for (auto& run : goodRuns) {
            tempVec.push_back(Span(run.y, run.xmin, run.xmax));
        }
    }
    return std::make_shared<SpanSet>(std::move(tempVec));
}

bool SpanSet::operator==(SpanSet const& other) const {
    // Check the equivalence of this SpanSet with another
    return _spanVector == other._spanVector;
}

bool SpanSet::operator!=(SpanSet const& other) const {
    // Check the equivalence of this SpanSet with another
    return _spanVector != other._spanVector;
}

std::shared_ptr<SpanSet> SpanSet::fromShape(int r, Stencil s, Point2I offset) {
    // Create a SpanSet from a given Stencil
    std::vector<Span> tempVec;
    tempVec.reserve(2 * r + 1);
    switch (s) {
        case Stencil::CIRCLE:
            for (auto dy = -r; dy <= r; ++dy) {
                int dx = static_cast<int>(sqrt(r * r - dy * dy));
                tempVec.push_back(Span(dy + offset.getY(), -dx + offset.getX(), dx + offset.getX()));
            }
            break;
        case Stencil::MANHATTAN:
            for (auto dy = -r; dy <= r; ++dy) {
                int dx = r - abs(dy);
                tempVec.push_back(Span(dy + offset.getY(), -dx + offset.getX(), dx + offset.getX()));
            }
            break;
        case Stencil::BOX:
            for (auto dy = -r; dy <= r; ++dy) {
                int dx = r;
                tempVec.push_back(Span(dy + offset.getY(), -dx + offset.getX(), dx + offset.getX()));
            }
            break;
    }
    return std::make_shared<SpanSet>(std::move(tempVec), false);
}

std::shared_ptr<SpanSet> SpanSet::fromShape(ellipses::Ellipse const& ellipse) {
    ellipses::PixelRegion pr(ellipse);
    return std::make_shared<SpanSet>(pr.begin(), pr.end());
}

std::shared_ptr<SpanSet> SpanSet::intersect(SpanSet const& other) const {
    // Check if the bounding boxes overlap, if not return null SpanSet
    if (!_bbox.overlaps(other.getBBox())) {
        return std::make_shared<SpanSet>();
    }
    std::vector<Span> tempVec;
    for (auto const& spn : _spanVector) {
        for (auto const& otherSpn : other) {
            if (spansOverlap(spn, otherSpn)) {
                auto newMin = std::max(spn.getMinX(), otherSpn.getMinX());
                auto newMax = std::min(spn.getMaxX(), otherSpn.getMaxX());
                auto newSpan = Span(spn.getY(), newMin, newMax);
                tempVec.push_back(newSpan);
            }
        }
    }
    return std::make_shared<SpanSet>(std::move(tempVec));
}

std::shared_ptr<SpanSet> SpanSet::intersectNot(SpanSet const& other) const {
    // Check if the bounding boxes overlap, if not simply return a copy of this
    if (!getBBox().overlaps(other.getBBox())) {
        return std::make_shared<SpanSet>(this->begin(), this->end());
    }
    /* This function must find all the areas in this and not in other. These SpanSets
     * may be overlapping with this less than other a1|    b1|   a2|    b2|,
     * with this greater than other b1|    a1|     b2|    a2|,
     * with this containing other a1|    b1|     b2|     a2|,
     * or with other containing this  b1|    a1|    a2|   b2|
     */
    std::vector<Span> tempVec;
    bool added;
    for (auto const& spn : _spanVector) {
        added = false;
        for (auto const& otherSpn : other) {
            if (spansOverlap(spn, otherSpn)) {
                added = true;
                /* To handle one span containing the other, the spans will be added
                 * piecewise, and let the SpanSet constructor normalize spans which
                 * end up contiguous. In the case where this is contained in other,
                 * these statements will all be false, and nothing will be added,
                 * which is the expected behavior.
                 */
                if (spn.getMinX() < otherSpn.getMinX()) {
                    tempVec.push_back(Span(spn.getY(), spn.getMinX(), otherSpn.getMinX() - 1));
                }
                if (spn.getMaxX() > otherSpn.getMaxX()) {
                    tempVec.push_back(Span(spn.getY(), otherSpn.getMaxX() + 1, spn.getMaxX()));
                }
            }
        }
        /* If added is still zero, that means it did not overlap any of the spans in other
         * and should be included in the new span
         */
        if (!added) {
            tempVec.push_back(Span(spn));
        }
    }
    return std::make_shared<SpanSet>(std::move(tempVec));
}

std::shared_ptr<SpanSet> SpanSet::union_(SpanSet const& other) const {
    /* Simply include Spans from both SpanSets in a new vector and let the SpanSet
     * constructor normalize any of the SpanSets which may be contiguous
     */
    std::size_t combineSize = size() + other.size();
    std::vector<Span> tempVec;
    tempVec.reserve(combineSize);
    // Copy this
    tempVec.insert(tempVec.end(), _spanVector.begin(), _spanVector.end());
    // Copy other
    tempVec.insert(tempVec.end(), other.begin(), other.end());
    return std::make_shared<SpanSet>(std::move(tempVec));
}

std::shared_ptr<SpanSet> SpanSet::transformedBy(LinearTransform const& t) const {
    // Transform points in SpanSet by LinearTransform
    return transformedBy(AffineTransform(t));
}

std::shared_ptr<SpanSet> SpanSet::transformedBy(AffineTransform const& t) const {
    // Transform points in SpanSet by AffineTransform
    return transformedBy(AffineXYTransform(t));
}

std::shared_ptr<SpanSet> SpanSet::transformedBy(XYTransform const& t) const {
    // Transform points in SpanSet by XYTransform
    // Transform the original bounding box
    Box2D newBBoxD;
    newBBoxD.include(t.forwardTransform(Point2D(_bbox.getMinX(), _bbox.getMinY())));
    newBBoxD.include(t.forwardTransform(Point2D(_bbox.getMinX(), _bbox.getMaxY())));
    newBBoxD.include(t.forwardTransform(Point2D(_bbox.getMaxX(), _bbox.getMinY())));
    newBBoxD.include(t.forwardTransform(Point2D(_bbox.getMaxX(), _bbox.getMaxY())));

    Box2I newBBoxI(newBBoxD);

    std::vector<Span> tempVec;
    for (int y = newBBoxI.getBeginY(); y < newBBoxI.getEndY(); ++y) {
        bool inSpan = false;  // Are we in a span?
        int start = -1;       // Start of span

        for (int x = newBBoxI.getBeginX(); x < newBBoxI.getEndX(); ++x) {
            Point2D p = t.reverseTransform(Point2D(x, y));
            int const xSource = std::floor(0.5 + p.getX());
            int const ySource = std::floor(0.5 + p.getY());

            if (contains(Point2I(xSource, ySource))) {
                if (!inSpan) {
                    inSpan = true;
                    start = x;
                }
            } else if (inSpan) {
                inSpan = false;
                tempVec.push_back(Span(y, start, x - 1));
            }
        }
        if (inSpan) {
            tempVec.push_back(Span(y, start, newBBoxI.getMaxX()));
        }
    }
    return std::make_shared<SpanSet>(std::move(tempVec));
}

template <typename ImageT>
void SpanSet::setImage(image::Image<ImageT>& image, ImageT val, Box2I const& region, bool doClip) const {
    Box2I bbox;
    if (region.isEmpty()) {
        bbox = image.getBBox();
    } else {
        bbox = region;
    }
    auto setterFunc = [](Point2I const& point, ImageT& out, ImageT in) { out = in; };
    try {
        if (doClip) {
            auto tmpSpan = this->clippedTo(bbox);
            tmpSpan->applyFunctor(setterFunc, image, val);
        } else {
            applyFunctor(setterFunc, image, val);
        }
    } catch (pex::exceptions::OutOfRangeError e) {
        throw LSST_EXCEPT(pex::exceptions::OutOfRangeError,
                          "Footprint Bounds Outside image, set doClip to true");
    }
}

template <typename T>
void SpanSet::setMask(image::Mask<T>& target, T bitmask) const {
    // Use a lambda to set bits in a mask at the locations given by SpanSet
    auto targetArray = target.getArray();
    auto xy0 = target.getBBox().getMin();
    auto maskFunctor = [](Point2I const& point, typename details::ImageNdGetter<T, 2, 1>::Reference maskVal,
                          T bitmask) { maskVal |= bitmask; };
    applyFunctor(maskFunctor, ndarray::ndImage(targetArray, xy0), bitmask);
}

template <typename T>
void SpanSet::clearMask(image::Mask<T>& target, T bitmask) const {
    // Use a lambda to clear bits in a mask at the locations given by SpanSet
    auto targetArray = target.getArray();
    auto xy0 = target.getBBox().getMin();
    auto clearMaskFunctor = [](Point2I const& point,
                               typename details::ImageNdGetter<T, 2, 1>::Reference maskVal,
                               T bitmask) { maskVal &= ~bitmask; };
    applyFunctor(clearMaskFunctor, ndarray::ndImage(targetArray, xy0), bitmask);
}

template <typename T>
std::shared_ptr<SpanSet> SpanSet::intersect(image::Mask<T> const& other, T bitmask) const {
    auto comparator = [bitmask](T pixelValue) { return (pixelValue & bitmask); };
    auto spanSetFromMask = SpanSet::fromMask(other, comparator);
    return intersect(*spanSetFromMask);
}

template <typename T>
std::shared_ptr<SpanSet> SpanSet::intersectNot(image::Mask<T> const& other, T bitmask) const {
    auto comparator = [bitmask](T pixelValue) { return (pixelValue & bitmask); };
    auto spanSetFromMask = SpanSet::fromMask(other, comparator);
    return intersectNot(*spanSetFromMask);
}

template <typename T>
std::shared_ptr<SpanSet> SpanSet::union_(image::Mask<T> const& other, T bitmask) const {
    auto comparator = [bitmask](T pixelValue) { return (pixelValue & bitmask); };
    auto spanSetFromMask = fromMask(other, comparator);
    return union_(*spanSetFromMask);
}

namespace {
// Singleton helper class that manages the schema and keys for the persistence of SpanSets
class SpanSetPersistenceHelper {
public:
    table::Schema spanSetSchema;
    table::Key<int> spanY;
    table::Key<int> spanX0;
    table::Key<int> spanX1;

    static SpanSetPersistenceHelper const& get() {
        static SpanSetPersistenceHelper instance;
        return instance;
    }

    // No copying
    SpanSetPersistenceHelper(const SpanSetPersistenceHelper&) = delete;
    SpanSetPersistenceHelper& operator=(const SpanSetPersistenceHelper&) = delete;

    // No Moving
    SpanSetPersistenceHelper(SpanSetPersistenceHelper&&) = delete;
    SpanSetPersistenceHelper& operator=(SpanSetPersistenceHelper&&) = delete;

private:
    SpanSetPersistenceHelper()
            : spanSetSchema(),
              spanY(spanSetSchema.addField<int>("y", "The row of the span", "pixel")),
              spanX0(spanSetSchema.addField<int>("x0", "First column of span (inclusive)", "pixel")),
              spanX1(spanSetSchema.addField<int>("x1", "Second column of span (inclusive)", "pixel")) {
        spanSetSchema.getCitizen().markPersistent();
    }
};

std::string getSpanSetPersistenceName() { return "SpanSet"; }

class SpanSetFactory : public table::io::PersistableFactory {
public:
    virtual std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                         CatalogVector const& catalogs) const {
        // There should only be one catalog saved
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        // Get the catalog with the spans
        auto spansCatalog = catalogs.front();
        // Retrieve the keys that will be used to reference the catalog
        auto const& keys = SpanSetPersistenceHelper::get();
        // Construct a temporary container which will later be turned into the SpanSet
        std::vector<Span> tempVec;
        tempVec.reserve(spansCatalog.size());
        for (auto const& val : spansCatalog) {
            tempVec.push_back(Span(val.get(keys.spanY), val.get(keys.spanX0), val.get(keys.spanX1)));
        }
        return std::make_shared<SpanSet>(std::move(tempVec));
    }
    explicit SpanSetFactory(std::string const& name) : table::io::PersistableFactory(name) {}
};

// insert the factory into the registry (instantiating an instance is sufficient, because the code
// that does the work is in the base class ctor)
SpanSetFactory registration(getSpanSetPersistenceName());

}  // end anonymous

std::string SpanSet::getPersistenceName() const { return getSpanSetPersistenceName(); }

void SpanSet::write(OutputArchiveHandle& handle) const {
    auto const& keys = SpanSetPersistenceHelper::get();
    auto spanCat = handle.makeCatalog(keys.spanSetSchema);
    spanCat.reserve(size());
    for (auto const& val : *this) {
        auto record = spanCat.addNew();
        record->set(keys.spanY, val.getY());
        record->set(keys.spanX0, val.getX0());
        record->set(keys.spanX1, val.getX1());
    }
    handle.saveCatalog(spanCat);
}

//
// Explicit instantiations
#define INSTANTIATE_IMAGE_TYPE(T)                                                                      \
    template void SpanSet::setImage<T>(image::Image<T> & image, T val,                                 \
                                       geom::Box2I const& region = geom::Box2I(), bool doClip = false) \
            const;

#define INSTANTIATE_MASK_TYPE(T)                                                                           \
    template void SpanSet::setMask<T>(image::Mask<T> & target, T bitmask) const;                           \
    template void SpanSet::clearMask<T>(image::Mask<T> & target, T bitmask) const;                         \
    template std::shared_ptr<SpanSet> SpanSet::intersect<T>(image::Mask<T> const& other, T bitmask) const; \
    template std::shared_ptr<SpanSet> SpanSet::intersectNot<T>(image::Mask<T> const& other, T bitmask)     \
            const;                                                                                         \
    template std::shared_ptr<SpanSet> SpanSet::union_<T>(image::Mask<T> const& other, T bitmask) const;

INSTANTIATE_IMAGE_TYPE(std::uint16_t);
INSTANTIATE_IMAGE_TYPE(std::uint64_t);
INSTANTIATE_IMAGE_TYPE(int);
INSTANTIATE_IMAGE_TYPE(float);
INSTANTIATE_IMAGE_TYPE(double);

INSTANTIATE_MASK_TYPE(image::MaskPixel)
}
}
}  // Close lsst::afw::geom
