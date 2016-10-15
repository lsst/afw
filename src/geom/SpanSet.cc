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

#include "lsst/afw/geom/SpanSet.h"
#include <algorithm>
#include <iterator>

namespace {
    /* These classes are used in the erode operator to quickly calculate the
     * contents of the shrunken SpanSet
     */
struct PrimaryRun {
        int m, y, xmin, xmax;
};

bool comparePrimaryRun(PrimaryRun const & first, PrimaryRun const & second) {
        if (first.y != second.y) {
            return first.y < second.y;
        } else if (first.m != second.m) {
            return first.m < second.m;
        } else {
            return first.xmin < second.xmin;
        }
}

class ComparePrimaryRunY{
 public:
            bool operator()(PrimaryRun const& pr, int yval) {
                return pr.y < yval;
            }
            bool operator()(int yval, PrimaryRun const& pr) {
                return yval < pr.y;
            }
};

class ComparePrimaryRunM{
 public:
            bool operator()(PrimaryRun const& pr, int mval) {
                return pr.m < mval;
            }
            bool operator()(int mval, PrimaryRun const& pr) {
                return mval < pr.m;
            }
};
}  // namespace

namespace geom = lsst::afw::geom;

// Expose properties of the underlying vector constaining spans such that the
// SpanSet can be concidered a container
geom::SpanSet::const_iterator geom::SpanSet::begin() const {
    // Return the constant version as SpanSets should be immutable
    return spanVector.cbegin();
}

geom::SpanSet::const_iterator geom::SpanSet::end() const {
    // Return the constant version as SpanSets should be immutable
    return spanVector.cend();
}

geom::SpanSet::const_iterator geom::SpanSet::cbegin() const {
    return spanVector.cbegin();
}

geom::SpanSet::const_iterator geom::SpanSet::cend() const {
    return spanVector.cend();
}

geom::SpanSet::const_reference geom::SpanSet::front() const {
    // Return constant version as SpanSets should be immutable
    return const_cast<geom::Span &>(spanVector.front());
}

geom::SpanSet::const_reference geom::SpanSet::back() const {
    // Return constant version as SpanSets should be immutable
    return const_cast<geom::Span &>(spanVector.back());
}

geom::SpanSet::size_type geom::SpanSet::size() const {
    return spanVector.size();
}

bool geom::SpanSet::empty() const {
    return spanVector.empty();
}

// Default constructor, creates a null null spanset which may be useful for
// compairisons
geom::SpanSet::SpanSet():spanVector(), bBox(), area(0) {}

// Construct a spanset from a Box2I object
geom::SpanSet::SpanSet(geom::Box2I const & box):bBox(box), area(box.getArea()) {
    int beginY = box.getMinY();
    int endY = box.getMaxY();

    int beginX = box.getMinX();
    int endX = box.getMaxX();

    for (int i = beginY; i < (endY+1); ++i) {
        spanVector.push_back(Span(i, beginX, endX));
    }
}

// Construct a SpanSet from a std vector by copying
geom::SpanSet::SpanSet(const std::vector<Span> & vec, bool normalize): spanVector(vec) {
    if (normalize) {
        runNormalize();
    }
    initialize();
}

// Construct a SpanSet from a std vector by moving
geom::SpanSet::SpanSet(std::vector<Span> && vec, bool normalize): spanVector(std::move(vec)) {
    if (normalize) {
        runNormalize();
    }
    initialize();
}

void geom::SpanSet::runNormalize() {
    // Ensure the span set is sorted according to Span < operator
    std::sort(spanVector.begin(), spanVector.end());

    // Now that the span is sorted, overlapping spans should be combined
    // Start from 1 as the comparison will always be with the previous element
    for (auto iter = ++(spanVector.begin()); iter != spanVector.end(); iter++) {
        if (spansContiguous(*std::prev(iter, 1), *iter)) {
            // These spans overlap, create one new one, covering the whole expanse
            *(iter-1) = Span((*(iter-1)).getY(),
                             std::min((*(iter-1)).getMinX(), (*iter).getMinX()),
                             std::max((*(iter-1)).getMaxX(), (*iter).getMaxX()));
            // Erase the current Span, as it is now contained in the previous element
            iter = spanVector.erase(iter);
            // Move the iterator back one to make sure the element gets run in the loop
            iter = iter - 1;
        }
    }
}


void geom::SpanSet::initialize() {
    /* This function exists to handle common functionality for most of the constructors. It  calculates the
     * bounding box for the spanset, and the area covered by the SpanSet
     */

    /* Because the array is sorted, the minimum and maximum values for Y will
       be in the first and last elements, only need to find the min and max X
       values */
    int minX = spanVector[0].getMinX();
    int maxX = spanVector[0].getMaxX();
    area = 0;

    for (const auto & span : spanVector) {
        if (span.getMinX() < minX) {
            minX = span.getMinX();
        }
        if (span.getMaxX() > maxX) {
            maxX = span.getMaxX();
        }
        // Plus one, because end point is inclusive
        area += span.getMaxX() - span.getMinX() + 1;
    }
    bBox = geom::Box2I(Point2I(minX, spanVector.front().getY()), Point2I(maxX, spanVector.back().getY()));
}

// Getter for the area property
std::size_t geom::SpanSet::getArea() const {
    return area;
}

// Getter for the bounding box of the SpanSet
geom::Box2I geom::SpanSet::getBBox() const {
    return bBox;
}


bool geom::SpanSet::isContiguous() const {
    /* A contiguous region means that each Span value should have some overlap in the x range with the
     * previous span, if the current and previous span have different y values
     */
    for (auto iter = spanVector.begin() + 1 ; iter != spanVector.end() ; iter++) {
        // This means they are on the same row, and are not contiguous in x
        if ((*(iter -1)).getY() == (*iter).getY() && !spansContiguous(*(iter - 1), *iter, false)) {
            // If two spans are not contiguous on the same row, they must bothe be contiguous with either
            // the row before or after
            bool notBefore(false);
            bool notAfter(false);
            // ensure the first span is not the begining, if so cant be joined before
            if ((iter - 1) != spanVector.begin()) {
                if ( !spansContiguous(*(iter - 2), *(iter - 1), false) ||
                     !spansContiguous(*(iter - 2), *iter, false)) {
                         // one of the spans was not contiguous, set notBeforeto true;
                         notBefore = true;
                }
            } else {
                notBefore = true;
            }
            if ((iter +1) != spanVector.end()) {
                if ( !spansContiguous(*(iter -1 ), *(iter + 1), false) ||
                     !spansContiguous(*iter, *(iter + 1), false)) {
                         notAfter = true;
                }
            } else {
                notAfter = true;
            }
            // either notBefore or notAfter must be false (meaning the join is either before or after)
            if (notBefore && notAfter) {
                // it is not before and it is not after, SpanSet can't be contiguous so return false
                return false;
            }
        } else if (!spansOverlap(*(iter - 1), *iter, false)) {
            return false;
        }
    }
    return true;
}


geom::SpanSet geom::SpanSet::shiftedBy(int x, int y) const {
    // Funtion to create a new spanset which is a copy of this, shifted by x and y
    return makeShift(x, y);
}

geom::SpanSet geom::SpanSet::shiftedBy(geom::Extent2I const & offset) const {
    // Funtion to create a new spanset which is a copy of this, shifted by an extent object
    return makeShift(offset.getX(), offset.getY());
}

geom::SpanSet geom::SpanSet::makeShift(int x, int y) const {
    // Implementation method common to all overloads of the shiftedBy method
    std::vector<Span> tempVec;
    tempVec.reserve(spanVector.size());
    for (auto const & spn : spanVector) {
        tempVec.push_back(Span(spn.getY() + y, spn.getMinX() + x, spn.getMaxX() + x));
    }
    return geom::SpanSet(std::move(tempVec), false);
}

geom::SpanSet geom::SpanSet::clippedTo(geom::Box2I const & box) const {
    /* Return a copy of the current SpanSet but only with values which are contained within
     * the supplied box
     */
    std::vector<Span> tempVec;
    for (auto const & spn : spanVector) {
        if (spn.getY() >= box.getMinY() && spn.getY() <= box.getMaxY() &&
            spansOverlap(spn, geom::Span(spn.getY(), box.getMinX(), box.getMaxX()))) {
            tempVec.push_back(Span(spn.getY(), std::max(box.getMinX(), spn.getMinX()),
                                   std::min(box.getMaxX(), spn.getMaxX())));
        }
    }
    return geom::SpanSet(std::move(tempVec), false);
}

bool geom::SpanSet::overlaps(SpanSet const & other) const {
    // Function to check if two SpanSets overlap
    for (auto const & otherSpan : other) {
        for (auto const & spn : spanVector) {
            if (spansOverlap(otherSpan, spn)) {
                return true;
            }
        }
    }
    return false;
}

bool geom::SpanSet::contains(geom::SpanSet const & other) const {
    // Function to check if a SpanSet is entirely contained within this
    for (auto const & otherSpn : other) {
        int counter = 0;
        for (auto const & spn : spanVector) {
            // Check that the end points of the span from other are contained in the
            // span from this
            if (spn.contains(geom::Point2I(otherSpn.getMinX(), otherSpn.getY())) &&
                spn.contains(geom::Point2I(otherSpn.getMaxX(), otherSpn.getY())))
                ++counter;
        }
        // if counter is equal to zero, then the current span from other is not
        // contained in any span in this, and the function should return false
        // short circuting any other spans from other
        if (counter == 0) {
            return false;
        }
    }
    return true;
}

bool geom::SpanSet::contains(geom::Point2I const & point) const {
    // Check to see if a given point is found within any spans in this
    for (auto & spn : spanVector) {
        if (spn.contains(point)) {
            return true;
        }
    }
    return false;
}

geom::Point2D geom::SpanSet::computeCentroid() const {
    // Find the centroid of the SpanSet
    unsigned int n = 0;
    double xc = 0, yc = 0;
    for (auto const & spn : spanVector) {
        int const y = spn.getY();
        int const x0 = spn.getMinX();
        int const x1 = spn.getMaxX();
        int const npix = x1 - x0 + 1;

        n += npix;
        xc += npix*0.5*(x1 + x0);
        yc += npix*y;
    }
    assert(n == area);

    return geom::Point2D(xc/area, yc/area);
}

geom::ellipses::Quadrupole geom::SpanSet::computeShape() const {
    // Compute the shape of the SpanSet
    geom::Point2D cen = computeCentroid();
    double const xc = cen.getX();
    double const yc = cen.getY();

    double sumxx = 0, sumxy = 0, sumyy = 0;
    for (auto const & spn : spanVector) {
        int const y = spn.getY();
        int const x0 = spn.getX0();
        int const x1 = spn.getX1();
        int const npix = x1 - x0 + 1;

        for (int x = x0; x <= x1; ++x) {
            sumxx += (x - xc)*(x - xc);
        }
        sumxy += npix*(0.5*(x1 + x0) - xc)*(y - yc);
        sumyy += npix*(y - yc)*(y - yc);
    }

    return geom::ellipses::Quadrupole(sumxx/area, sumyy/area, sumxy/area);
}

geom::SpanSet geom::SpanSet::dilate(int r, Stencil s) const {
    // Return a dilated SpanSet made with the given stencil, by creating a SpanSet
    // from the stencil and forwarding to the appropriate overloaded method
    geom::SpanSet stencilToSpanSet = spanSetFromShape(r, s);
    return dilate(stencilToSpanSet);
}

geom::SpanSet geom::SpanSet::dilate(SpanSet const & other) const {
    // Return a dilated Spanset by the given SpanSet
    std::vector<Span> tempVec;

    for (auto const & spn : spanVector) {
        for (auto const & otherSpn : other) {
            int const xmin = spn.getMinX() + otherSpn.getMinX();
            int const xmax = spn.getMaxX() + otherSpn.getMaxX();
            int const yval = spn.getY() + otherSpn.getY();
            tempVec.push_back(geom::Span(yval, xmin, xmax));
        }
    }
    // Allow constructor to handle merging adjacent and overlapping spans
    return SpanSet(std::move(tempVec));
}

geom::SpanSet geom::SpanSet::erode(int r, Stencil s) const {
    // Return an eroded SpanSet made with the given stencil, by creating a SpanSet
    // from the stencil and forwarding to the appropriate overloaded method
    geom::SpanSet stencilToSpanSet = spanSetFromShape(r, s);
    return erode(stencilToSpanSet);
}

geom::SpanSet geom::SpanSet::erode(SpanSet const & other) const {
    // Return a SpanSet erroded by the given SpanSet
    std::vector<Span> tempVec;

    // Calculate all possible primary runs.
    std::vector<PrimaryRun> primaryRuns;
    for (auto const & spn : spanVector) {
        int m = 0;
        for (auto const & otherSpn : other) {
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

        /* Discard runs for any balue of y for wich we find fewer groups than M,
         * the total Y range of the structuring element. THis is step 3.1 of the
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
                 // Consolidate all primary runs at this m so that they dont overlap.
                 std::list<PrimaryRun> candidateRuns;
                 int start_x = mRange.first->xmin;
                 int end_x = mRange.first->xmax;
                 for (auto run = mRange.first+1; run != mRange.second; ++run) {
                     if (run->xmin > end_x) {
                         // Start of a new run
                         candidateRuns.push_back(PrimaryRun{m, y, start_x, end_x});
                         start_x = run->xmin;
                         end_x = run->xmax;
                     } else {
                         // Continuation of an existing run
                         end_x = run->xmax;
                     }
                 }
                 candidateRuns.push_back(PrimaryRun{m, y, start_x, end_x});

                 // Otherwise, calculate the intersection of candidate runs at
                 // this m with good runs from all previous m.
                 if (m == 0) {
                     // For m = 0 we have nothing to compare to; all runs are accepted
                     std::swap(goodRuns, candidateRuns);
                 } else {
                     std::list<PrimaryRun> newlist;
                     for (auto & good : goodRuns) {
                         for (auto & cand : candidateRuns) {
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
         for (auto & run : goodRuns) {
             tempVec.push_back(geom::Span(run.y, run.xmin, run.xmax));
         }
    }
    return geom::SpanSet(std::move(tempVec));
}

bool geom::SpanSet::operator==(SpanSet const & other) const {
    // Check the equivelance of this SpanSet with another
    return spanVector == other.spanVector;
}

bool geom::SpanSet::operator!=(SpanSet const & other) const {
    // Check the equivelance of this SpanSet with another
    return spanVector != other.spanVector;
}

geom::SpanSet geom::SpanSet::spanSetFromShape(int r, Stencil s) {
    // Create a SpanSet from a given Stencil
    std::vector<Span> tempVec;
    tempVec.reserve(2*r + 1);
    switch (s) {
        case Stencil::CIRCLE:
            for (auto dy = -r; dy <= r; ++dy) {
                int dx = static_cast<int>(sqrt(r*r - dy*dy));
                tempVec.push_back(geom::Span(dy, -dx, dx));
            }
            break;
        case Stencil::MANHATTAN:
            for (auto dy = -r; dy <= r; ++dy) {
                int dx = r - abs(dy);
                tempVec.push_back(geom::Span(dy, -dx, dx));
            }
            break;
        case Stencil::BOX:
            for (auto dy = -r; dy <= r; ++dy) {
                int dx = r;
                tempVec.push_back(geom::Span(dy, -dx, dx));
            }
            break;
    }
    return geom::SpanSet(std::move(tempVec), false);
}

geom::SpanSet geom::SpanSet::intersect(geom::SpanSet const & other) const {
    // Check if the bounding boxes overlap, if not return null spanset
    if (!bBox.overlaps(other.getBBox())) {
        return geom::SpanSet();
    }
    std::vector<Span> tempVec;
    for (auto const & spn : spanVector) {
        for (auto const & otherSpn : other) {
            if (spansOverlap(spn, otherSpn)) {
                auto newMin = std::max(spn.getMinX(), otherSpn.getMinX());
                auto newMax = std::min(spn.getMaxX(), otherSpn.getMaxX());
                auto newSpan = geom::Span(spn.getY(), newMin, newMax);
                tempVec.push_back(newSpan);
            }
        }
    }
    return geom::SpanSet(std::move(tempVec), false);
}

geom::SpanSet geom::SpanSet::intersectNot(geom::SpanSet const & other) const {
    // Check if the bounding boxes overlap, if not simply return a copy of this
    if (!getBBox().overlaps(other.getBBox())) {
        return geom::SpanSet(*this);
    }
    /* This function must find all the areas in this and not in other. These spansets
     * may be overlapping with this less than other a1|    b1|   a2|    b2|,
     * with this greater than other b1|    a1|     b2|    a2|,
     * with this containing other a1|    b1|     b2|     a2|,
     * or with other containing this  b1|    a1|    a2|   b2|
     */
    std::vector<Span> tempVec;
    int added;
    for (auto const & spn : spanVector) {
        added = 0;
        for (auto const & otherSpn : other) {
            if (spansOverlap(spn, otherSpn)) {
                added = 1;
                /* To handle one span containing the other, the spans will be added
                 * peice wise, and let the SpanSet contructor normalize spans which
                 * end up contiguous. In the case where this is contained in other,
                 * these statements will all be false, and nothing will be added,
                 * which is the expected behaivor.
                 */
                if (spn.getMinX() < otherSpn.getMinX()) {
                    tempVec.push_back(geom::Span(spn.getY(), spn.getMinX(), otherSpn.getMinX()-1));
                }
                if (spn.getMaxX() > otherSpn.getMaxX()) {
                    tempVec.push_back(geom::Span(spn.getY(), otherSpn.getMaxX()+1, spn.getMaxX()));
                }
            }
        }
        /* If added is still zero, that means it did not verlap any of the spans in other
         * and should be included in the new span
         */
        if (added == 0) {
            tempVec.push_back(geom::Span(spn));
        }
    }
    return geom::SpanSet(std::move(tempVec));
}

geom::SpanSet geom::SpanSet::union_(geom::SpanSet const & other) const {
    /* Simply include Spans from both spansets in a new vector and let the SpanSet
     * constructor normalize any of the spansets which may be contiguous
     */
    int combineSize = size() + other.size();
    std::vector<Span> tempVec;
    tempVec.reserve(combineSize);
    // Copy this
    tempVec.insert(tempVec.end(), spanVector.begin(), spanVector.end());
    // Copy other
    tempVec.insert(tempVec.end(), other.begin(), other.end());
    return geom::SpanSet(std::move(tempVec));
}

geom::SpanSet geom::SpanSet::transformedBy(geom::LinearTransform const & t) const {
    // Transform points in SpanSet by LinearTransform
    return transformedBy(geom::AffineTransform(t));
}

geom::SpanSet geom::SpanSet::transformedBy(geom::AffineTransform const & t) const {
    // Transform points in SpanSet by AffineTransform
    return transformedBy(geom::AffineXYTransform(t));
}

geom::SpanSet geom::SpanSet::transformedBy(geom::XYTransform const & t) const {
    // Transform points in SpanSet by XYTransform
    // Transform the origional bounding box
    geom::Box2D newBBoxD;
    newBBoxD.include(t.forwardTransform(geom::Point2D(bBox.getMinX(), bBox.getMinY())));
    newBBoxD.include(t.forwardTransform(geom::Point2D(bBox.getMinX(), bBox.getMaxY())));
    newBBoxD.include(t.forwardTransform(geom::Point2D(bBox.getMaxX(), bBox.getMinY())));
    newBBoxD.include(t.forwardTransform(geom::Point2D(bBox.getMaxX(), bBox.getMaxY())));

    geom::Box2I newBBoxI(newBBoxD);

    std::vector<Span> tempVec;
    for (int y = newBBoxI.getBeginY(); y < newBBoxI.getEndY(); ++y) {
        bool inSpan = false;    // Are we in a span?
        int start = -1;         // Start of span

        for (int x = newBBoxI.getBeginX(); x < newBBoxI.getEndY(); ++x) {
            geom::Point2D p = t.reverseTransform(geom::Point2D(x, y));
            int const xSource = std::floor(0.5 + p.getX());
            int const ySource = std::floor(0.5 + p.getY());

            if (contains(geom::Point2I(xSource, ySource))) {
                if (!inSpan) {
                    inSpan = true;
                    start = x;
                }
            } else if (inSpan) {
                inSpan = false;
                tempVec.push_back(geom::Span(y, start, x-1));
            }
        }
        if (inSpan) {
            tempVec.push_back(geom::Span(y, start, newBBoxI.getMaxX()));
        }
    }
    return geom::SpanSet(std::move(tempVec));
}
