// -*- lsst-c++ -*-
#ifndef LSST_AFW_REGION_ListRegion_INCLUDED
#define LSST_AFW_REGION_ListRegion_INCLUDED

#include <vector>

#include "lsst/afw/pixels/Region.h"

namespace lsst { namespace afw { namespace region {

/**
 *  @brief A helper class defining patterns for dilating a pixel region.
 */
class StructuringElement {
public:

    enum Enum { CIRCLE, DIAMOND, BOX };

    StructuringElement(Enum pattern, int radius);

    int getRadius() const { return _radius; }

    int getOffset(int dy) const { return _offsets[_radius - std::abs(dy)]; }

private:
    int _radius;
    std::vector<int> _offsets;
};

class SpanRegion;

template <>
struct RegionTraits< SpanRegion > {
    typedef std::vector<Span>::const_iterator Iterator;
};

/**
 *  @brief The default region implementation.
 */
class SpanRegion : public Region< SpanRegion > {
public:

    typedef std::vector<Span>::const_iterator Iterator;

    Iterator begin() const { return _data->_spans.begin(); }
    Iterator end() const { return _data->_spans.end(); }

    int size() const { return _data->_spans.size(); }
    bool empty() const { return _data->_spans.empty(); }

    int getArea() const { return _data->_area; }
    geom::Box2I getBBox() const { return _data->_bbox; }

    /// @brief Move the footprint by the given offset.
    void shift(geom::Extent2I const & offset);

    /// @brief Remove all pixels from the region that are not contained by the given box.
    void clip(geom::Box2I const & box);

    /// @brief Dilate (binary convolve) the region with the given structuring element.
    void dilate(StructuringElement const & s);

    /// @brief Dilate with a circular structuring element of the given radius.
    void grow(int buffer) { dilate(StructuringElement(StructuringElement::CIRCLE, buffer)); }

    /// @brief Transform by the given AffineTransform (see Region::transform).
    void transform(AffineTransform const & t);

    template <typename Other>
    void intersect(Region<Other> const & other);

    template <typename Other>
    void intersect(RegionInverse<Other> const & other);

    template <typename Predicate>
    void intersect(Predicate predicate, geom::Box2I const & bbox);

    template <typename Other>
    void union_(Region<Other> const & other);

    template <typename Predicate>
    void union_(Predicate predicate, geom::Box2I const & bbox);

    template <typename Other>
    SpanRegion & operator&=(Region<Other> const & other) {
        intersect(other);
        return *this;
    }

    template <typename Other>
    SpanRegion & operator&=(RegionInverse<Other> const & other) {
        intersect(other);
        return *this;
    }

    template <typename Other>
    SpanRegion & operator|=(Region<Other> const & other) {
        union_(other);
        return *this;
    }    

    /// @brief Construct an empty region.
    SpanRegion() : _data(new Data()) {}

    /// @brief Construct from an iterator over spans.
    template <typename IteratorT>
    SpanRegion(IteratorT first, IteratorT last, bool normalize=true);

    /// @brief Construct from a vector of spans that will be destroyed in the processes.
    explicit SpanRegion(std::vector<Span> & spans, bool normalize=true);

    /// @brief Construct from another pixel region.
    template <typename Other>
    explicit SpanRegion(Region<Other> const & other);

    /// @brief Construct from a predicate region.
    template <typename Other>
    explicit SpanRegion(PredicateRegion<Other> const & other);

private:

    struct Data {
        int _area;
        geom::Box2I _bbox;
        std::vector<Span> _spans;

        Data() : _area(0) {}
    };

    void edit();

    boost::shared_ptr<Data> _data;
};

}}} // namespace lsst::afw::region

#endif // !LSST_AFW_REGION_SpanRegion_INCLUDED
