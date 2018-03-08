#include <cmath>
#include <algorithm>

#include "boost/geometry/geometry.hpp"
#include <memory>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/polygon/Polygon.h"

#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/aggregates.h"

typedef lsst::afw::geom::polygon::Polygon::Point LsstPoint;
typedef lsst::afw::geom::polygon::Polygon::Box LsstBox;
typedef std::vector<LsstPoint> LsstRing;
typedef boost::geometry::model::polygon<LsstPoint> BoostPolygon;
typedef boost::geometry::model::box<LsstPoint> BoostBox;
typedef boost::geometry::model::linestring<LsstPoint> BoostLineString;

namespace boost {
namespace geometry {
namespace traits {

// Setting up LsstPoint
template <>
struct tag<LsstPoint> {
    typedef point_tag type;
};
template <>
struct coordinate_type<LsstPoint> {
    typedef LsstPoint::Element type;
};
template <>
struct coordinate_system<LsstPoint> {
    typedef cs::cartesian type;
};
template <>
struct dimension<LsstPoint> : boost::mpl::int_<2> {};
template <std::size_t dim>
struct access<LsstPoint, dim> {
    static double get(LsstPoint const& p) { return p[dim]; }
    static void set(LsstPoint& p, LsstPoint::Element const& value) { p[dim] = value; }
};

// Setting up LsstBox
//
// No setters, because it's inefficient (can't set individual elements of Box2D directly).
// For box outputs from boost::geometry we'll use BoostBox and then convert.
template <>
struct tag<LsstBox> {
    typedef box_tag type;
};
template <>
struct point_type<LsstBox> {
    typedef LsstPoint type;
};
template <>
struct indexed_access<LsstBox, 0, 0> {
    static double get(LsstBox const& box) { return box.getMinX(); }
};
template <>
struct indexed_access<LsstBox, 1, 0> {
    static double get(LsstBox const& box) { return box.getMaxX(); }
};
template <>
struct indexed_access<LsstBox, 0, 1> {
    static double get(LsstBox const& box) { return box.getMinY(); }
};
template <>
struct indexed_access<LsstBox, 1, 1> {
    static double get(LsstBox const& box) { return box.getMaxY(); }
};

// Setting up LsstRing
template <>
struct tag<LsstRing> {
    typedef ring_tag type;
};
// template<> struct range_value<LsstRing> { typedef LsstPoint type; };
}  // namespace traits
}  // namespace geometry
}  // namespace boost

namespace {

/// @internal Convert BoostBox to LsstBox
LsstBox boostBoxToLsst(BoostBox const& box) { return LsstBox(box.min_corner(), box.max_corner()); }

/// @internal Convert box to corners
std::vector<LsstPoint> boxToCorners(LsstBox const& box) {
    std::vector<LsstPoint> corners;
    corners.reserve(4);
    corners.push_back(box.getMin());
    corners.push_back(LsstPoint(box.getMaxX(), box.getMinY()));
    corners.push_back(box.getMax());
    corners.push_back(LsstPoint(box.getMinX(), box.getMaxY()));
    return corners;
}

/**
 * @internal Sub-sample a line
 *
 * Add `num` points to `vector` between `first` and `second`
 */
void addSubSampledEdge(std::vector<LsstPoint>& vertices,  // Vector of points to which to add
                       LsstPoint const& first,            // First vertex defining edge
                       LsstPoint const& second,           // Second vertex defining edge
                       size_t const num                   // Number of parts to divide edge into
                       ) {
    lsst::afw::geom::Extent2D const delta = (second - first) / num;
    vertices.push_back(first);
    for (size_t i = 1; i < num; ++i) {
        vertices.push_back(first + delta * i);
    }
}

/// @internal Calculate area of overlap between polygon and pixel
double pixelOverlap(BoostPolygon const& poly, int const x, int const y) {
    std::vector<BoostPolygon> overlap;  // Overlap between pixel and polygon
    LsstBox const pixel(lsst::afw::geom::Point2D(x - 0.5, y - 0.5),
                        lsst::afw::geom::Point2D(x + 0.5, y + 0.5));
    boost::geometry::intersection(poly, pixel, overlap);
    double area = 0.0;
    for (std::vector<BoostPolygon>::const_iterator i = overlap.begin(); i != overlap.end(); ++i) {
        double const polyArea = boost::geometry::area(*i);
        area += std::min(polyArea, 1.0);  // remove any rounding error
    }
    return area;
}

/// @internal Set each pixel in a row to the amount of overlap with polygon
void pixelRowOverlap(std::shared_ptr<lsst::afw::image::Image<float>> const image, BoostPolygon const& poly,
                     int const xStart, int const xStop, int const y) {
    int x = xStart;
    for (lsst::afw::image::Image<float>::x_iterator i = image->x_at(x - image->getX0(), y - image->getY0());
         x <= xStop; ++i, ++x) {
        *i = pixelOverlap(poly, x, y);
    }
}

}  // anonymous namespace

namespace lsst {
namespace afw {
namespace geom {
namespace polygon {

/// @internal Stream vertices
std::ostream& operator<<(std::ostream& os, std::vector<LsstPoint> const& vertices) {
    os << "[";
    size_t num = vertices.size();
    for (size_t i = 0; i < num - 1; ++i) {
        os << vertices[i] << ",";
    }
    os << vertices[vertices.size() - 1] << "]";
    return os;
}

/// @internal Stream BoostPolygon
std::ostream& operator<<(std::ostream& os, BoostPolygon const& poly) {
    return os << "BoostPolygon(" << poly.outer() << ")";
}

std::ostream& operator<<(std::ostream& os, Polygon const& poly) {
    os << "Polygon(" << poly.getVertices() << ")";
    return os;
}

struct Polygon::Impl {
    Impl() : poly() {}
    explicit Impl(Polygon::Box const& box) : poly() {
        boost::geometry::assign(poly, box);
        // Assignment from a box is correctly handled by BoostPolygon, so doesn't need a "check()"
    }
    explicit Impl(std::vector<LsstPoint> const& vertices) : poly() {
        boost::geometry::assign(poly, vertices);
        check();  // because the vertices might not have the correct orientation (CW vs CCW) or be open
    }
    explicit Impl(BoostPolygon const& _poly) : poly(_poly) {}

    void check() { boost::geometry::correct(poly); }

    /// @internal Convert collection of Boost polygons to our own
    static std::vector<std::shared_ptr<Polygon>> convertBoostPolygons(
            std::vector<BoostPolygon> const& boostPolygons);

    template <class PolyT>
    bool overlaps(PolyT const& other) const {
        return !boost::geometry::disjoint(poly, other);
    }

    template <class PolyT>
    std::shared_ptr<Polygon> intersectionSingle(PolyT const& other) const;

    template <class PolyT>
    std::vector<std::shared_ptr<Polygon>> intersection(PolyT const& other) const;

    template <class PolyT>
    std::shared_ptr<Polygon> unionSingle(PolyT const& other) const;

    template <class PolyT>
    std::vector<std::shared_ptr<Polygon>> union_(PolyT const& other) const;

    template <class PolyT>
    std::vector<std::shared_ptr<Polygon>> symDifference(PolyT const& other) const;

    BoostPolygon poly;
};

std::vector<std::shared_ptr<Polygon>> Polygon::Impl::convertBoostPolygons(
        std::vector<BoostPolygon> const& boostPolygons) {
    std::vector<std::shared_ptr<Polygon>> lsstPolygons;
    lsstPolygons.reserve(boostPolygons.size());
    for (std::vector<BoostPolygon>::const_iterator i = boostPolygons.begin(); i != boostPolygons.end(); ++i) {
        std::shared_ptr<Polygon> tmp(new Polygon(std::make_shared<Polygon::Impl>(*i)));
        lsstPolygons.push_back(tmp);
    }
    return lsstPolygons;
}

template <class PolyT>
std::shared_ptr<Polygon> Polygon::Impl::intersectionSingle(PolyT const& other) const {
    std::vector<BoostPolygon> result;
    boost::geometry::intersection(poly, other, result);
    if (result.size() == 0) {
        throw LSST_EXCEPT(SinglePolygonException, "Polygons have no intersection");
    }
    if (result.size() > 1) {
        throw LSST_EXCEPT(
                SinglePolygonException,
                (boost::format("Multiple polygons (%d) created by intersection()") % result.size()).str());
    }
    return std::shared_ptr<Polygon>(new Polygon(std::make_shared<Impl>(result[0])));
}

template <class PolyT>
std::vector<std::shared_ptr<Polygon>> Polygon::Impl::intersection(PolyT const& other) const {
    std::vector<BoostPolygon> boostResult;
    boost::geometry::intersection(poly, other, boostResult);
    return convertBoostPolygons(boostResult);
}

template <class PolyT>
std::shared_ptr<Polygon> Polygon::Impl::unionSingle(PolyT const& other) const {
    std::vector<BoostPolygon> result;
    boost::geometry::union_(poly, other, result);
    if (result.size() != 1) {
        throw LSST_EXCEPT(
                SinglePolygonException,
                (boost::format("Multiple polygons (%d) created by union_()") % result.size()).str());
    }
    return std::shared_ptr<Polygon>(new Polygon(std::make_shared<Impl>(result[0])));
}

template <class PolyT>
std::vector<std::shared_ptr<Polygon>> Polygon::Impl::union_(PolyT const& other) const {
    std::vector<BoostPolygon> boostResult;
    boost::geometry::union_(poly, other, boostResult);
    return convertBoostPolygons(boostResult);
}

template <class PolyT>
std::vector<std::shared_ptr<Polygon>> Polygon::Impl::symDifference(PolyT const& other) const {
    std::vector<BoostPolygon> boostResult;
    boost::geometry::sym_difference(poly, other, boostResult);
    return convertBoostPolygons(boostResult);
}

Polygon::Polygon(Polygon const&) = default;
Polygon::Polygon(Polygon&&) = default;
Polygon& Polygon::operator=(Polygon const&) = default;
Polygon& Polygon::operator=(Polygon&&) = default;

Polygon::~Polygon() = default;

Polygon::Polygon(Polygon::Box const& box) : _impl(new Polygon::Impl(box)) {}

Polygon::Polygon(std::vector<LsstPoint> const& vertices) : _impl(new Polygon::Impl(vertices)) {}

Polygon::Polygon(Polygon::Box const& box, afw::geom::TransformPoint2ToPoint2 const & transform)
        : _impl(new Polygon::Impl()) {
    auto corners = transform.applyForward(boxToCorners(box));
    boost::geometry::assign(_impl->poly, corners);
    _impl->check();
}

Polygon::Polygon(Polygon::Box const& box, std::shared_ptr<afw::geom::XYTransform const> const& transform)
        : _impl(new Polygon::Impl()) {
    std::vector<LsstPoint> corners = boxToCorners(box);
    for (std::vector<LsstPoint>::iterator p = corners.begin(); p != corners.end(); ++p) {
        *p = transform->forwardTransform(*p);
    }
    boost::geometry::assign(_impl->poly, corners);
    _impl->check();
}

Polygon::Polygon(Polygon::Box const& box, afw::geom::AffineTransform const& transform)
        : _impl(new Polygon::Impl()) {
    std::vector<LsstPoint> corners = boxToCorners(box);
    for (std::vector<LsstPoint>::iterator p = corners.begin(); p != corners.end(); ++p) {
        *p = transform(*p);
    }
    boost::geometry::assign(_impl->poly, corners);
    _impl->check();
}

size_t Polygon::getNumEdges() const {
    // boost::geometry::models::polygon uses a "closed" polygon: the start/end point is included twice
    return boost::geometry::num_points(_impl->poly) - 1;
}

Polygon::Box Polygon::getBBox() const {
    return boostBoxToLsst(boost::geometry::return_envelope<BoostBox>(_impl->poly));
}

LsstPoint Polygon::calculateCenter() const {
    return boost::geometry::return_centroid<LsstPoint>(_impl->poly);
}

double Polygon::calculateArea() const { return boost::geometry::area(_impl->poly); }

double Polygon::calculatePerimeter() const { return boost::geometry::perimeter(_impl->poly); }

std::vector<std::pair<LsstPoint, LsstPoint>> Polygon::getEdges() const {
    std::vector<LsstPoint> const vertices = getVertices();
    std::vector<std::pair<LsstPoint, LsstPoint>> edges;
    edges.reserve(getNumEdges());
    for (std::vector<LsstPoint>::const_iterator i = vertices.begin(), j = vertices.begin() + 1;
         j != vertices.end(); ++i, ++j) {
        edges.push_back(std::make_pair(*i, *j));
    }
    return edges;
}

std::vector<LsstPoint> Polygon::getVertices() const { return _impl->poly.outer(); }

std::vector<LsstPoint>::const_iterator Polygon::begin() const { return _impl->poly.outer().begin(); }

std::vector<LsstPoint>::const_iterator Polygon::end() const {
    return _impl->poly.outer().end() - 1;  // Note removal of final "closed" point
}

bool Polygon::operator==(Polygon const& other) const {
    return boost::geometry::equals(_impl->poly, other._impl->poly);
}

bool Polygon::contains(LsstPoint const& point) const { return boost::geometry::within(point, _impl->poly); }

bool Polygon::overlaps(Polygon const& other) const { return _impl->overlaps(other._impl->poly); }

bool Polygon::overlaps(Box const& box) const { return _impl->overlaps(box); }

std::shared_ptr<Polygon> Polygon::intersectionSingle(Polygon const& other) const {
    return _impl->intersectionSingle(other._impl->poly);
}

std::shared_ptr<Polygon> Polygon::intersectionSingle(Box const& box) const {
    return _impl->intersectionSingle(box);
}

std::vector<std::shared_ptr<Polygon>> Polygon::intersection(Polygon const& other) const {
    return _impl->intersection(other._impl->poly);
}

std::vector<std::shared_ptr<Polygon>> Polygon::intersection(Box const& box) const {
    return _impl->intersection(box);
}

std::shared_ptr<Polygon> Polygon::unionSingle(Polygon const& other) const {
    return _impl->unionSingle(other._impl->poly);
}

std::shared_ptr<Polygon> Polygon::unionSingle(Box const& box) const { return _impl->unionSingle(box); }

std::vector<std::shared_ptr<Polygon>> Polygon::union_(Polygon const& other) const {
    return _impl->union_(other._impl->poly);
}

std::vector<std::shared_ptr<Polygon>> Polygon::union_(Box const& box) const { return _impl->union_(box); }

std::vector<std::shared_ptr<Polygon>> Polygon::symDifference(Polygon const& other) const {
    return _impl->symDifference(other._impl->poly);
}

std::vector<std::shared_ptr<Polygon>> Polygon::symDifference(Box const& box) const {
    return _impl->symDifference(box);
}

std::shared_ptr<Polygon> Polygon::simplify(double const distance) const {
    BoostPolygon result;
    boost::geometry::simplify(_impl->poly, result, distance);
    return std::shared_ptr<Polygon>(new Polygon(std::make_shared<Impl>(result)));
}

std::shared_ptr<Polygon> Polygon::convexHull() const {
    BoostPolygon hull;
    boost::geometry::convex_hull(_impl->poly, hull);
    return std::shared_ptr<Polygon>(new Polygon(std::make_shared<Impl>(hull)));
}

std::shared_ptr<Polygon> Polygon::transform(TransformPoint2ToPoint2 const& transform) const {
    auto newVertices = transform.applyForward(getVertices());
    return std::shared_ptr<Polygon>(new Polygon(std::make_shared<Impl>(newVertices)));
}


std::shared_ptr<Polygon> Polygon::transform(std::shared_ptr<XYTransform const> const& transform) const {
    std::vector<LsstPoint> vertices;  // New vertices
    vertices.reserve(getNumEdges());
    for (std::vector<LsstPoint>::const_iterator i = _impl->poly.outer().begin();
         i != _impl->poly.outer().end(); ++i) {
        vertices.push_back(transform->forwardTransform(*i));
    }
    return std::shared_ptr<Polygon>(new Polygon(std::make_shared<Impl>(vertices)));
}

std::shared_ptr<Polygon> Polygon::transform(AffineTransform const& transform) const {
    std::vector<LsstPoint> vertices;  // New vertices
    vertices.reserve(getNumEdges());
    for (std::vector<LsstPoint>::const_iterator i = _impl->poly.outer().begin();
         i != _impl->poly.outer().end(); ++i) {
        vertices.push_back(transform(*i));
    }
    return std::shared_ptr<Polygon>(new Polygon(std::make_shared<Impl>(vertices)));
}

std::shared_ptr<Polygon> Polygon::subSample(size_t num) const {
    std::vector<LsstPoint> vertices;  // New vertices
    vertices.reserve(getNumEdges() * num);
    std::vector<std::pair<Point, Point>> edges = getEdges();
    for (std::vector<std::pair<Point, Point>>::const_iterator i = edges.begin(); i != edges.end(); ++i) {
        addSubSampledEdge(vertices, i->first, i->second, num);
    }
    return std::shared_ptr<Polygon>(new Polygon(std::make_shared<Impl>(vertices)));
}

std::shared_ptr<Polygon> Polygon::subSample(double maxLength) const {
    std::vector<LsstPoint> vertices;  // New vertices
    vertices.reserve(getNumEdges() + static_cast<size_t>(::ceil(calculatePerimeter() / maxLength)));
    std::vector<std::pair<Point, Point>> edges = getEdges();
    for (std::vector<std::pair<Point, Point>>::const_iterator i = edges.begin(); i != edges.end(); ++i) {
        Point const &p1 = i->first, p2 = i->second;
        double const dist = ::sqrt(p1.distanceSquared(p2));
        addSubSampledEdge(vertices, p1, p2, static_cast<size_t>(::ceil(dist / maxLength)));
    }
    return std::shared_ptr<Polygon>(new Polygon(std::make_shared<Impl>(vertices)));
}

std::shared_ptr<afw::image::Image<float>> Polygon::createImage(afw::geom::Box2I const& bbox) const {
    typedef afw::image::Image<float> Image;
    std::shared_ptr<Image> image = std::make_shared<Image>(bbox);
    image->setXY0(bbox.getMin());
    *image = 0.0;
    afw::geom::Box2D bounds = getBBox();  // Polygon bounds
    int xMin = std::max(static_cast<int>(bounds.getMinX()), bbox.getMinX());
    int xMax = std::min(static_cast<int>(::ceil(bounds.getMaxX())), bbox.getMaxX());
    int yMin = std::max(static_cast<int>(bounds.getMinY()), bbox.getMinY());
    int yMax = std::min(static_cast<int>(::ceil(bounds.getMaxY())), bbox.getMaxY());
    for (int y = yMin; y <= yMax; ++y) {
        double const yPixelMin = static_cast<double>(y) - 0.5, yPixelMax = static_cast<double>(y) + 0.5;
        BoostPolygon row;  // A polygon of row y
        boost::geometry::assign(
                row, LsstBox(afw::geom::Point2D(xMin, yPixelMin), afw::geom::Point2D(xMax, yPixelMax)));
        std::vector<BoostPolygon> intersections;
        boost::geometry::intersection(_impl->poly, row, intersections);

        if (intersections.size() == 1 && boost::geometry::num_points(intersections[0]) == 5) {
            // This row is fairly tame, and should have a long run of pixels within the polygon
            BoostPolygon const& row = intersections[0];
            std::vector<double> top, bottom;
            top.reserve(2);
            bottom.reserve(2);
            bool failed = false;
            for (std::vector<Point>::const_iterator i = row.outer().begin(); i != row.outer().end() - 1;
                 ++i) {
                double const xCoord = i->getX(), yCoord = i->getY();
                if (yCoord == yPixelMin) {
                    bottom.push_back(xCoord);
                } else if (yCoord == yPixelMax) {
                    top.push_back(xCoord);
                } else {
                    failed = true;
                    break;
                }
            }
            if (!failed && top.size() == 2 && bottom.size() == 2) {
                std::sort(top.begin(), top.end());
                std::sort(bottom.begin(), bottom.end());
                int const xMin = std::min(top[0], bottom[0]);
                int const xStart = ::ceil(std::max(top[0], bottom[0])) + 1;
                int const xStop = std::min(top[1], bottom[1]) - 1;
                int const xMax = ::ceil(std::max(top[1], bottom[1]));
                pixelRowOverlap(image, _impl->poly, std::max(xMin, bbox.getMinX()),
                                std::min(xStart, bbox.getMaxX()), y);
                int x = xStart;
                for (Image::x_iterator i = image->x_at(std::max(xStart, bbox.getMinX()) - image->getX0(),
                                                       y - image->getY0());
                     x <= std::min(xStop, bbox.getMaxX()); ++i, ++x) {
                    *i = 1.0;
                }
                pixelRowOverlap(image, _impl->poly, std::max(xStop, bbox.getMinX()),
                                std::min(xMax, bbox.getMaxX()), y);
                continue;
            }
        }

        // Last resort: do each pixel independently...
        for (std::vector<BoostPolygon>::const_iterator p = intersections.begin(); p != intersections.end();
             ++p) {
            double xMinRow = xMax, xMaxRow = xMin;
            std::vector<LsstPoint> const vertices = p->outer();
            for (std::vector<LsstPoint>::const_iterator q = vertices.begin(); q != vertices.end(); ++q) {
                double const x = q->getX();
                if (x < xMinRow) xMinRow = x;
                if (x > xMaxRow) xMaxRow = x;
            }

            pixelRowOverlap(image, _impl->poly, std::max(static_cast<int>(xMinRow), bbox.getMinX()),
                            std::min(static_cast<int>(::ceil(xMaxRow)), bbox.getMaxX()), y);
        }
    }
    return image;
}

// -------------- Table-based Persistence -------------------------------------------------------------------

/*
 *
 */
namespace {

struct PolygonSchema {
    afw::table::Schema schema;
    afw::table::PointKey<double> vertices;

    static PolygonSchema const& get() {
        static PolygonSchema instance;
        return instance;
    }

    // No copying
    PolygonSchema(const PolygonSchema&) = delete;
    PolygonSchema& operator=(const PolygonSchema&) = delete;

    // No moving
    PolygonSchema(PolygonSchema&&) = delete;
    PolygonSchema& operator=(PolygonSchema&&) = delete;

private:
    PolygonSchema()
            : schema(),
              vertices(afw::table::PointKey<double>::addFields(schema, "vertices", "list of vertex points",
                                                               "")) {
        schema.getCitizen().markPersistent();
    }
};

class PolygonFactory : public table::io::PersistableFactory {
public:
    explicit PolygonFactory(std::string const& name) : table::io::PersistableFactory(name) {}

    virtual std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                         CatalogVector const& catalogs) const {
        static PolygonSchema const& keys = PolygonSchema::get();

        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        afw::table::BaseCatalog const& cat = catalogs.front();

        std::vector<LsstPoint> vertices;
        for (afw::table::BaseCatalog::const_iterator iter = cat.begin(); iter != cat.end(); ++iter) {
            vertices.push_back(iter->get(keys.vertices));
        }
        std::shared_ptr<Polygon> result(new Polygon(vertices));
        return result;
    }
};

std::string getPolygonPersistenceName() { return "Polygon"; }

PolygonFactory registration(getPolygonPersistenceName());

}  // anonymous namespace

std::string Polygon::getPersistenceName() const { return getPolygonPersistenceName(); }

void Polygon::write(OutputArchiveHandle& handle) const {
    static PolygonSchema const& keys = PolygonSchema::get();
    afw::table::BaseCatalog catalog = handle.makeCatalog(keys.schema);

    std::vector<LsstPoint> vertices = this->getVertices();
    for (std::vector<LsstPoint>::const_iterator i = vertices.begin(); i != vertices.end(); ++i) {
        std::shared_ptr<afw::table::BaseRecord> record = catalog.addNew();
        record->set(keys.vertices, *i);
    }

    handle.saveCatalog(catalog);
}
}  // namespace polygon
}  // namespace geom
}  // namespace afw
}  // namespace lsst
