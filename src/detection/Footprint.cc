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
 
/*****************************************************************************/
/** \file
 *
 * \brief Footprint and associated classes
 */
#include <cassert>
#include <string>
#include <typeinfo>
#include <algorithm>
#include "boost/format.hpp"
#include "lsst/pex/logging/Trace.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection/Peak.h"
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelFunctions.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/detection/FootprintFunctor.h"
#include "lsst/afw/detection/FootprintSet.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/ellipses/PixelRegion.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/utils/ieee.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>

using boost::serialization::make_nvp;

namespace lsst {
namespace afw {
namespace detection {

// anonymous namespace
namespace {

/*
 * Compare two Span%s by y, then x0, then x1
 *
 * A utility functor passed to sort; needed to dereference the boost::shared_ptrs.
 */
    struct compareSpanByYX :
        public std::binary_function<Span::ConstPtr, Span::ConstPtr, bool> {
        int operator()(Span::ConstPtr a, Span::ConstPtr b) {
            return (*a) < (*b);
        }
    };

/// Transform x,y in the frame of one image to another, via their WCSes
geom::Point2D transformPoint(double x, double y,
                             image::Wcs const& source,
                             image::Wcs const& target){
    return target.skyToPixel(*source.pixelToSky(x, y));
}


} //end namespace

/*****************************************************************************/
/// Counter for Footprint IDs
int Footprint::id = 0;

Footprint::Footprint(
    int nspan,
    geom::Box2I const & region
) : lsst::daf::base::Citizen(typeid(this)),
    _fid(++id),
    _area(0),
    _bbox(geom::Box2I()),
    _region(region),
    _normalized(true)
{
    if (nspan < 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            str(boost::format("Number of spans requested is -ve: %d") % nspan));
    }
}

Footprint::Footprint(
    geom::Box2I const& bbox, //!< The bounding box defining the rectangle
    geom::Box2I const& region //!< Bounding box of MaskedImage footprint
) : lsst::daf::base::Citizen(typeid(this)),
    _fid(++id),
    _area(0),
    _bbox(bbox),
    _region(region)
{
    int const x0 = bbox.getMinX();
    int const y0 = bbox.getMinY();
    int const x1 = bbox.getMaxX();
    int const y1 = bbox.getMaxY();

    for (int i = y0; i <= y1; ++i) {
        addSpan(i, x0, x1);
    }
    _normalized=true;
}

Footprint::Footprint(
    geom::Point2I const & center,
    double const radius,
    geom::BoxI const & region
) : lsst::daf::base::Citizen(typeid(this)),
    _fid(++id),
    _area(0),
    _bbox(geom::BoxI()),
    _region(region)
{
    int const r2 = static_cast<int>(radius*radius + 0.5); // rounded radius^2
    int const r = static_cast<int>(std::sqrt(static_cast<double>(r2))); // truncated radius; r*r <= r2

    for (int i = -r; i <= r; ++i) {
        int hlen = static_cast<int>(std::sqrt(static_cast<double>(r2 - i*i)));
        addSpan(center.getY() + i, center.getX() - hlen, center.getX() + hlen);
    }
    _normalized = true;
}

Footprint::Footprint(
    geom::ellipses::Ellipse const & ellipse,
    geom::Box2I const & region
) :  lsst::daf::base::Citizen(typeid(this)),
    _fid(++id),
    _area(0),
    _bbox(geom::Box2I()),
    _region(region),
    _normalized(true)
{
    geom::ellipses::PixelRegion pr(ellipse);
    for (
        geom::ellipses::PixelRegion::Iterator spanIter = pr.begin(), end = pr.end();
        spanIter != end;
        ++spanIter
    ) {
        if (!spanIter->isEmpty()) {
            addSpan(*spanIter);
        }
    }
    _normalized = true;
}

Footprint::Footprint(
    Footprint::SpanList const & spans,
    geom::Box2I const & region
) : lsst::daf::base::Citizen(typeid(this)),
    _fid(++id),
    _area(0),
    _bbox(geom::Box2I()),
    _region(region),
    _normalized(false)
{
    _spans.reserve(spans.size());
    for(SpanList::const_iterator i(spans.begin()); i != spans.end(); ++i) {
        addSpan(**i);
    }
}

Footprint::Footprint(Footprint const & other)
  : lsst::daf::base::Citizen(typeid(this)),
    _fid(++id),
    _bbox(other._bbox),
    _region(other._region)
{
    //deep copy spans
    _spans.reserve(other._spans.size());
    for(SpanList::const_iterator i(other._spans.begin());
        i != other._spans.end(); ++i
    ) {
        addSpan(**i);
    }
    _area = other._area;
    _normalized = other._normalized;

    //deep copy peaks
    _peaks.reserve(other._peaks.size());
    for(PeakList::const_iterator i(other._peaks.begin()); i != other._peaks.end(); ++i) {
        _peaks.push_back(Peak::Ptr(new Peak(**i)));
    }
}

Footprint::~Footprint() {
}

bool Footprint::contains(
    lsst::afw::geom::Point2I const& pix ///< Pixel to check
) const
{
    if (_bbox.contains(pix)) {
        for (Footprint::SpanList::const_iterator siter = _spans.begin(); siter != _spans.end(); ++siter) {
            if ((*siter)->contains(pix.getX(), pix.getY())) {
                return true;
            }
        }
    }

    return false;
}

void Footprint::clipTo(geom::Box2I const& bbox) {
    Footprint::SpanList::iterator it = _spans.begin();
    for (; it != _spans.end();) {
        Span *sp = it->get();
        if ((sp->getY() < bbox.getMinY()) ||
            (sp->getY() > bbox.getMaxY())) {
            // out of bounds in y -- erase
            it = _spans.erase(it);
            continue;
        }
        if ((sp->getX0() > bbox.getMaxX()) ||
            (sp->getX1() < bbox.getMinX())) {
            // out of bounds in x; erase
            it = _spans.erase(it);
            continue;
        }
        // clip
        if (sp->getX0() < bbox.getMinX()) {
            // clip span x0 to bbox
            sp->_x0 = bbox.getMinX();
        }
        if (sp->getX1() > bbox.getMaxX()) {
            // clip span x1 to bbox
            sp->_x1 = bbox.getMaxX();
        }
        ++it;
    }

    Footprint::PeakList::iterator pit = _peaks.begin();
    for (; pit != _peaks.end();) {
        Peak *pk = pit->get();
        if (!bbox.contains(geom::Point2I(pk->getIx(), pk->getIy()))) {
            pit = _peaks.erase(pit);
            continue;
        }
        ++pit;
    }

    if (_spans.empty()) {
        _bbox = geom::Box2I();
        _area = 0;
        _normalized = true;
    } else {
        _normalized = false;
        normalize();
    }
}

void Footprint::normalize() {
    if (_normalized) {
        return;
    }
    assert(!_spans.empty());
    //
    // Check that the spans are sorted, and (more importantly) that each pixel appears
    // in only one span
    //
    sort(_spans.begin(), _spans.end(), compareSpanByYX());

    Footprint::SpanList::iterator ptr = _spans.begin(), end = _spans.end();
    Span *lspan = ptr->get();  // Left span
    int y = lspan->_y;
    int x1 = lspan->_x1;
    _area = lspan->getWidth();
    int minX = lspan->_x0, minY=y, maxX=x1;

    ++ptr;

    for (; ptr != end; ++ptr) {
        Span *rspan = ptr->get(); // Right span
        if (rspan->_y == y) {
            if (rspan->_x0 <= x1 + 1) { // Spans overlap or touch
                if (rspan->_x1 > x1) {  // right span extends left span
                    //update area
                    _area += rspan->_x1 - x1;
                    //update end of current span
                    x1 = lspan->_x1 = rspan->_x1;
                    //update bounds
                    if(x1 > maxX) maxX = x1;
                }

                ptr = _spans.erase(ptr);
                end = _spans.end();   // delete the right span
                if (ptr == end) {
                    break;
                }

                --ptr;
                continue;
            }
            else{
                _area += rspan->getWidth();
                if(rspan->_x1 > maxX) maxX = rspan->_x1;
            }
        } else {
            _area += rspan->getWidth();
        }

        y = rspan->_y;
        x1 = rspan->_x1;

        lspan = rspan;
        if(lspan->_x0 < minX) minX = lspan->_x0;
        if(x1 > maxX) maxX = x1;
    }
    _bbox = geom::Box2I(geom::Point2I(minX, minY), geom::Point2I(maxX, y));

    _normalized = true;
}

Span const& Footprint::addSpan(
    int const y, //!< row value
    int const x0, //!< starting column
    int const x1 //!< ending column
) {
    if (x1 < x0) {
        return addSpan(y, x1, x0);
    }

    Span::Ptr sp(new Span(y, x0, x1));
    _spans.push_back(sp);

    _area += sp->getWidth();
    _normalized = false;

    _bbox.include(geom::Point2I(x0, y));
    _bbox.include(geom::Point2I(x1, y));

    return *sp.get();
}

const Span& Footprint::addSpan(
    Span const& span ///< new Span being added
) {
    return addSpan(span._y, span._x0, span._x1);
}

const Span& Footprint::addSpan(
    Span const& span, ///< new Span being added
    int dx,              ///< Add dx to span's x coords
    int dy               ///< Add dy to span's y coords
) {
    return addSpan(span._y + dy, span._x0 + dx, span._x1 + dx);
}

const Span& Footprint::addSpanInSeries(
    int const y, //!< row value
    int const x0, //!< starting column
    int const x1 //!< ending column
) {
    if (x1 < x0) {
        return addSpanInSeries(y, x1, x0);
    }
    if (_spans.size() == 0) {
      const Span& s = addSpan(y, x0, x1);
      _normalized = true;
      return s;
    }
    // merge contiguous spans
    Span::Ptr lastspan = _spans.back();
    if ((y == lastspan->getY()) &&
        (x0 == (lastspan->getX1() + 1))) {
      // contiguous.
      lastspan->_x1 = x1;
      _area += (1 + x1 - x0);
      _bbox.include(geom::Point2I(x1,y));
      return *lastspan;
    }
    if (!((y  >  lastspan->getY()) ||
          (x0 > (lastspan->getX1() + 1)))) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            str(boost::format("addSpanInSeries: new span %i,[%i,%i] is NOT in series after last span "
                              "%i,[%i,%i]") %
                y % x0 % x1 % lastspan->getY() % lastspan->getX0() % lastspan->getX1()));
    }
    const Span& s = addSpan(y, x0, x1);
    _normalized = true;
    return s;
}

void Footprint::shift(
    int dx,
    int dy
) {
    for (SpanList::iterator i = _spans.begin(); i != _spans.end(); ++i){
        Span::Ptr span = *i;

        span->_y += dy;
        span->_x0 += dx;
        span->_x1 += dx;
    }

    _bbox.shift(geom::Extent2I(dx, dy));
}

geom::Point2D
Footprint::getCentroid() const
{
    int n = 0;
    double xc = 0, yc = 0;
    for (Footprint::SpanList::const_iterator siter = _spans.begin(); siter != _spans.end(); ++siter) {
        Span::Ptr const span = *siter;
        int const y = span->getY();
        int const x0 = span->getX0();
        int const x1 = span->getX1();
        int const npix = x1 - x0 + 1;

        n += npix;
        xc += npix*0.5*(x1 + x0);
        yc += npix*y;
    }
    assert(n == _area);

    return geom::Point2D(xc/_area, yc/_area);
}

geom::ellipses::Quadrupole
Footprint::getShape() const
{
    geom::Point2D cen = getCentroid();
    double const xc = cen.getX();
    double const yc = cen.getY();

    double sumxx = 0, sumxy = 0, sumyy = 0;
    for (Footprint::SpanList::const_iterator siter = _spans.begin(); siter != _spans.end(); ++siter) {
        Span::Ptr const span = *siter;
        int const y = span->getY();
        int const x0 = span->getX0();
        int const x1 = span->getX1();
        int const npix = x1 - x0 + 1;

        for (int x = x0; x <= x1; ++x) {
            sumxx += (x - xc)*(x - xc);
        }
        sumxy += npix*(0.5*(x1 + x0) - xc)*(y - yc);
        sumyy += npix*(y - yc)*(y - yc);
    }

    return geom::ellipses::Quadrupole(sumxx/_area, sumyy/_area, sumxy/_area);
}

namespace {
    /*
     * Set the pixels in idImage which are in Footprint by adding or
     * replacing the specified value to the Image
     *
     * The ids that are overwritten are returned for the callers deliction
     */
    template<bool overwriteId, typename PixelT>
    void
    doInsertIntoImage(geom::Box2I const& _region, // unpacked from Footprint
                      Footprint::SpanList const& _spans,      // unpacked from Footprint
                      image::Image<PixelT>& idImage, // Image to contain the footprint
                      boost::uint64_t const id, // Add/replace id to idImage for pixels in Footprint
                      geom::Box2I const& region,              // Footprint's region (default: getRegion())
                      long const mask=0x0,                    // Don't overwrite bits in this mask
                      std::set<boost::uint64_t> *oldIds=NULL // if non-NULL, set the IDs that were overwritten
                   )
    {
        int width, height, x0, y0;
        if(!region.isEmpty()) {
            height = region.getHeight();
            width = region.getWidth();
            x0 = region.getMinX();
            y0 = region.getMinY();
        } else {
            height = _region.getHeight();
            width = _region.getWidth();
            x0 = _region.getMinX();
            y0 = _region.getMinY();
        }

        if (width != idImage.getWidth() || height != idImage.getHeight()) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              str(boost::format("Image of size (%dx%d) doesn't match "
                                                "Footprint's host Image of size (%dx%d)") %
                                  idImage.getWidth() % idImage.getHeight() % width % height));
        }

        if (id & mask) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              str(boost::format("Id 0x%x sets bits in the protected mask 0x%x") % id % mask));
        }

        typename std::set<boost::uint64_t>::const_iterator pos; // hint on where to insert into oldIds
        if (oldIds) {
            pos = oldIds->begin();
        }
        for (Footprint::SpanList::const_iterator spi = _spans.begin(); spi != _spans.end(); ++spi) {
            Span::Ptr const span = *spi;

            int const sy0 = span->getY() - y0;
            if (sy0 < 0 || sy0 >= height) {
                continue;
            }

            int sx0 = span->getX0() - x0;
            if (sx0 < 0) {
                sx0 = 0;
            }
            int sx1 = span->getX1() - x0;
            int const swidth = (sx1 >= width) ? width - sx0 : sx1 - sx0 + 1;

            for (typename image::Image<PixelT>::x_iterator ptr = idImage.x_at(sx0, sy0),
                     end = ptr + swidth; ptr != end; ++ptr) {
                if (overwriteId) {
                    long val = *ptr & ~mask;
                    if (val != 0 and oldIds != NULL) {
                        pos = oldIds->insert(pos, val); // update our hint, pos
                    }
                    *ptr = (*ptr & mask) + id;
                } else {
                    *ptr += id;
                }
            }
        }
    }
}

template<typename PixelT>
void
Footprint::clipToNonzero(typename image::Image<PixelT> const& img) {
    typedef typename lsst::afw::image::Image<PixelT> ImageT;
    int const ix0 = img.getX0();
    int const iy0 = img.getY0();
    const PixelT zero = 0;

    for (SpanList::iterator s = _spans.begin(); s < _spans.end(); ++s) {
        int const y = (*s)->getY();
        int const x0 = (*s)->getX0();
        int const x1 = (*s)->getX1();
        typename ImageT::x_iterator img_it = img.row_begin(y - iy0) + (x0 - ix0);
        int leftx, rightx;
        // find zero pixels on the left...
        for (leftx = x0; leftx <= x1; ++leftx, ++img_it) {
            if (*img_it != zero) {
                break;
            }
        }
        if (leftx > x1) {
            // whole span is zero; drop it.
            _normalized = false;
            _spans.erase(s);
            s--;
            continue;
        }
        // find zero pixels on the right...
        img_it = img.row_begin(y - iy0) + (x1 - ix0);
        for (rightx = x1; rightx >= leftx; --rightx, --img_it) {
            if (*img_it != zero) {
                break;
            }
        }
        if (leftx != x0) {
            (*s)->_x0 = leftx;
            _normalized = false;
        }
        if (rightx != x1) {
            (*s)->_x1 = rightx;
            _normalized = false;
        }
    }

    if (_spans.empty()) {
        _bbox = geom::Box2I();
        _area = 0;
        _normalized = true;
    }
}

template<typename PixelT>
void
Footprint::insertIntoImage(
    typename image::Image<PixelT>& idImage,
    boost::uint64_t const id,
    geom::Box2I const& region
) const
{
    static_cast<void>(doInsertIntoImage<false>(_region, _spans, idImage, id, region));
}

template<typename PixelT>
void
Footprint::insertIntoImage(
    image::Image<PixelT>& idImage,
    boost::uint64_t const id,
    bool overwriteId,
    long const mask,
    std::set<boost::uint64_t> *oldIds,
    geom::Box2I const& region
) const
{
    if (id > std::size_t(std::numeric_limits<PixelT>::max())) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::OutOfRangeException,
            "id out of range for image type"
        );
    }
    if (overwriteId) {
        doInsertIntoImage<true>(_region, _spans, idImage, id, region, mask, oldIds);
    } else {
        doInsertIntoImage<false>(_region, _spans, idImage, id, region, mask, oldIds);
    }
}

// Factory class used for table-based persistence; invoked via registry in afw::table::io
class FootprintFactory : public table::io::PersistableFactory {
public:

    virtual PTR(table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 2u);
        PTR(Footprint) result = boost::make_shared<Footprint>();
        result->readSpans(catalogs.front());
        result->readPeaks(catalogs.back());
        return result;
    }

    explicit FootprintFactory(std::string const & name) : table::io::PersistableFactory(name) {}

};

namespace {

// Singleton helper class that manages the schema and keys for persisting a Footprint
class FootprintPersistenceHelper : private boost::noncopyable {
public:
    table::Schema spanSchema;
    table::Key<int> spanY;
    table::Key<int> spanX0;
    table::Key<int> spanX1;
    table::Schema peakSchema;
    table::Key<float> peakX;
    table::Key<float> peakY;
    table::Key<float> peakValue;

    static FootprintPersistenceHelper const & get() {
        static FootprintPersistenceHelper instance;
        return instance;
    }

private:
    FootprintPersistenceHelper() :
        spanSchema(),
        spanY(spanSchema.addField<int>("y", "row position of span", "pixels")),
        spanX0(spanSchema.addField<int>("x0", "first column of span (inclusive)", "pixels")),
        spanX1(spanSchema.addField<int>("x1", "first column of span (inclusive)", "pixels")),
        peakSchema(),
        peakX(peakSchema.addField<float>("x", "column position of peak", "pixels")),
        peakY(peakSchema.addField<float>("y", "column position of peak", "pixels")),
        peakValue(peakSchema.addField<float>("value", "value of peak pixel", "pixels"))
    {
        spanSchema.getCitizen().markPersistent();
        peakSchema.getCitizen().markPersistent();
    }
};

std::string getFootprintPersistenceName() { return "Footprint"; }

// Insert the factory into the registry (instantiating an instance is sufficient, because
// the code that does the work is in the base class ctor)
FootprintFactory registration(getFootprintPersistenceName());

} // anonymous

std::string Footprint::getPersistenceName() const { return getFootprintPersistenceName(); }

std::string Footprint::getPythonModule() const { return "lsst.afw.detection"; }

void Footprint::write(OutputArchiveHandle & handle) const {
    FootprintPersistenceHelper const & keys = FootprintPersistenceHelper::get();
    afw::table::BaseCatalog spanCat = handle.makeCatalog(keys.spanSchema);
    spanCat.reserve(_spans.size());
    for (SpanList::const_iterator i = _spans.begin(); i != _spans.end(); ++i) {
        PTR(afw::table::BaseRecord) record = spanCat.addNew();
        record->set(keys.spanY, (**i).getY());
        record->set(keys.spanX0, (**i).getX0());
        record->set(keys.spanX1, (**i).getX1());
    }
    handle.saveCatalog(spanCat);
    afw::table::BaseCatalog peakCat = handle.makeCatalog(keys.peakSchema);
    peakCat.reserve(_peaks.size());
    for (PeakList::const_iterator i = _peaks.begin(); i != _peaks.end(); ++i) {
        PTR(afw::table::BaseRecord) record = peakCat.addNew();
        record->set(keys.peakX, (**i).getFx());
        record->set(keys.peakY, (**i).getFy());
        record->set(keys.peakValue, (**i).getPeakValue());
    }
    handle.saveCatalog(peakCat);
}

void Footprint::readSpans(afw::table::BaseCatalog const & spanCat) {
    FootprintPersistenceHelper const & keys = FootprintPersistenceHelper::get();
    for (afw::table::BaseCatalog::const_iterator i = spanCat.begin(); i != spanCat.end(); ++i) {
        addSpan(i->get(keys.spanY), i->get(keys.spanX0), i->get(keys.spanX1));
    }
}

void Footprint::readPeaks(afw::table::BaseCatalog const & peakCat) {
    FootprintPersistenceHelper const & keys = FootprintPersistenceHelper::get();
    for (afw::table::BaseCatalog::const_iterator i = peakCat.begin(); i != peakCat.end(); ++i) {
        _peaks.push_back(
            boost::make_shared<Peak>(i->get(keys.peakX), i->get(keys.peakY), i->get(keys.peakValue))
        );
    }
}

template <typename Archive>
void Footprint::serialize(Archive & ar, const unsigned int version) {
    ar & make_nvp("spans", _spans);
    ar & make_nvp("peaks", _peaks);
    ar & make_nvp("area", _area);
    ar & make_nvp("normalized", _normalized);

    int x0, y0, width, height;
    int rx0, ry0, rwidth, rheight;
    if(Archive::is_saving::value) {
        geom::Box2I const & bbox = getBBox();
        x0 = bbox.getMinX();
        y0 = bbox.getMinY();
        width = bbox.getWidth();
        height = bbox.getHeight();

        geom::Box2I const & region = getRegion();
        rx0 = region.getMinX();
        ry0 = region.getMinY();
        rwidth = region.getWidth();
        rheight = region.getHeight();
    }

    ar & make_nvp("x0", x0) & make_nvp("y0", y0) & make_nvp("width", width) & make_nvp("height", height);
    ar & make_nvp("rx0", rx0) & make_nvp("ry0", ry0) & make_nvp("rwidth", rwidth)
        & make_nvp("rheight", rheight);

    if(Archive::is_loading::value) {
        _bbox = geom::BoxI(geom::Point2I(x0, y0), geom::Extent2I(width, height));
        _region = geom::BoxI(geom::Point2I(rx0, ry0), geom::Extent2I(rwidth, rheight));
    }
}

template void Footprint::serialize(boost::archive::text_oarchive &, unsigned int const);
template void Footprint::serialize(boost::archive::text_iarchive &, unsigned int const);
template void Footprint::serialize(boost::archive::xml_oarchive &, unsigned int const);
template void Footprint::serialize(boost::archive::xml_iarchive &, unsigned int const);
template void Footprint::serialize(boost::archive::binary_oarchive &, unsigned int const);
template void Footprint::serialize(boost::archive::binary_iarchive &, unsigned int const);

Footprint & Footprint::operator=(Footprint & other) {
    _region = other._region;

    //deep copy spans
    _spans = SpanList();
    _spans.reserve(other._spans.size());
    for(SpanList::const_iterator i(other._spans.begin());
        i != other._spans.end(); ++i
    ) {
        addSpan(**i);
    }
    _area = other._area;
    _normalized = other._normalized;
    _bbox = other._bbox;

    //deep copy peaks
    _peaks = PeakList();
    _peaks.reserve(other._peaks.size());
    for(PeakList::iterator i(other._peaks.begin()); i != other._peaks.end(); ++i) {
        _peaks.push_back(Peak::Ptr(new Peak(**i)));
    }
    return *this;
}

template<typename MaskT>
void Footprint::intersectMask(
    lsst::afw::image::Mask<MaskT> const & mask,
    MaskT const bitmask
) {
    geom::Box2I maskBBox = mask.getBBox(image::PARENT);

    //this operation makes no sense on non-normalized footprints.
    //make sure this is normalized
    normalize();

    SpanList::iterator s(_spans.begin());
    while((*s)->getY() < maskBBox.getMinY() && s != _spans.end()){
        ++s;
    }


    int x0, x1, y;
    SpanList maskedSpans;
    int maskedArea=0;
    for( ; s != _spans.end(); ++s) {
        y = (*s)->getY();

        if (y > maskBBox.getMaxY())
            break;

        x0 = (*s)->getX0();
        x1 = (*s)->getX1();

        if(x1 < maskBBox.getMinX() || x0 > maskBBox.getMaxX()) {
            //span is entirely outside the image mask. cannot be used
            continue;
        }

        //clip the span to be within the mask
        if(x0 < maskBBox.getMinX()) x0 = maskBBox.getMinX();
        if(x1 > maskBBox.getMaxX()) x1 = maskBBox.getMaxX();

        //Image iterators are always specified with respect to (0,0)
        //regardless what the image::XY0 is set to.
        typename image::Mask<MaskT>::const_x_iterator mIter = mask.x_at(
            x0 - maskBBox.getMinX(), y - maskBBox.getMinY()
        );

        //loop over all span locations, slicing the span at maskedPixels
        for(int x = x0; x <= x1; ++x, ++mIter) {
            if((*mIter & bitmask) != 0) {
                //masked pixel found within span
                if (x > x0) {
                    //add beginning of span to the output
                    //the fixed span contains all the unmasked pixels up to,
                    //but not including this masked pixel
                    Span::Ptr maskedSpan(new Span(y, x0, x- 1));
                    maskedSpans.push_back(maskedSpan);
                    maskedArea += maskedSpan->getWidth();
                }
                //set the next Span to start after this pixel
                x0 = x + 1;
            }
        }

        //add last section of span
        if(x0 <= x1) {
            Span::Ptr maskedSpan(new Span(y, x0, x1));
            maskedSpans.push_back(maskedSpan);
            maskedArea += maskedSpan->getWidth();
        }
    }
    _area = maskedArea;
    _spans = maskedSpans;
    _bbox.clip(maskBBox);
}


PTR(Footprint) Footprint::transform(
    image::Wcs const& source,
    image::Wcs const& target,
    geom::Box2I const& region,
    bool doClip
) const {
    // Transform the original bounding box
    geom::Box2I const& fpBox = getBBox(); // Original bounding box
    geom::Box2D tBoxD;
    // If slow, could consider linearising the WCSes and combining the
    // linear versions to a single transform, and then using that to
    // transform all the points.
    tBoxD.include(transformPoint(fpBox.getMinX(), fpBox.getMinY(), source, target));
    tBoxD.include(transformPoint(fpBox.getMinX(), fpBox.getMaxY(), source, target));
    tBoxD.include(transformPoint(fpBox.getMaxX(), fpBox.getMinY(), source, target));
    tBoxD.include(transformPoint(fpBox.getMaxX(), fpBox.getMaxY(), source, target));
    geom::Box2I tBoxI(tBoxD);

    // enumerate points in the new bbox that, when reverse-transformed, are within the given footprint.
    PTR(Footprint) fpNew = boost::make_shared<Footprint>(0, region);

    for (int y = tBoxI.getBeginY(); y < tBoxI.getEndY(); ++y) {
        bool inSpan = false;            // Are we in a span?
        int start = -1;                  // Start of span

        for (int x = tBoxI.getBeginX(); x < tBoxI.getEndX(); ++x) {
            lsst::afw::geom::Point2D const& p = transformPoint(x, y, target, source);
            int const xSource = std::floor(0.5 + p.getX());
            int const ySource = std::floor(0.5 + p.getY());

            if (contains(lsst::afw::geom::Point2I(xSource, ySource))) {
                if (!inSpan) {
                    inSpan = true;
                    start = x;
                }
            } else if (inSpan) {
                inSpan = false;
                fpNew->addSpan(y, start, x - 1);
            }
        }
        if (inSpan) {
            fpNew->addSpan(y, start, tBoxI.getMaxX());
        }
    }
    if (doClip) {
        fpNew->clipTo(region);
    }
    return fpNew;
}

/**
   Returns *true* iff this Footprint satisfies the "normalized" conditions.

   Useful as an "assert" during algorithm development.
 */
bool _checkNormalized(Footprint const& foot) {
    Footprint copy(foot);
    copy.normalize();
    if (copy.getArea() != foot.getArea()) {
        return false;
    }
    if (copy.getSpans().size() != foot.getSpans().size()) {
        return false;
    }
    Footprint::SpanList const& spansa = foot.getSpans();
    Footprint::SpanList const& spansb = copy.getSpans();
    Footprint::SpanList::const_iterator spa = spansa.begin();
    Footprint::SpanList::const_iterator spb = spansb.begin();
    for (; spa != spansa.end(); ++spa, ++spb) {
        if ((*spa)->getY() != (*spb)->getY()) {
            return false;
        }
        if ((*spa)->getX0() != (*spb)->getX0()) {
            return false;
        }
        if ((*spa)->getX1() != (*spb)->getX1()) {
            return false;
        }
    }
    return true;
}

/************************************************************************************************************/

template<typename MaskT>
Footprint::Ptr footprintAndMask(
        Footprint::Ptr const& fp,                                   ///< The initial Footprint
        typename lsst::afw::image::Mask<MaskT>::Ptr const& mask,    ///< The mask to & with foot
        MaskT const bitmask                                       ///< Only consider these bits
) {
    Footprint::Ptr newFp(new Footprint());
    return newFp;
}

/************************************************************************************************************/

template<typename MaskT>
MaskT setMaskFromFootprint(
    image::Mask<MaskT> *mask,              ///< Mask to set
    Footprint const& foot,      ///< Footprint specifying desired pixels
    MaskT const bitmask                    ///< Bitmask to OR into mask
) {

    int const width = static_cast<int>(mask->getWidth());
    int const height = static_cast<int>(mask->getHeight());

    for (Footprint::SpanList::const_iterator siter = foot.getSpans().begin();
         siter != foot.getSpans().end(); ++siter) {
        Span::Ptr const span = *siter;
        int const y = span->getY() - mask->getY0();
        if (y < 0 || y >= height) {
            continue;
        }

        int x0 = span->getX0() - mask->getX0();
        int x1 = span->getX1() - mask->getX0();
        x0 = (x0 < 0) ? 0 : (x0 >= width ? width - 1 : x0);
        x1 = (x1 < 0) ? 0 : (x1 >= width ? width - 1 : x1);

        for (typename image::Image<MaskT>::x_iterator ptr = mask->x_at(x0, y),
                 end = mask->x_at(x1 + 1, y); ptr != end; ++ptr) {
            *ptr |= bitmask;
        }
    }

    return bitmask;
}

/************************************************************************************************************/

template<typename MaskT>
MaskT clearMaskFromFootprint(
    image::Mask<MaskT> *mask,              ///< Mask to set
    Footprint const& foot,      ///< Footprint specifying desired pixels
    MaskT const bitmask                    ///< Bitmask
) {
    int const width = static_cast<int>(mask->getWidth());
    int const height = static_cast<int>(mask->getHeight());

    for (Footprint::SpanList::const_iterator siter = foot.getSpans().begin();
         siter != foot.getSpans().end(); ++siter) {
        Span::Ptr const span = *siter;
        int const y = span->getY() - mask->getY0();
        if (y < 0 || y >= height) {
            continue;
        }

        int x0 = span->getX0() - mask->getX0();
        int x1 = span->getX1() - mask->getX0();
        x0 = (x0 < 0) ? 0 : (x0 >= width ? width - 1 : x0);
        x1 = (x1 < 0) ? 0 : (x1 >= width ? width - 1 : x1);

        for (typename image::Image<MaskT>::x_iterator ptr = mask->x_at(x0, y),
                 end = mask->x_at(x1 + 1, y); ptr != end; ++ptr) {
            *ptr &= ~bitmask;
        }
    }

    return bitmask;
}

/************************************************************************************************************/

template<typename MaskT>
MaskT setMaskFromFootprintList(
        image::Mask<MaskT> *mask,                        ///< Mask to set
        std::vector<Footprint::Ptr> const& footprints,  ///< Footprint list specifying desired pixels
        MaskT const bitmask                                 ///< Bitmask to OR into mask
) {
    for (std::vector<Footprint::Ptr>::const_iterator fiter = footprints.begin();
         fiter != footprints.end(); ++fiter) {
        (void)setMaskFromFootprint(mask, **fiter, bitmask);
    }

    return bitmask;
}

/************************************************************************************************************/

template<typename MaskT>
MaskT setMaskFromFootprintList(
        image::Mask<MaskT> *mask,                        ///< Mask to set
        CONST_PTR(std::vector<Footprint::Ptr>) const & footprints,  ///< Footprint list specifying desired pixels
        MaskT const bitmask                                 ///< Bitmask to OR into mask
                                         ) {
    return setMaskFromFootprintList(mask, *footprints, bitmask);
}

/************************************************************************************************************/
namespace {
template<typename ImageT>
class SetFootprint : public FootprintFunctor<ImageT> {
public:
    SetFootprint(ImageT const& image,
                 typename ImageT::Pixel value) :
        FootprintFunctor<ImageT>(image), _value(value) {}


    void operator()(typename ImageT::xy_locator loc, int, int) {
        *loc = _value;
    }
private:
    typename ImageT::Pixel _value;
};
}

template<typename ImageT>
typename ImageT::Pixel setImageFromFootprint(
        ImageT *image,                    ///< image to set
        Footprint const& foot, ///< Footprint defining desired pixels
        typename ImageT::Pixel const value ///< value to set Image to
) {
    SetFootprint<ImageT> setit(*image, value);
    setit.apply(foot);

    return value;
}

template<typename ImageT>
typename ImageT::Pixel setImageFromFootprintList(
        ImageT *image,                                  ///< image to set
        CONST_PTR(std::vector<Footprint::Ptr>) footprints,  ///< Footprint list specifying desired pixels
        typename ImageT::Pixel const value              ///< value to set Image to
                                                           ) {
    return setImageFromFootprintList(image, *footprints, value);
}

template<typename ImageT>
typename ImageT::Pixel setImageFromFootprintList(
        ImageT *image,                                  ///< image to set
        std::vector<Footprint::Ptr> const& footprints,  ///< Footprint list specifying desired pixels
        typename ImageT::Pixel const value              ///< value to set Image to
) {
    SetFootprint<ImageT> setit(*image, value);
    for (std::vector<Footprint::Ptr>::const_iterator fiter = footprints.begin(),
             end = footprints.end(); fiter != end; ++fiter) {
        setit.apply(**fiter);
    }

    return value;
}

/************************************************************************************************************/
/*
 * Worker routine for the pmSetFootprintArrayIDs/pmSetFootprintID (and pmMergeFootprintArrays)
 */
template <typename IDPixelT>
static void set_footprint_id(
    typename image::Image<IDPixelT>::Ptr idImage,   // the image to set
    Footprint const& foot, // the footprint to insert
    int const id,                     // the desired ID
    int dx=0, int dy=0                // Add these to all x/y in the Footprint
) {
    for (Footprint::SpanList::const_iterator i = foot.getSpans().begin();
         i != foot.getSpans().end(); ++i) {
        Span::Ptr const span = *i;
        for (typename image::Image<IDPixelT>::x_iterator ptr =
                 idImage->x_at(span->getX0() + dx, span->getY() + dy),
                 end = ptr + span->getWidth(); ptr != end; ++ptr) {
            *ptr = id;
        }
    }
}

template <typename IDPixelT>
static void
set_footprint_array_ids(
    typename image::Image<IDPixelT>::Ptr idImage, // the image to set
    std::vector<Footprint::Ptr> const& footprints, // the footprints to insert
    bool const relativeIDs // show IDs starting at 0, not Footprint->id
) {
    int id = 0;                         // first index will be 1

    for (std::vector<Footprint::Ptr>::const_iterator fiter = footprints.begin();
         fiter != footprints.end(); ++fiter) {
        Footprint::Ptr const foot = *fiter;

        if (relativeIDs) {
            ++id;
        } else {
            id = foot->getId();
        }

        set_footprint_id<IDPixelT>(idImage, *foot, id);
    }
}

template void set_footprint_array_ids<int>(
    image::Image<int>::Ptr idImage,
    std::vector<Footprint::Ptr> const& footprints,
    bool const relativeIDs);

/******************************************************************************/
/*
 * Set an image to the value of footprint's ID wherever they may fall
 *
 * @param footprints the footprints to insert
 * @param relativeIDs show the IDs starting at 1, not pmFootprint->id
 */
template <typename IDImageT>
typename boost::shared_ptr<image::Image<IDImageT> > setFootprintArrayIDs(
    std::vector<Footprint::Ptr> const& footprints,
    bool const relativeIDs
) {
    std::vector<Footprint::Ptr>::const_iterator fiter = footprints.begin();
    if (fiter == footprints.end()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            "You didn't provide any footprints"
        );
    }
    Footprint::Ptr const foot = *fiter;

    typename image::Image<IDImageT>::Ptr idImage(
        new image::Image<IDImageT>(foot->getRegion())
    );
    *idImage = 0;
    /*
     * do the work
     */
    set_footprint_array_ids<IDImageT>(idImage, footprints, relativeIDs);

    return idImage;
}

template image::Image<int>::Ptr setFootprintArrayIDs(
    std::vector<Footprint::Ptr> const& footprints,
    bool const relativeIDs);
/*
 * Set an image to the value of Footprint's ID wherever it may fall
 */
template <typename IDImageT>
typename boost::shared_ptr<image::Image<IDImageT> > setFootprintID(
                                          Footprint::Ptr const& foot, // the Footprint to insert
                                          int const id // the desired ID
                                                                     ) {
    typename image::Image<IDImageT>::Ptr idImage(new image::Image<IDImageT>(foot->getBBox()));
    *idImage = 0;
    /*
     * do the work
     */
    set_footprint_id<IDImageT>(idImage, *foot, id);

    return idImage;
}

template image::Image<int>::Ptr setFootprintID(Footprint::Ptr const& foot, int const id);

/************************************************************************************************************/
/*
 * Grow a Footprint isotropically by r pixels, returning a new Footprint
 *
 * N.b. this is slow, as it uses a convolution with a disk
 */
namespace {
Footprint::Ptr growFootprintSlow(
        Footprint const& foot, //!< The Footprint to grow
        int ngrow                              //!< how much to grow foot
                                                 ) {
    if (ngrow < 0) {
        ngrow = 0;                      // ngrow == 0 => no grow
    }

    if (foot.getNpix() == 0) {          // an empty Footprint
        return Footprint::Ptr(new Footprint);
    }

    /*
     * We'll insert the footprints into an image, then convolve with a disk,
     * then extract a footprint from the result --- this is magically what we want.
     */
    geom::Box2I bbox = foot.getBBox();
    bbox.grow(2*ngrow);
    image::Image<int>::Ptr idImage(new image::Image<int>(bbox));
    *idImage = 0;
    idImage->setXY0(0, 0);

    set_footprint_id<int>(idImage, foot, 1, -bbox.getMinX(), -bbox.getMinY());


    image::Image<double>::Ptr circle_im(
        new image::Image<double>(geom::Extent2I(2*ngrow + 1, 2*ngrow + 1))
    );
    *circle_im = 0;
    for (int r = -ngrow; r <= ngrow; ++r) {
        image::Image<double>::x_iterator row = circle_im->x_at(0, r + ngrow);
        for (int c = -ngrow; c <= ngrow; ++c, ++row) {
            if (r*r + c*c <= ngrow*ngrow) {
                *row = 8;
            }
        }
    }

    math::FixedKernel::Ptr circle(new math::FixedKernel(*circle_im));
    // Here's the actual grow step
    image::MaskedImage<int>::Ptr convolvedImage(new image::MaskedImage<int>(idImage->getDimensions()));
    math::convolve(*convolvedImage->getImage(), *idImage, *circle, false);

    PTR(FootprintSet) grownList(new FootprintSet(*convolvedImage, 0.5, "", 1));

    assert (grownList->getFootprints()->size() > 0);
    Footprint::Ptr grown = *grownList->getFootprints()->begin();
    //
    // Fix the coordinate system to be that of foot
    //
    grown->shift(bbox.getMinX(), bbox.getMinY());
    grown->setRegion(foot.getRegion());

    return grown;
}
}

/************************************************************************************************************/

namespace {
    Footprint::Ptr _mergeFootprints(Footprint const& aFoot, Footprint const& bFoot) {
        PTR(Footprint) foot(new Footprint());

        Footprint::PeakList const& aPeaks = aFoot.getPeaks();
        Footprint::PeakList const& bPeaks = bFoot.getPeaks();
        Footprint::PeakList & peaks = foot->getPeaks();
        peaks.reserve(aPeaks.size() + bPeaks.size());
        peaks.insert(peaks.begin(), aPeaks.begin(), aPeaks.end());
        peaks.insert(peaks.end()-1, bPeaks.begin(), bPeaks.end());
        assert(peaks.size() == (aPeaks.size() + bPeaks.size()));

        Footprint::SpanList const& aSpans = aFoot.getSpans();
        Footprint::SpanList const& bSpans = bFoot.getSpans();
        Footprint::SpanList::const_iterator aSpan = aSpans.begin();
        Footprint::SpanList::const_iterator bSpan = bSpans.begin();
        Footprint::SpanList::const_iterator aEnd = aSpans.end();
        Footprint::SpanList::const_iterator bEnd = bSpans.end();

        foot->getSpans().reserve(std::max(aSpans.size(), bSpans.size()));

        while ((aSpan != aEnd) && (bSpan != bEnd)) {
            int y = (*aSpan)->getY();
            int x0 = (*aSpan)->getX0();
            int x1 = (*aSpan)->getX1();
            int yb  = (*bSpan)->getY();
            int xb0 = (*bSpan)->getX0();
            int xb1 = (*bSpan)->getX1();

            if ((y < yb) || (y == yb && (x1 < (xb0-1)))) {
                // A is earlier -- add A
                foot->addSpanInSeries(y, x0, x1);
                ++aSpan;
                continue;
            }
            if ((yb < y) || (y == yb && (xb1 < (x0-1)))) {
                // B is earlier -- add B
                foot->addSpanInSeries(yb, xb0, xb1);
                ++bSpan;
                continue;
            }

            assert(yb == y);
            // Overlap -- find connected spans from both iterators.
            x0 = std::min(x0, xb0);
            x1 = std::max(x1, xb1);
            // Union all connected spans
            ++aSpan;
            ++bSpan;
            while (true) {
                if ((aSpan != aEnd) &&
                    ((*aSpan)->getY() == y) &&
                    ((*aSpan)->getX0() <= (x1+1))) {
                    // *aSpan continues this span.
                    x1 = std::max(x1, (*aSpan)->getX1());
                    ++aSpan;
                    continue;
                }
                if ((bSpan != bEnd) &&
                    ((*bSpan)->getY() == y) &&
                    ((*bSpan)->getX0() <= (x1+1))) {
                    // *bSpan continues this span.
                    x1 = std::max(x1, (*bSpan)->getX1());
                    ++bSpan;
                    continue;
                }
                break;
            }
            foot->addSpanInSeries(y, x0, x1);
        }
        // At this point either "aSpan" or "bSpan" is at the end.

        // Add any remaining spans from "A".
        for (; aSpan != aEnd; ++aSpan) {
            foot->addSpanInSeries((*aSpan)->getY(), (*aSpan)->getX0(), (*aSpan)->getX1());
        }
        // Add any remaining spans from "B".
        for (; bSpan != bEnd; ++bSpan) {
            foot->addSpanInSeries((*bSpan)->getY(), (*bSpan)->getX0(), (*bSpan)->getX1());
        }
        return foot;
    }
}

Footprint::Ptr mergeFootprints(Footprint& foot1, Footprint& foot2) {
    foot1.normalize();
    foot2.normalize();
    return _mergeFootprints(foot1, foot2);
}

Footprint::Ptr mergeFootprints(Footprint const& foot1, Footprint const& foot2) {
    if (!foot1.isNormalized() || !foot2.isNormalized()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            "mergeFootprints(const Footprints) requires normalize()d Footprints.");
    }
    return _mergeFootprints(foot1, foot2);
}

/************************************************************************************************************/

void nearestFootprint(std::vector<Footprint::Ptr> const& foots,
                      image::Image<boost::uint16_t>::Ptr argmin,
                      image::Image<boost::uint16_t>::Ptr dist)
{
    /*
     * insert the footprints into an image, set all the pixels to the
     * Manhattan distance from the nearest set pixel.
     */
    typedef boost::uint16_t dtype;
    typedef boost::uint16_t itype;

    const itype nil = 0xffff;

    const geom::Box2I bbox = argmin->getBBox(image::PARENT);
    *argmin = 0;
    *dist = 0;

    int const x0 = bbox.getMinX();
    int const y0 = bbox.getMinY();

    for (size_t i=0; i<foots.size(); ++i) {
        // Set all the pixels in the footprint to 1
        set_footprint_id<itype>(argmin, *foots[i], i, -x0, -y0);
        set_footprint_id<dtype>(dist,   *foots[i], 1, -x0, -y0);
    }

    int const height = dist->getHeight();
    int const width  = dist->getWidth();

    // traverse from bottom left to top right
    for (int y = 0; y != height; ++y) {
        image::Image<dtype>::xy_locator dim = dist->xy_at(0, y);
        image::Image<itype>::xy_locator aim = argmin->xy_at(0, y);
        for (int x = 0; x != width; ++x, ++dim.x(), ++aim.x()) {
            if (dim(0, 0) == 1) {
                // first pass and pixel was on, it gets a zero
                dim(0, 0) = 0;
                // its argmin is already set
            } else {
                // pixel was off. It is at most the sum of lengths of
                // the array away from a pixel that is on
                dim(0, 0) = width + height;
                aim(0, 0) = nil;
                // or one more than the pixel to the north
                if (y > 0) {
                    dtype ndist = dim(0,-1) + 1;
                    if (ndist < dim(0,0)) {
                        dim(0,0) = ndist;
                        aim(0,0) = aim(0,-1);
                    }
                }
                // or one more than the pixel to the west
                if (x > 0) {
                    dtype ndist = dim(-1,0) + 1;
                    if (ndist < dim(0,0)) {
                        dim(0,0) = ndist;
                        aim(0,0) = aim(-1,0);
                    }
                }
            }
        }
    }
    // traverse from top right to bottom left
    for (int y = height - 1; y >= 0; --y) {
        image::Image<dtype>::xy_locator dim = dist->xy_at(width-1, y);
        image::Image<itype>::xy_locator aim = argmin->xy_at(width-1, y);
        for (int x = width - 1; x >= 0; --x, --dim.x(), --aim.x()) {
            // either what we had on the first pass or one more than
            // the pixel to the south
            if (y + 1 < height) {
                dtype ndist = dim(0,1) + 1;
                if (ndist < dim(0,0)) {
                    dim(0,0) = ndist;
                    aim(0,0) = aim(0,1);
                }
            }
            // or one more than the pixel to the east
            if (x + 1 < width) {
                dtype ndist = dim(1,0) + 1;
                if (ndist < dim(0,0)) {
                    dim(0,0) = ndist;
                    aim(0,0) = aim(1,0);
                }
            }
        }
    }
}

Footprint::Ptr growFootprint(
        Footprint const& foot,          //!< The Footprint to grow
        int ngrow,                      //!< how much to grow foot
        bool isotropic                  //!< Grow isotropically (as opposed to a Manhattan metric)
                                        //!< @note Isotropic grows are significantly slower
                            )
{
    if (isotropic) {
        return growFootprintSlow(foot, ngrow);
    }

    if (ngrow < 0) {
        ngrow = 0;                      // ngrow == 0 => no grow
    }
    /*
     * We'll insert the footprints into an image, set all the pixels
     * to the Manhattan distance from the nearest set pixel, then
     * extract a footprint from the result
     *
     * Cf. http://ostermiller.org/dilate_and_erode.html
     */
    geom::Box2I bbox = foot.getBBox();
    bbox.grow(ngrow);
    image::Image<int>::Ptr idImage(new image::Image<int>(bbox));
    *idImage = 0;
    idImage->setXY0(0, 0);

    // Set all the pixels in the footprint to 1
    set_footprint_id<int>(idImage, foot, 1, -bbox.getMinX(), -bbox.getMinY());
    //
    // Set the idImage to the Manhattan distance from the nearest set pixel
    //
    int const height = idImage->getHeight();
    int const width = idImage->getWidth();

    // traverse from bottom left to top right
    for (int y = 0; y != height; ++y) {
        image::Image<int>::xy_locator im = idImage->xy_at(0, y);

        for (int x = 0; x != width; ++x, ++im.x()) {
            if (im(0, 0) == 1) {
                // first pass and pixel was on, it gets a zero
                im(0, 0) = 0;
            } else {
                // pixel was off. It is at most the sum of lengths of the array away from a pixel that is on
                im(0, 0) = width + height;
                // or one more than the pixel to the north
                if (y > 0) {
                    // im(0, 0)[0] == static_cast<int>(im(0, 0))
                    im(0, 0) = std::min(im(0, 0)[0], im(0, -1) + 1);
                }
                // or one more than the pixel to the west
                if (x > 0) {
                    im(0, 0) = std::min(im(0, 0)[0], im(-1, 0) + 1);
                }
            }
        }
    }
    // traverse from top right to bottom left
    for (int y = height - 1; y >= 0; --y) {
        image::Image<int>::xy_locator im = idImage->xy_at(width - 1, y);
        for (int x = width - 1; x >= 0; --x, --im.x()) {
            // either what we had on the first pass or one more than the pixel to the south
            if (y + 1 < height) {
                im(0, 0) = std::min(im(0, 0)[0], im(0, 1) + 1);
            }
            // or one more than the pixel to the east
            if (x + 1 < width) {
                im(0, 0) = std::min(im(0, 0)[0], im(1, 0) + 1);
            }
        }
    }

    image::MaskedImage<int>::Ptr midImage(new image::MaskedImage<int>(idImage));
    // XXX Why do I need a -ve threshold when parity == false? I'm looking for pixels below ngrow
    PTR(FootprintSet) grownList(new FootprintSet(*midImage, Threshold(-ngrow, Threshold::VALUE, false)));
    assert (grownList->getFootprints()->size() > 0);
    Footprint::Ptr grown = *grownList->getFootprints()->begin();
    //
    // Fix the coordinate system to be that of foot
    //
    grown->shift(bbox.getMinX(), bbox.getMinY());
    grown->setRegion(foot.getRegion());

    return grown;
}

Footprint::Ptr growFootprint(Footprint::Ptr const& foot, int ngrow, bool isotropic) {
    return growFootprint(*foot, ngrow, isotropic);
}

PTR(Footprint) growFootprint(Footprint const& old, ///< Footprint to grow
                             int nGrow,            ///< How many pixels to grow it
                             bool left,            ///< grow to the left
                             bool right,           ///< grow to the right
                             bool up,              ///< grow up
                             bool down             ///< grow down
                            )
{
    Footprint::Ptr grown(new Footprint(0, old.getRegion()));

    for (Footprint::SpanList::const_iterator siter = old.getSpans().begin();
            siter != old.getSpans().end(); ++siter) {
        CONST_PTR(Span) span = *siter;
        int y=span->getY();
        int x0 = (left) ? span->getX0() - nGrow : span->getX0();
        int x1 = (right) ? span->getX1() + nGrow : span->getX1();
        grown->addSpan(y, x0, x1);
        if (up) {
            for(int i=1; i <=nGrow; ++i) {
                grown->addSpan(y+i,span->getX0(), span->getX1());
            }
        }
        if (down) {
            for(int i=1; i <=nGrow; ++i) {
                grown->addSpan(y-i, span->getX0(), span->getX1());
            }
        }
    }

    //normalize to remove overlapped spans and correct bbox
    grown->normalize();
    return grown;
}

/************************************************************************************************************/

std::vector<geom::Box2I> footprintToBBoxList(Footprint const& foot) {
    typedef boost::uint16_t ImageT;
    geom::Box2I fpBBox = foot.getBBox();
    image::Image<ImageT>::Ptr idImage(
        new image::Image<ImageT>(fpBBox.getDimensions())
    );
    *idImage = 0;
    int const height = fpBBox.getHeight();
    geom::Extent2I shift(fpBBox.getMinX(), fpBBox.getMinY());
    foot.insertIntoImage(*idImage, 1, fpBBox);

    std::vector<geom::Box2I> bboxes;
    /*
     * Our strategy is to find a row of pixels in the Footprint and interpret it as the first
     * row of a rectangular set of pixels.  We then extend this rectangle upwards as far as it
     * will go, and define that as a BBox.  We clear all those pixels, and repeat until there
     * are none left.  I.e. a Footprint will get cut up like this:
     *
     *       .555...
     *       22.3314
     *       22.331.
     *       .000.1.
     * (as shown in Footprint_1.py)
     */

    int y0 = 0;                         // the first row with non-zero pixels in it
    while (y0 < height) {
        geom::Box2I bbox;            // our next BBox
        for (int y = y0; y != height; ++y) {
            // Look for a set pixel in this row
            image::Image<ImageT>::x_iterator begin = idImage->row_begin(y), end = idImage->row_end(y);
            image::Image<ImageT>::x_iterator first = std::find(begin, end, 1);

            if (first != end) {                     // A pixel is set in this row
                image::Image<ImageT>::x_iterator last = std::find(first, end, 0) - 1;
                int const x0 = first - begin;
                int const x1 = last  - begin;

                std::fill(first, last + 1, 0);       // clear pixels; we don't want to see them again

                bbox.include(geom::Point2I(x0, y));     // the LLC
                bbox.include(geom::Point2I(x1, y));     // the LRC; initial guess for URC

                // we found at least one pixel so extend the BBox upwards
                for (++y; y != height; ++y) {
                    if (std::find(idImage->at(x0, y), idImage->at(x1 + 1, y), 0) != idImage->at(x1 + 1, y)) {
                        break;  // some pixels weren't set, so the BBox stops here, (actually in previous row)
                    }
                    std::fill(idImage->at(x0, y), idImage->at(x1 + 1, y), 0);

                    bbox.include(geom::Point2I(x1, y)); // the new URC
                }

                bbox.shift(shift);
                bboxes.push_back(bbox);
            } else {
                y0 = y + 1;
            }
            break;
        }
    }

    return bboxes;
}


template<typename ImageOrMaskedImageT>
void
copyWithinFootprint(Footprint const& foot,
                    PTR(ImageOrMaskedImageT) const input,
                    PTR(ImageOrMaskedImageT) output) {
    Footprint::SpanList spans = foot.getSpans();
    for (Footprint::SpanList::iterator sp = spans.begin();
         sp != spans.end(); ++sp) {
        int y  = (*sp)->getY();
        int x0 = (*sp)->getX0();
        int x1 = (*sp)->getX1();
        typename ImageOrMaskedImageT::const_x_iterator initer = input->x_at(
            x0 - input->getX0(),  y - input->getY0());
        typename ImageOrMaskedImageT::x_iterator outiter = output->x_at(
            x0 - output->getX0(), y - output->getY0());
        for (int x=x0; x <= x1; ++x, ++initer, ++outiter) {
            *outiter = *initer;
        }
    }
}




#if 0

/************************************************************************************************************/
/*
 * Grow a psArray of pmFootprints isotropically by r pixels, returning a new psArray of new pmFootprints
 */
psArray *pmGrowFootprintArray(psArray const *footprints, // footprints to grow
                              int r) {  // how much to grow each footprint
    assert (footprints->n == 0 || pmIsFootprint(footprints->data[0]));

    if (footprints->n == 0) {           // we don't know the size of the footprint's region
        return psArrayAlloc(0);
    }
    /*
     * We'll insert the footprints into an image, then convolve with a disk,
     * then extract a footprint from the result --- this is magically what we want.
     */
    psImage *idImage = pmSetFootprintArrayIDs(footprints, true);
    if (r <= 0) {
        r = 1;                          // r == 1 => no grow
    }
    psKernel *circle = psKernelAlloc(-r, r, -r, r);
    assert (circle->image->numRows == 2*r + 1 && circle->image->numCols == circle->image->numRows);
    for (int i = 0; i <= r; ++i) {
        for (int j = 0; j <= r; ++j) {
            if (i*i + j*j <= r*r) {
                circle->kernel[i][j] =
                    circle->kernel[i][-j] =
                    circle->kernel[-i][j] =
                    circle->kernel[-i][-j] = 1;
            }
        }
    }

    psImage *grownIdImage = psImageConvolveDirect(idImage, circle); // Here's the actual grow step
    psFree(circle);

    psArray *grown = pmFindFootprints(grownIdImage, 0.5, 1); // and here we rebuild the grown footprints
    assert (grown != NULL);
    psFree(idImage);
    psFree(grownIdImage);
    /*
     * Now assign the peaks appropriately.  We could do this more efficiently
     * using grownIdImage (which we just freed), but this is easy and probably fast enough
     */
    psArray const *peaks = pmFootprintArrayToPeaks(footprints);
    pmPeaksAssignToFootprints(grown, peaks);
    psFree((psArray *)peaks);

    return grown;
}

/************************************************************************************************************/
/*
 * Merge together two psArrays of pmFootprints neither of which is damaged.
 *
 * The returned psArray may contain elements of the inital psArrays (with
 * their reference counters suitable incremented)
 */
psArray *pmMergeFootprintArrays(psArray const *footprints1, // one set of footprints
                                psArray const *footprints2, // the other set
                                int const includePeaks) { // which peaks to set? 0x1 => footprints1, 0x2 => 2
    assert (footprints1->n == 0 || pmIsFootprint(footprints1->data[0]));
    assert (footprints2->n == 0 || pmIsFootprint(footprints2->data[0]));

    if (footprints1->n == 0 || footprints2->n == 0) {           // nothing to do but put copies on merged
        psArray const *old = (footprints1->n == 0) ? footprints2 : footprints1;

        psArray *merged = psArrayAllocEmpty(old->n);
        for (int i = 0; i < old->n; ++i) {
            psArrayAdd(merged, 1, old->data[i]);
        }

        return merged;
    }
    /*
     * We have real work to do as some pmFootprints in footprints2 may overlap
     * with footprints1
     */
    {
        pmFootprint *fp1 = footprints1->data[0];
        pmFootprint *fp2 = footprints2->data[0];
        if (fp1->region.x0 != fp2->region.x0 ||
            fp1->region.x1 != fp2->region.x1 ||
            fp1->region.y0 != fp2->region.y0 ||
            fp1->region.y1 != fp2->region.y1) {
            psError(PS_ERR_BAD_PARAMETER_SIZE, true,
                    "The two pmFootprint arrays correspnond to different-sized regions");
            return NULL;
        }
    }
    /*
     * We'll insert first one set of footprints then the other into an image, then
     * extract a footprint from the result --- this is magically what we want.
     */
    psImage *idImage = pmSetFootprintArrayIDs(footprints1, true);
    set_footprint_array_ids(idImage, footprints2, true);

    psArray *merged = pmFindFootprints(idImage, 0.5, 1);
    assert (merged != NULL);
    psFree(idImage);
    /*
     * Now assign the peaks appropriately.  We could do this more efficiently
     * using idImage (which we just freed), but this is easy and probably fast enough
     */
    if (includePeaks & 0x1) {
        psArray const *peaks = pmFootprintArrayToPeaks(footprints1);
        pmPeaksAssignToFootprints(merged, peaks);
        psFree((psArray *)peaks);
    }

    if (includePeaks & 0x2) {
        psArray const *peaks = pmFootprintArrayToPeaks(footprints2);
        pmPeaksAssignToFootprints(merged, peaks);
        psFree((psArray *)peaks);
    }

    return merged;
}

/************************************************************************************************************/
/*
 * Given a psArray of pmFootprints and another of pmPeaks, assign the peaks to the
 * footprints in which that fall; if they _don't_ fall in a footprint, add a suitable
 * one to the list.
 */
psErrorCode
pmPeaksAssignToFootprints(psArray *footprints,  // the pmFootprints
                          psArray const *peaks) { // the pmPeaks
    assert (footprints != NULL);
    assert (footprints->n == 0 || pmIsFootprint(footprints->data[0]));
    assert (peaks != NULL);
    assert (peaks->n == 0 || pmIsPeak(peaks->data[0]));

    if (footprints->n == 0) {
        if (peaks->n > 0) {
            return psError(PS_ERR_BAD_PARAMETER_SIZE, true, "Your list of footprints is empty");
        }
        return PS_ERR_NONE;
    }
    /*
     * Create an image filled with the object IDs, and use it to assign pmPeaks to the
     * objects
     */
    psImage *ids = pmSetFootprintArrayIDs(footprints, true);
    assert (ids != NULL);
    assert (ids->type.type == PS_TYPE_S32);
    int const y0 = ids->y0;
    int const x0 = ids->x0;
    int const numRows = ids->numRows;
    int const numCols = ids->numCols;

    for (int i = 0; i < peaks->n; ++i) {
        pmPeak *peak = peaks->data[i];
        int const x = peak->x - x0;
        int const y = peak->y - y0;

        assert (x >= 0 && x < numCols && y >= 0 && y < numRows);
        int id = ids->data.S32[y][x - x0];

        if (id == 0) {                  // peak isn't in a footprint, so make one for it
            pmFootprint *nfp = pmFootprintAlloc(1, ids);
            pmFootprintAddSpan(nfp, y, x, x);
            psArrayAdd(footprints, 1, nfp);
            psFree(nfp);
            id = footprints->n;
        }

        assert (id >= 1 && id <= footprints->n);
        pmFootprint *fp = footprints->data[id - 1];
        psArrayAdd(fp->peaks, 5, peak);
    }

    psFree(ids);
    //
    // Make sure that peaks within each footprint are sorted and unique
    //
    for (int i = 0; i < footprints->n; ++i) {
        pmFootprint *fp = footprints->data[i];
        fp->peaks = psArraySort(fp->peaks, pmPeakSortBySN);

        for (int j = 1; j < fp->peaks->n; ++j) { // check for duplicates
            if (fp->peaks->data[j] == fp->peaks->data[j-1]) {
                (void)psArrayRemoveIndex(fp->peaks, j);
                j--;                    // we moved everything down one
            }
        }
    }

    return PS_ERR_NONE;
}

/************************************************************************************************************/
 /*
  * Examine the peaks in a pmFootprint, and throw away the ones that are not sufficiently
  * isolated.  More precisely, for each peak find the highest coll that you'd have to traverse
  * to reach a still higher peak --- and if that coll's more than nsigma DN below your
  * starting point, discard the peak.
  */
psErrorCode pmFootprintCullPeaks(psImage const *img, // the image wherein lives the footprint
                                 psImage const *weight, // corresponding variance image
                                 pmFootprint *fp, // Footprint containing mortal peaks
                                 float const nsigma_delta, // how many sigma above local background a peak
                                  // needs to be to survive
                                 float const min_threshold) { // minimum permitted coll height
    assert (img != NULL);
    assert (img->type.type == PS_TYPE_F32);
    assert (weight != NULL);
    assert (weight->type.type == PS_TYPE_F32);
    assert (img->y0 == weight->y0 && img->x0 == weight->x0);
    assert (fp != NULL);

    if (fp->peaks == NULL || fp->peaks->n == 0) { // nothing to do
        return PS_ERR_NONE;
    }

    psRegion subRegion;                 // desired subregion; 1 larger than bounding box (grr)
    subRegion.x0 = fp->bbox.x0;
    subRegion.x1 = fp->bbox.x1 + 1;
    subRegion.y0 = fp->bbox.y0;
    subRegion.y1 = fp->bbox.y1 + 1;
    psImage const *subImg = psImageSubset((psImage *)img, subRegion);
    psImage const *subWt = psImageSubset((psImage *)weight, subRegion);
    assert (subImg != NULL && subWt != NULL);
    //
    // We need a psArray of peaks brighter than the current peak.  We'll fake this
    // by reusing the fp->peaks but lying about n.
    //
    // We do this for efficiency (otherwise I'd need two peaks lists), and we are
    // rather too chummy with psArray in consequence.  But it works.
    //
    psArray *brightPeaks = psArrayAlloc(0);
    psFree(brightPeaks->data);
    brightPeaks->data = psMemIncrRefCounter(fp->peaks->data);// use the data from fp->peaks
    //
    // The brightest peak is always safe; go through other peaks trying to cull them
    //
    for (int i = 1; i < fp->peaks->n; ++i) { // n.b. fp->peaks->n can change within the loop
        pmPeak const *peak = fp->peaks->data[i];
        int x = peak->x - subImg->x0;
        int y = peak->y - subImg->y0;
        //
        // Find the level nsigma below the peak that must separate the peak
        // from any of its friends
        //
        assert (x >= 0 && x < subImg->numCols && y >= 0 && y < subImg->numRows);
        float const stdev = std::sqrt(subWt->data.F32[y][x]);
        float threshold = subImg->data.F32[y][x] - nsigma_delta*stdev;
        if (lsst::utils::isnan(threshold) || threshold < min_threshold) {
#if 1                                   // min_threshold is assumed to be below the detection threshold,
                                        // so all the peaks are pmFootprint, and this isn't the brightest
            (void)psArrayRemoveIndex(fp->peaks, i);
            i--;                        // we moved everything down one
            continue;
#else
#error n.b. We will be running LOTS of checks at this threshold, so only find the footprint once
            threshold = min_threshold;
#endif
        }
        if (threshold > subImg->data.F32[y][x]) {
            threshold = subImg->data.F32[y][x] - 10*FLT_EPSILON;
        }

        int const peak_id = 1;          // the ID for the peak of interest
        brightPeaks->n = i;             // only stop at a peak brighter than we are
        pmFootprint *peakFootprint = pmFindFootprintAtPoint(subImg, threshold, brightPeaks, peak->y, peak->x);
        brightPeaks->n = 0;             // don't double free
        psImage *idImg = pmSetFootprintID(peakFootprint, peak_id);
        psFree(peakFootprint);

        int j;
        for (j = 0; j < i; ++j) {
            pmPeak const *peak2 = fp->peaks->data[j];
            int x2 = peak2->x - subImg->x0;
            int y2 = peak2->y - subImg->y0;
            int const peak2_id = idImg->data.S32[y2][x2]; // the ID for some other peak

            if (peak2_id == peak_id) {  // There's a brighter peak within the footprint above
                ;                       // threshold; so cull our initial peak
                (void)psArrayRemoveIndex(fp->peaks, i);
                i--;                    // we moved everything down one
                break;
            }
        }
        if (j == i) {
            ++j;
        }

        psFree(idImg);
    }

    brightPeaks->n = 0;
    psFree(brightPeaks);
    psFree((psImage *)subImg);
    psFree((psImage *)subWt);

    return PS_ERR_NONE;
}

/*
 * Cull an entire psArray of pmFootprints
 */
psErrorCode
pmFootprintArrayCullPeaks(psImage const *img, // the image wherein lives the footprint
                          psImage const *weight,        // corresponding variance image
                          psArray *footprints, // array of pmFootprints
                          float const nsigma_delta, // how many sigma above local background a peak
                                  // needs to be to survive
                          float const min_threshold) { // minimum permitted coll height
    for (int i = 0; i < footprints->n; ++i) {
        pmFootprint *fp = footprints->data[i];
        if (pmFootprintCullPeaks(img, weight, fp, nsigma_delta, min_threshold) != PS_ERR_NONE) {
            return psError(PS_ERR_UNKNOWN, false, "Culling pmFootprint %d", fp->id);
        }
    }

    return PS_ERR_NONE;
}

/************************************************************************************************************/
/*
 * Extract the peaks in a psArray of pmFootprints, returning a psArray of pmPeaks
 */
psArray *pmFootprintArrayToPeaks(psArray const *footprints) {
    assert(footprints != NULL);
    assert(footprints->n == 0 || pmIsFootprint(footprints->data[0]));

    int npeak = 0;
    for (int i = 0; i < footprints->n; ++i) {
        pmFootprint const *fp = footprints->data[i];
        npeak += fp->peaks->n;
    }

    psArray *peaks = psArrayAllocEmpty(npeak);

    for (int i = 0; i < footprints->n; ++i) {
        pmFootprint const *fp = footprints->data[i];
        for (int j = 0; j < fp->peaks->n; ++j) {
            psArrayAdd(peaks, 1, fp->peaks->data[j]);
        }
    }

    return peaks;
}
#endif

/************************************************************************************************************/
//
// Explicit instantiations
// \cond
//
//
template
void Footprint::intersectMask(
    image::Mask<image::MaskPixel> const& mask,
    image::MaskPixel bitMask);

template
Footprint::Ptr footprintAndMask(
    Footprint::Ptr const& foot,
    image::Mask<image::MaskPixel>::Ptr const& mask,
    image::MaskPixel bitMask);

template
image::MaskPixel setMaskFromFootprintList(
    image::Mask<image::MaskPixel> *mask,
    CONST_PTR(std::vector<Footprint::Ptr>) const& footprints,
    image::MaskPixel const bitmask);
template
image::MaskPixel setMaskFromFootprintList(
    image::Mask<image::MaskPixel> *mask,
    std::vector<Footprint::Ptr> const& footprints,
    image::MaskPixel const bitmask);
template image::MaskPixel setMaskFromFootprint(
    image::Mask<image::MaskPixel> *mask,
    Footprint const& foot, image::MaskPixel const bitmask);
template image::MaskPixel clearMaskFromFootprint(
    image::Mask<image::MaskPixel> *mask,
    Footprint const& foot, image::MaskPixel const bitmask);

#define INSTANTIATE_NUMERIC(TYPE) \
template \
TYPE setImageFromFootprint(image::Image<TYPE> *image,        \
                                      Footprint const& footprint, \
                                      TYPE const value);                \
template \
TYPE setImageFromFootprintList(image::Image<TYPE> *image, \
                                          std::vector<Footprint::Ptr> const& footprints, \
                                          TYPE const value); \
template \
TYPE setImageFromFootprintList(image::Image<TYPE> *image, \
                                          CONST_PTR(std::vector<Footprint::Ptr>) footprints, \
                                          TYPE const value); \
template \
void copyWithinFootprint(Footprint const&,                          \
                         PTR(lsst::afw::image::Image<TYPE>) const,  \
                         PTR(lsst::afw::image::Image<TYPE>));       \
template \
void copyWithinFootprint(Footprint const&,                          \
                         PTR(lsst::afw::image::MaskedImage<TYPE>) const,  \
                         PTR(lsst::afw::image::MaskedImage<TYPE>));	\
template								\
 void Footprint::clipToNonzero(lsst::afw::image::Image<TYPE> const&);	\


INSTANTIATE_NUMERIC(float);
INSTANTIATE_NUMERIC(double);
// There's no reason these shouldn't have setImageFromFootprint(), etc, instantiated
INSTANTIATE_NUMERIC(boost::uint16_t);
INSTANTIATE_NUMERIC(int);
INSTANTIATE_NUMERIC(boost::uint64_t);


#define INSTANTIATE_MASK(PIXEL)                                         \
template                                                                \
void Footprint::insertIntoImage(                                        \
    lsst::afw::image::Image<PIXEL>& idImage,                            \
    boost::uint64_t const id,                                           \
    geom::Box2I const& region=geom::Box2I()                             \
    ) const;                                                            \
template                                                                \
void Footprint::insertIntoImage(                                        \
    lsst::afw::image::Image<PIXEL>& idImage,                            \
    boost::uint64_t const id,                                           \
    bool const overwriteId, long const idMask,                          \
    std::set<boost::uint64_t> *oldIds,                                  \
    geom::Box2I const& region=geom::Box2I()                             \
    ) const;                                                            \

INSTANTIATE_MASK(boost::uint16_t);
INSTANTIATE_MASK(int);
INSTANTIATE_MASK(boost::uint64_t);


}}}
// \endcond

//  LocalWords:  SpanList
