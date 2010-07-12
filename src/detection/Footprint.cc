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
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelFunctions.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/utils/ieee.h"

namespace afwMath = lsst::afw::math;
namespace afwDetect = lsst::afw::detection;
namespace afwImage = lsst::afw::image;

/******************************************************************************/
/**
 * \brief Factory method for creating Threshold objects
 *
 * \return desired Threshold
 */
afwDetect::Threshold afwDetect::createThreshold(
    float const value,                  ///< value of threshold
    std::string const typeStr,          ///<  string representation of a ThresholdType. This parameter is 
                                        ///< optional. Allowed values are: "variance", "value", "stdev"
    bool const polarity                 ///< If true detect positive objects, false for negative
) {
    Threshold::ThresholdType thresholdType;
    if (typeStr.compare("value") == 0) {
        thresholdType = Threshold::VALUE;           
    } else if (typeStr.compare("stdev") == 0) {
        thresholdType = Threshold::STDEV;
    } else if (typeStr.compare("variance") == 0) {
        thresholdType = Threshold::VARIANCE;
    } else {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Unsopported Threshold type: %s") % typeStr).str());
    }    

    return Threshold(value, thresholdType, polarity);
}


/******************************************************************************/
/**
 * Return a string-representation of a Span
 */
std::string afwDetect::Span::toString() const {
    return (boost::format("%d: %d..%d") % _y % _x0 % _x1).str();
}

namespace {
/*
 * Compare two Span%s by y, then x0, then x1
 *
 * A utility functor passed to sort
 */
    struct compareSpanByYX : public std::binary_function<afwDetect::Span::ConstPtr,
                                                         afwDetect::Span::ConstPtr, bool> {
        int operator()(afwDetect::Span::ConstPtr a, afwDetect::Span::ConstPtr b) {
            if (a->getY() < b->getY()) {
                return true;
            } else if (a->getY() == b->getY()) {
                if (a->getX0() < b->getX0()) {
                    return true;
                } else if (a->getX0() == b->getX0()) {
                    if (a->getX1() < b->getX1()) {
                        return true;
                    }
                }
            }
            return false;
        }
    };
}

/************************************************************************************************************/
/// Counter for Footprint IDs
int afwDetect::Footprint::id = 0;
/**
 * Create a Footprint
 *
 * \throws lsst::pex::exceptions::InvalidParameterException in nspan is < 0
 */
afwDetect::Footprint::Footprint(int nspan,         //!< initial number of Span%s in this Footprint
                                afwImage::BBox const region) //!< Bounding box of MaskedImage footprint
    : lsst::daf::data::LsstBase(typeid(this)),
      _fid(++id),
      _npix(0),
      _bbox(afwImage::BBox()),
      _region(region),
      _normalized(false) {
    if (nspan < 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          (boost::format("Number of spans requested is -ve: %d") % nspan).str());
    }
    _npix = 0;
}

/**
 * Create a rectangular Footprint
 */
afwDetect::Footprint::Footprint(afwImage::BBox const& bbox, //!< The bounding box defining the rectangle
                                afwImage::BBox const region) //!< Bounding box of MaskedImage footprint
    : lsst::daf::data::LsstBase(typeid(this)),
      _fid(++id),
      _npix(0),
      _bbox(afwImage::BBox()),
      _region(region),
      _normalized(false) {
    int const x0 = bbox.getX0();
    int const y0 = bbox.getY0();
    int const x1 = bbox.getX1();
    int const y1 = bbox.getY1();

    for (int i = y0; i <= y1; i++) {
        addSpan(i, x0, x1);
    }
}

/**
 * Create a circular Footprint
 */
afwDetect::Footprint::Footprint(afwImage::BCircle const& circle, //!< The center and radius of the circle
                                afwImage::BBox const region)  //!< Bounding box of MaskedImage footprint
    : lsst::daf::data::LsstBase(typeid(this)),
      _fid(++id),
      _npix(0),
      _bbox(afwImage::BBox()),
      _region(region),
      _normalized(false) {
    int const xc = circle.getCenter().getX(); // x-centre
    int const yc = circle.getCenter().getY(); // y-centre
    int const r2 = static_cast<int>(circle.getRadius()*circle.getRadius() + 0.5); // rounded radius^2
    int const r = static_cast<int>(std::sqrt(static_cast<double>(r2))); // truncated radius; r*r <= r2

    for (int i = -r; i <= r; i++) {
        int hlen = static_cast<int>(std::sqrt(static_cast<double>(r2 - i*i)));
        addSpan(yc + i, xc - hlen, xc + hlen);
    }
}

/**
 * Destroy a Footprint
 */
afwDetect::Footprint::~Footprint() {
}

/**
 * Normalise a Footprint, soring spans and setting the BBox
 */
void afwDetect::Footprint::normalize() {
    if (!_normalized) {
        assert(!_spans.empty());

        //
        // Check that the spans are sorted, and (more importantly) that each pixel appears
        // in only one span
        //
        sort(_spans.begin(), _spans.end(), compareSpanByYX());

        afwDetect::Footprint::SpanList::iterator ptr = _spans.begin(), end = _spans.end();
        
        afwDetect::Span *lspan = ptr->get();  // Left span
        int y = lspan->_y;
        int x1 = lspan->_x1;
        ++ptr;

        for (; ptr != end; ++ptr) {
            afwDetect::Span *rspan = ptr->get(); // Right span
            if (rspan->_y == y) {
                if (rspan->_x0 <= x1 + 1) { // Spans overlap or touch
                    if (rspan->_x1 > x1) {  // right span extends left span
                        x1 = lspan->_x1 = rspan->_x1;
                    }

                    ptr = _spans.erase(ptr);
                    end = _spans.end();   // delete the right span
                    if (ptr == end) {
                        break;
                    }
                    
                    --ptr;
                    continue;
                }
            }

            y = rspan->_y;
            x1 = rspan->_x1;
            
            lspan = rspan;
        }

        //_peaks = psArraySort(fp->peaks, pmPeakSortBySN);
        setNpix();
        setBBox();
        _normalized = true;
    }
}

/**
 * Add a Span to a footprint, returning a reference to the new Span.
 */
afwDetect::Span const& afwDetect::Footprint::addSpan(int const y, //!< row value
                                                     int const x0, //!< starting column
                                                     int const x1 //!< ending column
                                                    ) {
    if (x1 < x0) {
        return this->addSpan(y, x1, x0);
    }

    afwDetect::Span::Ptr sp(new afwDetect::Span(y, x0, x1));
    _spans.push_back(sp);

    _npix += x1 - x0 + 1;
    _normalized = false;

    _bbox.grow(afwImage::PointI(x0, y));
    _bbox.grow(afwImage::PointI(x1, y));

    return *sp.get();
}
/**
 * Add a Span to a Footprint returning a reference to the new Span
 */
const afwDetect::Span& afwDetect::Footprint::addSpan(afwDetect::Span const& span ///< new Span being added
                              ) {
    afwDetect::Span::Ptr sp(new afwDetect::Span(span));

    _spans.push_back(sp);

    _npix += span._x1 - span._x0 + 1;
    _normalized = false;

    _bbox.grow(afwImage::PointI(span._x0, span._y));
    _bbox.grow(afwImage::PointI(span._x1, span._y));

    return *sp;
}

/**
 * Add a Span to a Footprint returning a reference to the new Span
 */
const afwDetect::Span& afwDetect::Footprint::addSpan(afwDetect::Span const& span, ///< new Span being added
                                                     int dx,              ///< Add dx to span's x coords
                                                     int dy               ///< Add dy to span's y coords
                              ) {
    return addSpan(span._y + dy, span._x0 + dx, span._x1 + dx);
}
/**
 * Shift a Footprint by <tt>(dx, dy)</tt>
 */
void afwDetect::Footprint::shift(int dx, //!< How much to move footprint in column direction
                                 int dy  //!< How much to move in row direction
                      ) {
    for (Footprint::SpanList::iterator siter = _spans.begin(); siter != _spans.end(); ++siter){
        afwDetect::Span::Ptr span = *siter;

        span->_y += dy;
        span->_x0 += dx;
        span->_x1 += dx;
    }

    _bbox.shift(dx, dy);
}

/**
 * Tell \c this to calculate its bounding box
 */
void afwDetect::Footprint::setBBox() {
    if (_spans.size() == 0) {
        return;
    }

    SpanList::const_iterator spi;
    spi = _spans.begin();
    const Span::Ptr sp = *spi;
    int x0 = sp->_x0;
    int x1 = sp->_x1;
    int y0 = sp->_y;
    int y1 = sp->_y;

    for (; spi != _spans.end(); spi++) {
        afwDetect::Span::ConstPtr span = *spi;
        if (span->_x0 < x0) x0 = span->_x0;
        if (span->_x1 > x1) x1 = span->_x1;
        if (span->_y < y0) y0 = span->_y;
        if (span->_y > y1) y1 = span->_y;
    }

    _bbox = afwImage::BBox(afwImage::PointI(x0, y0), afwImage::PointI(x1, y1));
}

/**
 * Tell \c this to count its pixels
 */
int afwDetect::Footprint::setNpix() {
    _npix = 0;
    for (Footprint::SpanList::const_iterator spi = _spans.begin(); spi != _spans.end(); spi++) {
        afwDetect::Span::Ptr const sp = *spi;
        _npix += sp->_x1 - sp->_x0 + 1;
    }

    return _npix;
}

/**
 * Set the pixels in idImage which are in Footprint by adding the specified value to the Image
 */
void afwDetect::Footprint::insertIntoImage(
                  afwImage::Image<boost::uint16_t>& idImage, //!< Image to contain the footprint
                  int const id, //!< Add id to idImage for pixels in the Footprint
                  afwImage::BBox const& region //!< Footprint's region (default: getRegion())
                                          ) const {
    int const width =  (region ? region : _region).getWidth();
    int const height = (region ? region : _region).getHeight();
    int const x0 =     (region ? region : _region).getX0();
    int const y0 =     (region ? region : _region).getY0();

    if (width != idImage.getWidth() || height != idImage.getHeight()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          (boost::format("Image of size (%dx%d) doesn't match"
                                         "Footprint's host Image of size (%dx%d)") %
                           idImage.getWidth() % idImage.getHeight() % width % height).str());
    }

    for (Footprint::SpanList::const_iterator spi = _spans.begin(); spi != _spans.end(); ++spi) {
        afwDetect::Span::Ptr const span = *spi;

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

        for (afwImage::Image<boost::uint16_t>::x_iterator ptr = idImage.x_at(sx0, sy0),
                 end = ptr + swidth; ptr != end; ++ptr) {
            *ptr += id;
        }
    }
}

/************************************************************************************************************/
/**
 * \brief Return a Footprint that's the intersection of a Footprint with a Mask
 *
 * The resulting Footprint contains only pixels for which (mask & bitMask) != 0;
 * it may have disjoint pieces
 *
 * \note This isn't a member of Footprint as Footprint isn't templated over MaskT
 *
 * \returns Returns the new Footprint
 */
template<typename MaskT>
afwDetect::Footprint::Ptr afwDetect::footprintAndMask(
        Footprint::Ptr const&,                                   ///< The initial Footprint
        typename lsst::afw::image::Mask<MaskT>::Ptr const&,      ///< The mask to & with foot
        MaskT                                                    ///< Only consider these bits
                                                     )
{
    Footprint::Ptr out(new afwDetect::Footprint());

    return out;
}

/************************************************************************************************************/
/**
 * \brief OR bitmask into all the Mask's pixels which are in the Footprint
 *
 * \return bitmask
 */
template<typename MaskT>
MaskT afwDetect::setMaskFromFootprint(afwImage::Mask<MaskT> *mask,              ///< Mask to set
                                      Footprint const& foot,      ///< Footprint specifying desired pixels
                                      MaskT const bitmask                    ///< Bitmask to OR into mask
                                     ) {

    int const width = static_cast<int>(mask->getWidth());
    int const height = static_cast<int>(mask->getHeight());

    for (afwDetect::Footprint::SpanList::const_iterator siter = foot.getSpans().begin();
         siter != foot.getSpans().end(); siter++) {
        afwDetect::Span::Ptr const span = *siter;
        int const y = span->getY() - mask->getY0();
        if (y < 0 || y >= height) {
            continue;
        }

        int x0 = span->getX0() - mask->getX0();
        int x1 = span->getX1() - mask->getX0();
        x0 = (x0 < 0) ? 0 : (x0 >= width ? width - 1 : x0);
        x1 = (x1 < 0) ? 0 : (x1 >= width ? width - 1 : x1);

        for (typename afwImage::Image<MaskT>::x_iterator ptr = mask->x_at(x0, y),
                 end = mask->x_at(x1 + 1, y); ptr != end; ++ptr) {
            *ptr |= bitmask;
        }
    }

    return bitmask;
}

/************************************************************************************************************/
/**
 * \brief OR bitmask into all the Mask's pixels which are in the set of Footprint%s
 *
 * \return bitmask
 */
template<typename MaskT>
MaskT afwDetect::setMaskFromFootprintList(
        afwImage::Mask<MaskT> *mask,                        ///< Mask to set
        std::vector<Footprint::Ptr> const& footprints,  ///< Footprint list specifying desired pixels
        MaskT const bitmask                             ///< Bitmask to OR into mask
                                               ) {
    for (std::vector<afwDetect::Footprint::Ptr>::const_iterator fiter = footprints.begin();
         fiter != footprints.end(); ++fiter) {
        (void)setMaskFromFootprint(mask, **fiter, bitmask);
    }

    return bitmask;
}

/************************************************************************************************************/
namespace {
template<typename ImageT>
class SetFootprint : public afwDetect::FootprintFunctor<ImageT> {
public:
    SetFootprint(ImageT const& image,
                 typename ImageT::Pixel value) :
        afwDetect::FootprintFunctor<ImageT>(image), _value(value) {} 


    void operator()(typename ImageT::xy_locator loc, int, int) {
        *loc = _value;
    }
private:
    typename ImageT::Pixel _value;
};
}

/**
 * \brief Set all image pixels in a Footprint to a given value
 *
 * \return value
 */
template<typename ImageT>
typename ImageT::Pixel afwDetect::setImageFromFootprint(
        ImageT *image,                    ///< image to set
        afwDetect::Footprint const& foot, ///< Footprint defining desired pixels
        typename ImageT::Pixel const value ///< value to set Image to
                                                       ) {
    SetFootprint<ImageT> setit(*image, value);
    setit.apply(foot);

    return value;
}

/**
 * \brief Set all image pixels in a set of Footprint%s to a given value
 *
 * \return value
 */
template<typename ImageT>
typename ImageT::Pixel afwDetect::setImageFromFootprintList(
        ImageT *image,                                  ///< image to set
        std::vector<Footprint::Ptr> const& footprints,  ///< Footprint list specifying desired pixels
        typename ImageT::Pixel const value              ///< value to set Image to
                                                           ) {
    SetFootprint<ImageT> setit(*image, value);
    for (std::vector<afwDetect::Footprint::Ptr>::const_iterator fiter = footprints.begin(),
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
static void set_footprint_id(typename afwImage::Image<IDPixelT>::Ptr idImage,   // the image to set
                             afwDetect::Footprint const& foot, // the footprint to insert
                             int const id,                     // the desired ID
                             int dx=0, int dy=0                // Add these to all x/y in the Footprint
                            ) {
    for (afwDetect::Footprint::SpanList::const_iterator siter = foot.getSpans().begin();
                                                        siter != foot.getSpans().end(); siter++) {
        afwDetect::Span::Ptr const span = *siter;
        for (typename afwImage::Image<IDPixelT>::x_iterator ptr =
                 idImage->x_at(span->getX0() + dx, span->getY() + dy),
                 end = ptr + span->getWidth(); ptr != end; ++ptr) {
            *ptr = id;
        }
    }
}

template <typename IDPixelT>
static void
set_footprint_array_ids(typename afwImage::Image<IDPixelT>::Ptr idImage, // the image to set
                        std::vector<afwDetect::Footprint::Ptr> const& footprints, // the footprints to insert
                        bool const relativeIDs) { // show IDs starting at 0, not Footprint->id
    int id = 0;                         // first index will be 1

    for (std::vector<afwDetect::Footprint::Ptr>::const_iterator fiter = footprints.begin();
         fiter != footprints.end(); ++fiter) {
        afwDetect::Footprint::Ptr const foot = *fiter;

        if (relativeIDs) {
            id++;
        } else {
            id = foot->getId();
        }

        set_footprint_id<IDPixelT>(idImage, *foot, id);
    }
}

template void set_footprint_array_ids<int>(afwImage::Image<int>::Ptr idImage,
                                           std::vector<afwDetect::Footprint::Ptr> const& footprints,
                                           bool const relativeIDs);

/************************************************************************************************************/
/*
 * Create an image from a Footprint's bounding box
 */
template <typename IDImageT>
static typename afwImage::Image<IDImageT>::Ptr makeImageFromBBox(afwImage::BBox const bbox) {
    typename afwImage::Image<IDImageT>::Ptr idImage(new afwImage::Image<IDImageT>(bbox.getDimensions()));
    idImage->setXY0(bbox.getLLC());

    return idImage;
}

/************************************************************************************************************/
/*
 * Set an image to the value of footprint's ID wherever they may fall
 */
template <typename IDImageT>
typename boost::shared_ptr<afwImage::Image<IDImageT> > setFootprintArrayIDs(
        std::vector<afwDetect::Footprint::Ptr> const& footprints, // the footprints to insert
        bool const relativeIDs                          // show IDs starting at 1, not pmFootprint->id
                                               ) {
    std::vector<afwDetect::Footprint::Ptr>::const_iterator fiter = footprints.begin();
    if (fiter == footprints.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          "You didn't provide any footprints");
    }
    afwDetect::Footprint::Ptr const foot = *fiter;

    typename afwImage::Image<IDImageT>::Ptr idImage = makeImageFromBBox<IDImageT>(foot->getRegion());
    *idImage = 0;
    /*
     * do the work
     */
    set_footprint_array_ids<IDImageT>(idImage, footprints, relativeIDs);

    return idImage;
}

template afwImage::Image<int>::Ptr setFootprintArrayIDs(
                                        std::vector<afwDetect::Footprint::Ptr> const& footprints,
                                        bool const relativeIDs);
/*
 * Set an image to the value of Footprint's ID wherever it may fall
 */
template <typename IDImageT>
typename boost::shared_ptr<afwImage::Image<IDImageT> > setFootprintID(
                                          afwDetect::Footprint::Ptr const& foot, // the Footprint to insert
                                          int const id // the desired ID
                                                                     ) {
    typename afwImage::Image<IDImageT>::Ptr idImage = makeImageFromBBox<IDImageT>(foot->getBBox());
    *idImage = 0;
    /*
     * do the work
     */
    set_footprint_id<IDImageT>(idImage, *foot, id);

    return idImage;
}

template afwImage::Image<int>::Ptr setFootprintID(afwDetect::Footprint::Ptr const& foot, int const id);

/************************************************************************************************************/
/*
 * Grow a Footprint isotropically by r pixels, returning a new Footprint
 *
 * N.b. this is slow, as it uses a convolution with a disk
 */
namespace {
afwDetect::Footprint::Ptr growFootprintSlow(
        afwDetect::Footprint const& foot, //!< The Footprint to grow
        int ngrow                              //!< how much to grow foot
                                                 ) {
    if (ngrow < 0) {
        ngrow = 0;                      // ngrow == 0 => no grow
    }

    if (foot.getNpix() == 0) {          // an empty Footprint
        return afwDetect::Footprint::Ptr(new afwDetect::Footprint);
    }

    /*
     * We'll insert the footprints into an image, then convolve with a disk,
     * then extract a footprint from the result --- this is magically what we want.
     */
    afwImage::BBox bbox = foot.getBBox();
    bbox.grow(afwImage::PointI(bbox.getX0() - 2*ngrow - 1, bbox.getY0() - 2*ngrow - 1));
    bbox.grow(afwImage::PointI(bbox.getX1() + 2*ngrow + 1, bbox.getY1() + 2*ngrow + 1));
    afwImage::Image<int>::Ptr idImage = makeImageFromBBox<int>(bbox);
    *idImage = 0;
    idImage->setXY0(afwImage::PointI(0, 0));

    set_footprint_id<int>(idImage, foot, 1, -bbox.getX0(), -bbox.getY0());

    afwImage::Image<double>::Ptr circle_im(new afwImage::Image<double>(2*ngrow + 1, 2*ngrow + 1));
    *circle_im = 0;
    for (int r = -ngrow; r <= ngrow; ++r) {
        afwImage::Image<double>::x_iterator row = circle_im->x_at(0, r + ngrow);
        for (int c = -ngrow; c <= ngrow; ++c, ++row) {
            if (r*r + c*c <= ngrow*ngrow) {
                *row = 8;
            }
        }
    }

    afwMath::FixedKernel::Ptr circle(new afwMath::FixedKernel(*circle_im));
    // Here's the actual grow step
    afwImage::MaskedImage<int>::Ptr convolvedImage(new afwImage::MaskedImage<int>(idImage->getDimensions()));
    afwMath::convolve(*convolvedImage->getImage(), *idImage, *circle, false);

    afwDetect::FootprintSet<int>::Ptr
        grownList(new afwDetect::FootprintSet<int>(*convolvedImage, 0.5, "", 1));

    assert (grownList->getFootprints().size() > 0);
    afwDetect::Footprint::Ptr grown = *grownList->getFootprints().begin();
    //
    // Fix the coordinate system to be that of foot
    //
    grown->shift(bbox.getX0(), bbox.getY0());
    grown->setRegion(foot.getRegion());

    return grown;
}
}

/************************************************************************************************************/
/**
 * Grow a Footprint by r pixels, returning a new Footprint
 */
afwDetect::Footprint::Ptr afwDetect::growFootprint(
        afwDetect::Footprint const& foot,      //!< The Footprint to grow
        int ngrow,                             //!< how much to grow foot
        bool isotropic                         //!< Grow isotropically (as opposed to a Manhattan metric)
                                               //!< @note Isotropic grows are significantly slower
                                                 ) {

    if (isotropic) {
        return growFootprintSlow(foot, ngrow);
    }

    if (ngrow < 0) {
        ngrow = 0;                      // ngrow == 0 => no grow
    }
    /*
     * We'll insert the footprints into an image, set all the pixels to the Manhatten distance from the
     * nearest set pixel, then extract a footprint from the result
     *
     * Cf. http://ostermiller.org/dilate_and_erode.html
     */
    afwImage::BBox bbox = foot.getBBox();
    bbox.grow(afwImage::PointI(bbox.getX0() - ngrow - 1, bbox.getY0() - ngrow - 1));
    bbox.grow(afwImage::PointI(bbox.getX1() + ngrow + 1, bbox.getY1() + ngrow + 1));
    afwImage::Image<int>::Ptr idImage = makeImageFromBBox<int>(bbox);
    *idImage = 0;
    idImage->setXY0(afwImage::PointI(0, 0));
    
    // Set all the pixels in the footprint to 1
    set_footprint_id<int>(idImage, foot, 1, -bbox.getX0(), -bbox.getY0()); 
    //
    // Set the idImage to the Manhattan distance from the nearest set pixel
    //
    int const height = idImage->getHeight();
    int const width = idImage->getWidth();

    // traverse from bottom left to top right
    for (int y = 0; y != height; ++y) {
        afwImage::Image<int>::xy_locator im = idImage->xy_at(0, y);

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
        afwImage::Image<int>::xy_locator im = idImage->xy_at(width - 1, y);
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

    afwImage::MaskedImage<int>::Ptr midImage(new afwImage::MaskedImage<int>(idImage));
    // XXX Why do I need a -ve threshold when parity == false? I'm looking for pixels below ngrow
    FootprintSet<int>::Ptr grownList(new FootprintSet<int>(*midImage,
                                                           Threshold(-ngrow, afwDetect::Threshold::VALUE,
                                                                     false)));
    assert (grownList->getFootprints().size() > 0);
    afwDetect::Footprint::Ptr grown = *grownList->getFootprints().begin();
    //
    // Fix the coordinate system to be that of foot
    //
    grown->shift(bbox.getX0(), bbox.getY0());
    grown->setRegion(foot.getRegion());

    return grown;
}

afwDetect::Footprint::Ptr afwDetect::growFootprint(Footprint::Ptr const& foot, int ngrow, bool isotropic) {
    return growFootprint(*foot, ngrow, isotropic);
}

/************************************************************************************************************/
/**
 * Return a list of BBox%s, whose union contains exactly the pixels in foot, neither more nor less
 *
 * Useful in generating sets of meas::algorithms::Defects for the ISR
 */
std::vector<afwImage::BBox> afwDetect::footprintToBBoxList(afwDetect::Footprint const& foot
                                                       ) {
    typedef boost::uint16_t ImageT;
    afwImage::Image<ImageT>::Ptr idImage(new afwImage::Image<ImageT>(foot.getBBox().getDimensions()));
    *idImage = 0;
    int const height = idImage->getHeight();

    foot.insertIntoImage(*idImage, 1, foot.getBBox());

    std::vector<afwImage::BBox> bboxes;
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
        afwImage::BBox bbox;            // our next BBox
        for (int y = y0; y != height; ++y) {
            // Look for a set pixel in this row
            afwImage::Image<ImageT>::x_iterator begin = idImage->row_begin(y), end = idImage->row_end(y);
            afwImage::Image<ImageT>::x_iterator first = std::find(begin, end, 1);

            if (first != end) {                     // A pixel is set in this row
                afwImage::Image<ImageT>::x_iterator last = std::find(first, end, 0) - 1;
                int const x0 = first - begin;
                int const x1 = last  - begin;

                std::fill(first, last + 1, 0);       // clear pixels; we don't want to see them again

                bbox.grow(afwImage::PointI(x0, y));     // the LLC
                bbox.grow(afwImage::PointI(x1, y));     // the LRC; initial guess for URC
                
                // we found at least one pixel so extend the BBox upwards
                for (++y; y != height; ++y) {
                    if (std::find(idImage->at(x0, y), idImage->at(x1 + 1, y), 0) != idImage->at(x1 + 1, y)) {
                        break;  // some pixels weren't set, so the BBox stops here, (actually in previous row)
                    }
                    std::fill(idImage->at(x0, y), idImage->at(x1 + 1, y), 0);
                    
                    bbox.grow(afwImage::PointI(x1, y)); // the new URC
                }

                bbox.shift(foot.getBBox().getX0(), foot.getBBox().getY0());
                bboxes.push_back(bbox);
            } else {
                y0 = y + 1;
            }
            break;
        }
    }

    return bboxes;
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
    for (int i = 0; i <= r; i++) {
        for (int j = 0; j <= r; j++) {
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
        for (int i = 0; i < old->n; i++) {
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

    for (int i = 0; i < peaks->n; i++) {
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
    for (int i = 0; i < footprints->n; i++) {
        pmFootprint *fp = footprints->data[i];
        fp->peaks = psArraySort(fp->peaks, pmPeakSortBySN);

        for (int j = 1; j < fp->peaks->n; j++) { // check for duplicates
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
    for (int i = 1; i < fp->peaks->n; i++) { // n.b. fp->peaks->n can change within the loop
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
        for (j = 0; j < i; j++) {
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
            j++;
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
    for (int i = 0; i < footprints->n; i++) {
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
    for (int i = 0; i < footprints->n; i++) {
        pmFootprint const *fp = footprints->data[i];
        npeak += fp->peaks->n;
    }
    
    psArray *peaks = psArrayAllocEmpty(npeak);

    for (int i = 0; i < footprints->n; i++) {
        pmFootprint const *fp = footprints->data[i];
        for (int j = 0; j < fp->peaks->n; j++) {
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
template
afwDetect::Footprint::Ptr afwDetect::footprintAndMask(afwDetect::Footprint::Ptr const& foot,
                                                      afwImage::Mask<afwImage::MaskPixel>::Ptr const& mask,
                                                      afwImage::MaskPixel bitMask);

template
afwImage::MaskPixel afwDetect::setMaskFromFootprintList(afwImage::Mask<afwImage::MaskPixel> *mask,
                                                     std::vector<afwDetect::Footprint::Ptr> const& footprints,
                                                     afwImage::MaskPixel const bitmask);
template
afwImage::MaskPixel afwDetect::setMaskFromFootprint(afwImage::Mask<afwImage::MaskPixel> *mask,
                                                    Footprint const& foot, afwImage::MaskPixel const bitmask);

#define INSTANTIATE(TYPE) \
template \
TYPE afwDetect::setImageFromFootprint(afwImage::Image<TYPE> *image,        \
                                      afwDetect::Footprint const& footprint, \
                                      TYPE const value);                \
template \
TYPE afwDetect::setImageFromFootprintList(afwImage::Image<TYPE> *image, \
                                          std::vector<afwDetect::Footprint::Ptr> const& footprints, \
                                          TYPE const value); \

INSTANTIATE(float)

// \endcond
