/*****************************************************************************/
/** \file
 *
 * \brief Footprint and associated classes
 */
#include <cassert>
#include <string>
#include <typeinfo>
#include <boost/format.hpp>
#include "lsst/pex/logging/Trace.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelFunctions.h"
#include "lsst/afw/detection/Footprint.h"

namespace math = lsst::afw::math;
namespace detection = lsst::afw::detection;
namespace image = lsst::afw::image;

/************************************************************************************************************/
/**
 * Return a string-representation of a Span
 */
std::string detection::Span::toString() {
    return (boost::format("%d: %d..%d") % _y % _x0 % _x1).str();
}

/**
 * Compare two Span%s by y, then x0, then x1
 *
 * A utility function passed to qsort
 * \note This should be replaced by functor so that we can use std::sort
 */
int detection::Span::compareByYX(const void **a, const void **b) {
    const detection::Span *sa = *reinterpret_cast<const detection::Span **>(a);
    const detection::Span *sb = *reinterpret_cast<const detection::Span **>(b);

    if (sa->_y < sb->_y) {
	return -1;
    } else if (sa->_y == sb->_y) {
	if (sa->_x0 < sb->_x0) {
	    return -1;
	} else if (sa->_x0 == sb->_x0) {
	    if (sa->_x1 < sb->_x1) {
		return -1;
	    } else if (sa->_x1 == sb->_x1) {
		return 0;
	    } else {
		return 1;
	    }
	} else {
	    return 1;
	}
    } else {
	return 1;
    }
}

/************************************************************************************************************/
/// Counter for Footprint IDs
int detection::Footprint::id = 0;
/**
 * Create a Footprint
 *
 * \throws lsst::pex::exceptions::InvalidParameterException in nspan is < 0
 */
detection::Footprint::Footprint(int nspan,         //!< initial number of Span%s in this Footprint
                                const image::BBox region) //!< Bounding box of MaskedImage footprint lives in
    : lsst::daf::data::LsstBase(typeid(this)),
      _fid(++id),
      _npix(0),
      _spans(*new detection::Footprint::SpanList),
      _bbox(image::BBox()),
      _peaks(*new std::vector<Peak::Ptr>),
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
detection::Footprint::Footprint(const image::BBox& bbox, //!< The bounding box defining the rectangle
                                const image::BBox region) //!< Bounding box of MaskedImage footprint lives in
    : lsst::daf::data::LsstBase(typeid(this)),
      _fid(++id),
      _npix(0),
      _spans(*new detection::Footprint::SpanList),
      _bbox(image::BBox()),
      _peaks(*new std::vector<Peak::Ptr>),
      _region(region),
      _normalized(false) {
    const int x0 = bbox.getX0();
    const int y0 = bbox.getY0();
    const int x1 = bbox.getX1();
    const int y1 = bbox.getY1();

    for (int i = y0; i <= y1; i++) {
        addSpan(i, x0, x1);
    }
}

/**
 * Create a circular Footprint
 */
detection::Footprint::Footprint(image::BCircle const& circle, //!< The center and radius of the circle
                                image::BBox const region)     //!< Bounding box of MaskedImage footprint lives in
    : lsst::daf::data::LsstBase(typeid(this)),
      _fid(++id),
      _npix(0),
      _spans(*new detection::Footprint::SpanList),
      _bbox(image::BBox()),
      _peaks(*new std::vector<Peak::Ptr>),
      _region(region),
      _normalized(false) {
    const int xc = circle.getCenter().getX(); // x-centre
    const int yc = circle.getCenter().getY(); // y-centre
    const int r2 = static_cast<int>(circle.getRadius()*circle.getRadius() + 0.5); // rounded radius^2
    const int r = static_cast<int>(std::sqrt(static_cast<double>(r2))); // truncated radius; r*r <= r2
   
    for(int i = -r; i <= r; i++) {
        int hlen = static_cast<int>(std::sqrt(static_cast<double>(r2 - i*i)));
        addSpan(yc + i, xc - hlen, xc + hlen);
    }
}

/**
 * Destroy a Footprint
 */
detection::Footprint::~Footprint() {
    delete &_spans;
    delete &_peaks;
}

/**
 * Normalise a Footprint, soring spans and setting the BBox
 */
void detection::Footprint::normalize() {
    if (!_normalized) {
	//_peaks = psArraySort(fp->peaks, pmPeakSortBySN);
        setBBox();
	_normalized = true;
    }
}

/**
 * Add a Span to a footprint, returning a reference to the new Span.
 */
detection::Span const& detection::Footprint::addSpan(int const y, //!< row value
                                                     int const x0, //!< starting column
                                                     int const x1 //!< ending column
                                                    ) {
    if (x1 < x0) {
        return this->addSpan(y, x1, x0);
    }

    detection::Span::Ptr sp(new detection::Span(y, x0, x1));
    _spans.push_back(sp);
    
    _npix += x1 - x0 + 1;

    _bbox.grow(image::PointI(x0, y));
    _bbox.grow(image::PointI(x1, y));

    return *sp.get();
}
/**
 * Add a Span to a Footprint returning a reference to the new Span
 */
const detection::Span& detection::Footprint::addSpan(detection::Span const& span ///< new Span being added
                              ) {
    detection::Span::Ptr sp(new detection::Span(span));
    
    _spans.push_back(sp);
    
    _npix += span._x1 - span._x0 + 1;

    _bbox.grow(image::PointI(span._x0, span._y));
    _bbox.grow(image::PointI(span._x1, span._y));

    return *sp;
}

/**
 * Add a Span to a Footprint returning a reference to the new Span
 */
const detection::Span& detection::Footprint::addSpan(detection::Span const& span, ///< new Span being added
                                                     int dx,                      ///< Add dx to span's x coords
                                                     int dy                       ///< Add dy to span's y coords
                              ) {
    return addSpan(span._y + dy, span._x0 + dx, span._x1 + dx);
}
/**
 * Shift a Footprint by <tt>(dx, dy)</tt>
 */
void detection::Footprint::shift(int dx, //!< How much to move footprint in column direction
                                 int dy  //!< How much to move in row direction
                      ) {
    for (Footprint::SpanList::iterator siter = _spans.begin(); siter != _spans.end(); ++siter){
        detection::Span::Ptr span = *siter;

        span->_y += dy;
        span->_x0 += dx;
        span->_x1 += dx;
    }

    _bbox.shift(dx, dy);
}

/**
 * Tell \c this to calculate its bounding box
 */
void detection::Footprint::setBBox() {
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
        const detection::Span::Ptr sp = *spi;
	if (sp->_x0 < x0) x0 = sp->_x0;
	if (sp->_x1 > x1) x1 = sp->_x1;
	if (sp->_y < y0) y0 = sp->_y;
	if (sp->_y > y1) y1 = sp->_y;
    }

    _bbox = image::BBox(image::PointI(x0, y0), image::PointI(x1, y1));
}

/**
 * Tell \c this to count its pixels
 */
int detection::Footprint::setNpix() {
    _npix = 0;
    for (Footprint::SpanList::const_iterator spi = _spans.begin(); spi != _spans.end(); spi++) {
        const detection::Span::Ptr sp = *spi;
        _npix += sp->_x1 - sp->_x0 + 1;
   }

   return _npix;
}

/**
 * Set the pixels in idImage which are in Footprint by adding the specified value to the Image
 */
void detection::Footprint::insertIntoImage(image::Image<boost::uint16_t>& idImage, //!< Image to contain the footprint
                                           int const id //!< Add id to idImage for pixels in the Footprint
                                          ) const {
    int const width = _region.getWidth();
    int const height = _region.getHeight();
    int const x0 = _region.getX0();
    int const y0 = _region.getY0();

    if (width != idImage.getWidth() || height != idImage.getHeight()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          (boost::format("Image of size (%dx%d) doesn't match Footprint's host Image of size (%dx%d)") %
                              idImage.getWidth() % idImage.getHeight() % width % height).str());
    }

    for (Footprint::SpanList::const_iterator spi = _spans.begin(); spi != _spans.end(); ++spi) {
        detection::Span::Ptr const span = *spi;

        for (image::Image<boost::uint16_t>::x_iterator ptr = idImage.x_at(span->getX0() - x0, span->getY() - y0),
                 end = ptr + span->getWidth(); ptr != end; ++ptr) {
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
detection::Footprint::Ptr detection::footprintAndMask(
	detection::Footprint::Ptr const& foot,          ///< The initial Footprint
        typename image::Mask<MaskT>::Ptr const& mask,   ///< The mask to & with foot
        MaskT bitMask                   ///< Only consider these bits
                                                     ) {
    detection::Footprint::Ptr out(new detection::Footprint());

    return out;
}

/************************************************************************************************************/
/**
 * \brief OR bitmask into all the Mask's pixels which are in the Footprint
 *
 * \return bitmask
 */
template<typename MaskT>
MaskT detection::setMaskFromFootprint(typename image::Mask<MaskT>::Ptr mask, ///< Mask to set
                                      detection::Footprint const& foot,      ///< Footprint specifying desired pixels
                                      MaskT const bitmask                    ///< Bitmask to OR into mask
                                     ) {

    int const width = static_cast<int>(mask->getWidth());
    int const height = static_cast<int>(mask->getHeight());    
    
    for (detection::Footprint::SpanList::const_iterator siter = foot.getSpans().begin();
         siter != foot.getSpans().end(); siter++) {
        detection::Span::Ptr const span = *siter;
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
/**
 * \brief OR bitmask into all the Mask's pixels which are in the set of Footprint%s
 *
 * \return bitmask
 */
template<typename MaskT>
MaskT detection::setMaskFromFootprintList(
	typename image::Mask<MaskT>::Ptr mask, ///< Mask to set
        std::vector<detection::Footprint::Ptr> const& footprints, ///< Footprint list specifying desired pixels
        MaskT const bitmask             ///< Bitmask to OR into mask
                                               ) {
    for (std::vector<detection::Footprint::Ptr>::const_iterator fiter = footprints.begin();
         fiter != footprints.end(); ++fiter) {
        (void)setMaskFromFootprint(mask, **fiter, bitmask);
    }
    
    return bitmask;
}

/************************************************************************************************************/
/*
 * Worker routine for the pmSetFootprintArrayIDs/pmSetFootprintID (and pmMergeFootprintArrays)
 */
template <typename IDPixelT>
static void set_footprint_id(typename image::Image<IDPixelT>::Ptr idImage,	// the image to set
                             detection::Footprint::Ptr foot, // the footprint to insert
                             const int id,          // the desired ID
                             int dx = 0, int dy = 0 // Add these to all x/y in the Footprint
                            ) {
    for (detection::Footprint::SpanList::const_iterator siter = foot->getSpans().begin();
							siter != foot->getSpans().end(); siter++) {
        detection::Span::Ptr const span = *siter;
        for (typename image::Image<IDPixelT>::x_iterator ptr = idImage->x_at(span->getX0() + dx, span->getY() + dy),
                 end = ptr + span->getWidth(); ptr != end; ++ptr) {
            *ptr += id;
        }
    }
}

template <typename IDPixelT>
static void
set_footprint_array_ids(typename image::Image<IDPixelT>::Ptr idImage, // the image to set
                        const std::vector<detection::Footprint::Ptr>& footprints, // the footprints to insert
			const bool relativeIDs) { // show IDs starting at 0, not Footprint->id
    int id = 0;				// first index will be 1

    for (std::vector<detection::Footprint::Ptr>::const_iterator fiter = footprints.begin();
         fiter != footprints.end(); ++fiter) {
        const detection::Footprint::Ptr foot = *fiter;
        
        if (relativeIDs) {
            id++;
        } else {
            id = foot->getId();
        }
        
        set_footprint_id<IDPixelT>(idImage, foot, id);
    }
}

template static void set_footprint_array_ids<int>(image::Image<int>::Ptr idImage,
                                                  const std::vector<detection::Footprint::Ptr>& footprints,
                                                  const bool relativeIDs);

/************************************************************************************************************/
/*
 * Create an image from a Footprint's bounding box
 */
template <typename IDImageT>
static typename image::Image<IDImageT>::Ptr makeImageFromBBox(const image::BBox bbox) {
    typename image::Image<IDImageT>::Ptr idImage(new image::Image<IDImageT>(bbox.getDimensions()));
    idImage->setXY0(bbox.getLLC());

    return idImage;
}

/************************************************************************************************************/
/*
 * Set an image to the value of footprint's ID wherever they may fall
 */
template <typename IDImageT>
typename boost::shared_ptr<image::Image<IDImageT> > setFootprintArrayIDs(
	std::vector<detection::Footprint::Ptr> const& footprints, // the footprints to insert
        bool const relativeIDs                                    // show IDs starting at 1, not pmFootprint->id
                                               ) {
    std::vector<detection::Footprint::Ptr>::const_iterator fiter = footprints.begin();
    if (fiter == footprints.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, "You didn't provide any footprints");
    }
    const detection::Footprint::Ptr foot = *fiter;

    typename image::Image<IDImageT>::Ptr idImage = makeImageFromBBox<IDImageT>(foot->getRegion());
    *idImage = 0;
    /*
     * do the work
     */
    set_footprint_array_ids<IDImageT>(idImage, footprints, relativeIDs);
    
    return idImage;
}

template image::Image<int>::Ptr setFootprintArrayIDs(std::vector<detection::Footprint::Ptr> const& footprints,
                                                     bool const relativeIDs);
/*
 * Set an image to the value of Footprint's ID wherever it may fall
 */
template <typename IDImageT>
typename boost::shared_ptr<image::Image<IDImageT> > setFootprintID(detection::Footprint::Ptr const& foot, // the Footprint to insert
                                                    int const id // the desired ID
                                                   ) {
    typename image::Image<IDImageT>::Ptr idImage = makeImageFromBBox<IDImageT>(foot->getBBox());
    *idImage = 0;
    /*
     * do the work
     */
    set_footprint_id<IDImageT>(idImage, foot, id);

    return idImage;
}

template image::Image<int>::Ptr setFootprintID(detection::Footprint::Ptr const& foot, int const id);

/************************************************************************************************************/
/*
 * Grow a Footprint isotropically by r pixels, returning a new Footprint
 */
detection::Footprint::Ptr detection::growFootprint(
	detection::Footprint::Ptr const &foot, //!< The Footprint to grow 
        int ngrow                       //!< how much to grow foot
                                                 ) {
    if (ngrow <= 0) {
	ngrow = 1;                      // ngrow == 1 => no grow
    }
    /*
     * We'll insert the footprints into an image, then convolve with a disk,
     * then extract a footprint from the result --- this is magically what we want.
     */
    image::BBox bbox = foot->getBBox();
    bbox.grow(image::PointI(bbox.getX0() - 2*ngrow - 1, bbox.getY0() - 2*ngrow - 1));
    bbox.grow(image::PointI(bbox.getX1() + 2*ngrow + 1, bbox.getY1() + 2*ngrow + 1));
    image::Image<int>::Ptr idImage = makeImageFromBBox<int>(bbox);
    idImage->setXY0(image::PointI(0, 0));

    set_footprint_id<int>(idImage, foot, 1, -bbox.getX0(), -bbox.getY0());

    image::Image<double>::Ptr circle_im(new image::Image<double>(2*ngrow + 1, 2*ngrow + 1));
    *circle_im = 0;
    for (int r = -ngrow; r <= ngrow; ++r) {
        image::Image<double>::x_iterator row = circle_im->x_at(0, r + ngrow);
	for (int c = -ngrow; c <= ngrow; ++c, ++row) {
	    if (r*r + c*c <= ngrow*ngrow) {
                *row = 8;
	    }
	}
    }

    math::FixedKernel::PtrT circle(new math::FixedKernel(*circle_im));
    // Here's the actual grow step
    image::MaskedImage<int>::Ptr convolvedImage(new image::MaskedImage<int>(idImage->getDimensions()));
    math::convolve(*convolvedImage->getImage(), *idImage, *circle, 0, false);

    DetectionSet<int>::Ptr grownList(new DetectionSet<int>(*convolvedImage, 0.5, "", 1));

    assert (grownList->getFootprints().size() > 0);
    detection::Footprint::Ptr grown = *grownList->getFootprints().begin();
    //
    // Fix the coordinate system to be that of foot
    //
    grown->shift(bbox.getX0(), bbox.getY1());

    return grown;
}

#if 0

/************************************************************************************************************/
/*
 * Grow a psArray of pmFootprints isotropically by r pixels, returning a new psArray of new pmFootprints
 */
psArray *pmGrowFootprintArray(const psArray *footprints, // footprints to grow
			      int r) {	// how much to grow each footprint
    assert (footprints->n == 0 || pmIsFootprint(footprints->data[0]));

    if (footprints->n == 0) {		// we don't know the size of the footprint's region
	return psArrayAlloc(0);
    }
    /*
     * We'll insert the footprints into an image, then convolve with a disk,
     * then extract a footprint from the result --- this is magically what we want.
     */
    psImage *idImage = pmSetFootprintArrayIDs(footprints, true);
    if (r <= 0) {
	r = 1;				// r == 1 => no grow
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
    psFree(idImage); psFree(grownIdImage);
    /*
     * Now assign the peaks appropriately.  We could do this more efficiently
     * using grownIdImage (which we just freed), but this is easy and probably fast enough
     */
    const psArray *peaks = pmFootprintArrayToPeaks(footprints);
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
psArray *pmMergeFootprintArrays(const psArray *footprints1, // one set of footprints
				const psArray *footprints2, // the other set
				const int includePeaks) { // which peaks to set? 0x1 => footprints1, 0x2 => 2
    assert (footprints1->n == 0 || pmIsFootprint(footprints1->data[0]));
    assert (footprints2->n == 0 || pmIsFootprint(footprints2->data[0]));

    if (footprints1->n == 0 || footprints2->n == 0) {		// nothing to do but put copies on merged
	const psArray *old = (footprints1->n == 0) ? footprints2 : footprints1;

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
	const psArray *peaks = pmFootprintArrayToPeaks(footprints1);
	pmPeaksAssignToFootprints(merged, peaks);
	psFree((psArray *)peaks);
    }

    if (includePeaks & 0x2) {
	const psArray *peaks = pmFootprintArrayToPeaks(footprints2);
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
pmPeaksAssignToFootprints(psArray *footprints,	// the pmFootprints
			  const psArray *peaks) { // the pmPeaks
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
    const int y0 = ids->y0;
    const int x0 = ids->x0;
    const int numRows = ids->numRows;
    const int numCols = ids->numCols;

    for (int i = 0; i < peaks->n; i++) {
	pmPeak *peak = peaks->data[i];
	const int x = peak->x - x0;
	const int y = peak->y - y0;
	
	assert (x >= 0 && x < numCols && y >= 0 && y < numRows);
	int id = ids->data.S32[y][x - x0];

	if (id == 0) {			// peak isn't in a footprint, so make one for it
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
		j--;			// we moved everything down one
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
psErrorCode pmFootprintCullPeaks(const psImage *img, // the image wherein lives the footprint
				 const psImage *weight,	// corresponding variance image
				 pmFootprint *fp, // Footprint containing mortal peaks
				 const float nsigma_delta, // how many sigma above local background a peak
				 	// needs to be to survive
				 const float min_threshold) { // minimum permitted coll height
    assert (img != NULL); assert (img->type.type == PS_TYPE_F32);
    assert (weight != NULL); assert (weight->type.type == PS_TYPE_F32);
    assert (img->y0 == weight->y0 && img->x0 == weight->x0);
    assert (fp != NULL);

    if (fp->peaks == NULL || fp->peaks->n == 0) { // nothing to do
	return PS_ERR_NONE;
    }

    psRegion subRegion;			// desired subregion; 1 larger than bounding box (grr)
    subRegion.x0 = fp->bbox.x0; subRegion.x1 = fp->bbox.x1 + 1;
    subRegion.y0 = fp->bbox.y0; subRegion.y1 = fp->bbox.y1 + 1;
    const psImage *subImg = psImageSubset((psImage *)img, subRegion);
    const psImage *subWt = psImageSubset((psImage *)weight, subRegion);
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
	const pmPeak *peak = fp->peaks->data[i];
	int x = peak->x - subImg->x0;
	int y = peak->y - subImg->y0;
	//
	// Find the level nsigma below the peak that must separate the peak
	// from any of its friends
	//
	assert (x >= 0 && x < subImg->numCols && y >= 0 && y < subImg->numRows);
	const float stdev = std::sqrt(subWt->data.F32[y][x]);
	float threshold = subImg->data.F32[y][x] - nsigma_delta*stdev;
	if (isnan(threshold) || threshold < min_threshold) {
#if 1					// min_threshold is assumed to be below the detection threshold,
					// so all the peaks are pmFootprint, and this isn't the brightest
	    (void)psArrayRemoveIndex(fp->peaks, i);
	    i--;			// we moved everything down one
	    continue;
#else
#error n.b. We will be running LOTS of checks at this threshold, so only find the footprint once
	    threshold = min_threshold;
#endif
	}
	if (threshold > subImg->data.F32[y][x]) {
	    threshold = subImg->data.F32[y][x] - 10*FLT_EPSILON;
	}

	const int peak_id = 1;		// the ID for the peak of interest
	brightPeaks->n = i;		// only stop at a peak brighter than we are
	pmFootprint *peakFootprint = pmFindFootprintAtPoint(subImg, threshold, brightPeaks, peak->y, peak->x);
	brightPeaks->n = 0;		// don't double free
	psImage *idImg = pmSetFootprintID(peakFootprint, peak_id);
	psFree(peakFootprint);

	int j;
	for (j = 0; j < i; j++) {
	    const pmPeak *peak2 = fp->peaks->data[j];
	    int x2 = peak2->x - subImg->x0;
	    int y2 = peak2->y - subImg->y0;
	    const int peak2_id = idImg->data.S32[y2][x2]; // the ID for some other peak

	    if (peak2_id == peak_id) {	// There's a brighter peak within the footprint above
		;			// threshold; so cull our initial peak
		(void)psArrayRemoveIndex(fp->peaks, i);
		i--;			// we moved everything down one
		break;
	    }
	}
	if (j == i) {
	    j++;
	}

	psFree(idImg);
    }

    brightPeaks->n = 0; psFree(brightPeaks);
    psFree((psImage *)subImg);
    psFree((psImage *)subWt);

    return PS_ERR_NONE;
}

/*
 * Cull an entire psArray of pmFootprints
 */
psErrorCode
pmFootprintArrayCullPeaks(const psImage *img, // the image wherein lives the footprint
			  const psImage *weight,	// corresponding variance image
			  psArray *footprints, // array of pmFootprints
			  const float nsigma_delta, // how many sigma above local background a peak
    					// needs to be to survive
			  const float min_threshold) { // minimum permitted coll height
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
psArray *pmFootprintArrayToPeaks(const psArray *footprints) {
   assert(footprints != NULL);
   assert(footprints->n == 0 || pmIsFootprint(footprints->data[0]));

   int npeak = 0;
   for (int i = 0; i < footprints->n; i++) {
      const pmFootprint *fp = footprints->data[i];
      npeak += fp->peaks->n;
   }

   psArray *peaks = psArrayAllocEmpty(npeak);
   
   for (int i = 0; i < footprints->n; i++) {
      const pmFootprint *fp = footprints->data[i];
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
detection::Footprint::Ptr detection::footprintAndMask(detection::Footprint::Ptr const & foot,
                                                      image::Mask<image::MaskPixel>::Ptr const & mask,
                                                      image::MaskPixel bitMask);
        
template
image::MaskPixel detection::setMaskFromFootprintList(image::Mask<image::MaskPixel>::Ptr mask,
                                                     std::vector<detection::Footprint::Ptr> const& footprints,
                                                     image::MaskPixel const bitmask);
// \endcond
