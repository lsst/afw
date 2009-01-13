/**
 * \file
 *
 * \brief Utilities to detect sets of Footprint%s
 *
 * Create and use an lsst::afw::detection::DetectionSet, a collection of pixels above (or below) some threshold
 * in an Image
 *
 * The "collections of pixels" are represented as lsst::afw::detection::Footprint%s, so an example application
 * would be:
 * \code
    namespace image = lsst::afw::image; namespace detection = lsst::afw::detection;

    image::MaskedImage<float> img(10,20);
    *img.getImage() = 100;

    detection::DetectionSet<float> sources(img, 10);
    cout << "Found " << sources.getFootprints().size() << " sources" << std::endl;
 * \endcode
 */
#include <algorithm>
#include <cassert>
#include <string>
#include <typeinfo>
#include <boost/format.hpp>
#include <lsst/daf/base/DataProperty.h>
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"

#include "lsst/afw/math/Statistics.h"
#include "lsst/afw/detection/Peak.h"
#include "lsst/afw/detection/Footprint.h"

namespace detection = lsst::afw::detection;
namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

/************************************************************************************************************/

namespace {
    /// Don't let doxygen see this block  \cond
/*
 * run-length code for part of object
 */
    class IdSpan {
    public:
        typedef boost::shared_ptr<IdSpan> Ptr;
        
        explicit IdSpan(int id, int y, int x0, int x1) : id(id), y(y), x0(x0), x1(x1) {}
        int id;                         /* ID for object */
        int y;				/* Row wherein IdSpan dwells */
        int x0, x1;                     /* inclusive range of columns */
    };
/*
 * comparison functor; sort by ID then row
 */
    struct IdSpanCompar : public std::binary_function<const IdSpan::Ptr, const IdSpan::Ptr, bool> {
        bool operator()(const IdSpan::Ptr a, const IdSpan::Ptr b) {
            if(a->id < b->id) {
                return true;
            } else if(a->id > b->id) {
                return false;
            } else {
                return (a->y < b->y) ? true : false;
            }
        }
    };
/*
 * Follow a chain of aliases, returning the final resolved value.
 */
    int resolve_alias(const std::vector<int>& aliases, /* list of aliases */
                      int id) {         /* alias to look up */
        int resolved = id;              /* resolved alias */
        
        while(id != aliases[id]) {
            resolved = id = aliases[id];
        }
        
        return(resolved);
    }
    /// \endcond
}

/************************************************************************************************************/
/**
 * Dtor for DetectionSet
 */
template<typename ImagePixelT, typename MaskPixelT>
detection::DetectionSet<ImagePixelT, MaskPixelT>::~DetectionSet() {
    delete &_footprints;
}

/**
 * \brief Find a Detection Set given a MaskedImage and a threshold
 *
 * Go through an image, finding sets of connected pixels above threshold
 * and assembling them into Footprint%s;  the resulting set of objects
 * is returned as an \c array<Footprint::Ptr>
 *
 * If threshold.getPolarity() is true, pixels above the Threshold are
 * assembled into Footprints; if it's false, then pixels \e below Threshold
 * are processed (Threshold will probably have to be below the background level
 * for this to make sense, e.g. for difference imaging)
 */
template<typename ImagePixelT, typename MaskPixelT>
detection::DetectionSet<ImagePixelT, MaskPixelT>::DetectionSet(
	const image::MaskedImage<ImagePixelT, MaskPixelT> &maskedImg, //!< MaskedImage to search for objects
        const Threshold& threshold,     //!< threshold to find objects
        const std::string& planeName,   //!< mask plane to set (if != "")
        const int npixMin)              //!< minimum number of pixels in an object
    : lsst::daf::data::LsstBase(typeid(this)),
      _footprints(*new FootprintList()),
      _region(*new image::BBox(image::PointI(maskedImg.getX0(), maskedImg.getY0()),
                               maskedImg.getWidth(), maskedImg.getHeight())) {
    int id;				/* object ID */
    int in_span;                        /* object ID of current IdSpan */
    int nobj = 0;			/* number of objects found */
    int x0 = 0;			        /* unpacked from a IdSpan */

    typedef typename image::Image<ImagePixelT> ImageT;
    
    const typename ImageT::Ptr img = maskedImg.getImage();
    const int row0 = img->getY0();
    const int col0 = img->getX0();
    const int height = img->getHeight();
    const int width = img->getWidth();
    assert (row0 == 0 && col0 == 0);    // address previous comment

    float thresholdParam = -1;          // standard deviation of image (may be needed by Threshold)
    if (threshold.getType() == Threshold::STDEV || threshold.getType() == Threshold::VARIANCE) {
        math::Statistics stats = math::make_Statistics(*img, math::STDEV);
        double const sd = stats.getValue(math::STDEV);
        
        if (threshold.getType() == Threshold::VARIANCE) {
            thresholdParam = sd*sd;
        } else {
            thresholdParam = sd;
        }
    }
/*
 * Storage for arrays that identify objects by ID. We want to be able to
 * refer to idp[-1] and idp[width], hence the (width + 2)
 */
    std::vector<int> id1(width + 2);
    std::fill(id1.begin(), id1.end(), 0);
    std::vector<int> id2(width + 2);
    std::fill(id2.begin(), id2.end(), 0);
    std::vector<int>::iterator idc = id1.begin() + 1; // object IDs in current/
    std::vector<int>::iterator idp = id2.begin() + 1; //                       previous row

    std::vector<int> aliases;           // aliases for initially disjoint parts of Footprints
    aliases.reserve(1 + height/20);	// initial size of aliases

    std::vector<IdSpan::Ptr> spans; // row:x0,x1 for objects
    spans.reserve(aliases.capacity());	// initial size of spans

    aliases.push_back(0);               // 0 --> 0
/*
 * Go through image identifying objects
 */
    typedef typename image::Image<ImagePixelT>::x_iterator x_iterator;
    const float thresholdVal = threshold.getValue(thresholdParam);
    const bool polarity = threshold.getPolarity();

    in_span = 0;			// not in a span
    for (int y = 0; y != height; ++y) {
        if (idc == id1.begin() + 1) {
            idc = id2.begin() + 1;
            idp = id1.begin() + 1;
        } else {
            idc = id1.begin() + 1;
            idp = id2.begin() + 1;
        }
        std::fill_n(idc - 1, width + 2, 0);
        
        in_span = 0;			/* not in a span */

        x_iterator pixPtr = img->row_begin(y);
        for (int x = 0; x < width; ++x, ++pixPtr) {
	     ImagePixelT pixVal = (polarity ? *pixPtr : -(*pixPtr));

            if (pixVal < thresholdVal) {
                if (in_span) {
                    IdSpan *sp = new IdSpan(in_span, y, x0, x - 1);
                    IdSpan::Ptr spp(sp);
                    spans.push_back(spp);

                    in_span = 0;
                }
            } else {			/* a pixel to fix */
                if(idc[x - 1] != 0) {
                    id = idc[x - 1];
                } else if(idp[x - 1] != 0) {
                    id = idp[x - 1];
                } else if(idp[x] != 0) {
                    id = idp[x];
                } else if(idp[x + 1] != 0) {
                    id = idp[x + 1];
                } else {
                    id = ++nobj;
                    aliases.push_back(id);
                }

                idc[x] = id;
                if(!in_span) {
                    x0 = x; in_span = id;
                }
/*
 * Do we need to merge ID numbers? If so, make suitable entries in aliases[]
 */
                if(idp[x + 1] != 0 && idp[x + 1] != id) {
                    aliases[resolve_alias(aliases, idp[x + 1])] = resolve_alias(aliases, id);
	       
                    idc[x] = id = idp[x + 1];
                }
            }
        }

        if(in_span) {
            IdSpan *sp = new IdSpan(in_span, y, x0, width - 1);
            IdSpan::Ptr spp(sp);
            spans.push_back(spp);
        }
    }
/*
 * Resolve aliases; first alias chains, then the IDs in the spans
 */
    for (unsigned int i = 0; i < spans.size(); i++) {
        spans[i]->id = resolve_alias(aliases, spans[i]->id);
    }
/*
 * Sort spans by ID, so we can sweep through them once
 */
    if(spans.size() > 0) {
        std::sort(spans.begin(), spans.end(), IdSpanCompar());
    }
/*
 * Build Footprints from spans
 */
    unsigned int i0;                    // initial value of i
    if(spans.size() > 0) {
        id = spans[0]->id;
        i0 = 0;
        for (unsigned int i = 0; i <= spans.size(); i++) { // <= size to catch the last object
            if(i == spans.size() || spans[i]->id != id) {
                Footprint *fp = new Footprint(i - i0, _region);
	    
                for(; i0 < i; i0++) {
                    fp->addSpan(spans[i0]->y + row0, spans[i0]->x0 + col0, spans[i0]->x1 + col0);
                }

                if (fp->getNpix() < npixMin) {
                    delete fp;
                } else {
                    Footprint::Ptr fpp(fp);
                    _footprints.push_back(fpp);
                }
            }

            if (i < spans.size()) {
                id = spans[i]->id;
            }
        }
    }
/*
 * Set Mask if requested
 */
    if (planeName == "") {
        return;
    }
    //
    // Define the maskPlane
    //
    const typename image::Mask<MaskPixelT>::Ptr mask = maskedImg.getMask();
    mask->addMaskPlane(planeName);

    MaskPixelT const bitPlane = mask->getPlaneBitMask(planeName);
    //
    // Set the bits where objects are detected
    //
    for (FootprintList::const_iterator fiter = _footprints.begin(); fiter != _footprints.end(); ++fiter) {
        const Footprint::Ptr foot = *fiter;

        for (std::vector<Span::Ptr>::const_iterator siter = foot->getSpans().begin();
             siter != foot->getSpans().end(); siter++) {
            const Span::Ptr span = *siter;
            for (typename image::Mask<MaskPixelT>::x_iterator ptr = mask->x_at(span->getX0(), span->getY()),
                     end = ptr + span->getX1() - span->getX0() + 1; ptr != end; ++ptr) {
                *ptr |= bitPlane;
            }
        }
    }
}

/************************************************************************************************************/
/**
 * Return a DetectionSet consisting a Footprint containing the point (x, y) (if above threshold)
 *
 * \todo Implement this.  There's RHL Pan-STARRS code to do it, but it isn't yet converted to LSST C++
 */
template<typename ImagePixelT, typename MaskPixelT>
detection::DetectionSet<ImagePixelT, MaskPixelT>::DetectionSet(
	const image::MaskedImage<ImagePixelT, MaskPixelT> &img, //!< Image to search for objects
        const Threshold& threshold,          //!< threshold to find objects
        int x,                               //!< Footprint should include this pixel (column)
        int y,                               //!< Footprint should include this pixel (row) 
        const std::vector<Peak> *peaks)      //!< Footprint should include at most one of these peaks
    : lsst::daf::data::LsstBase(typeid(this)),
      _footprints(*new FootprintList())
    {
}

/************************************************************************************************************/
namespace {
    /// Don't let doxygen see this block  \cond
    /*
     * A data structure to hold the starting point for a search for pixels above threshold,
     * used by pmFindFootprintAtPoint
     *
     * We don't want to find this span again --- it's already part of the footprint ---
     * so we set appropriate mask bits
     */
    //
    // An enum for what we should do with a Startspan
    //
    enum DIRECTION {DOWN = 0,	// scan down from this span
                    UP,		// scan up from this span
                    RESTART,     // restart scanning from this span
                    DONE		// this span is processed
    };
    //
    // A Class that remembers how to [re-]start scanning the image for pixels
    //
    template<typename MaskPixelT>
    class Startspan {
    public:
        typedef std::vector<boost::shared_ptr<Startspan> > Ptr;
        
        Startspan(const detection::Span *span, image::Mask<MaskPixelT> *mask, const DIRECTION dir);
        ~Startspan() { delete _span; }

        bool getSpan() { return _span; }
        bool Stop() { return _stop; }
        DIRECTION getDirection() { return _direction; }

        static int detectedPlane;       // The MaskPlane to use for detected pixels
        static int stopPlane;           // The MaskPlane to use for pixels that signal us to stop searching
    private:
        detection::Span::Ptr const _span; // The initial Span
        DIRECTION _direction;		// How to continue searching for further pixels
        bool _stop;                     // should we stop searching?
    };

    template<typename MaskPixelT>
    Startspan<MaskPixelT>::Startspan(const detection::Span *span, // The span in question
                         image::Mask<MaskPixelT> *mask, // Pixels that we've already detected
                         const DIRECTION dir // Should we continue searching towards the top of the image?
                        ) :
        _span(span),
        _direction(dir),
        _stop(false) {

        if (mask != NULL) {			// remember that we've detected these pixels
            mask->setMaskPlaneValues(detectedPlane, span->getX0(), span->getX1(), span->getY());

            const int y = span->getY() - mask->getY0();
            for (int x = span->getX0() - mask->getX0(); x <= span->getX1() - mask->getX0(); x++) {
                if (mask(x, y, stopPlane)) {
                    _stop = true;
                    break;
                }
            }
        }
    }

    template<typename ImagePixelT, typename MaskPixelT>
    class StartspanSet {
    public:
        StartspanSet(image::MaskedImage<ImagePixelT, MaskPixelT>& image) :
            _image(image->getImage()),
            _mask(image->getMask()),
            _spans(*new std::vector<typename Startspan<MaskPixelT>::Ptr>()) {}
        ~StartspanSet() { delete &_spans; }

        bool add(detection::Span *span, const DIRECTION dir, bool addToMask = true);
        bool process(detection::Footprint *fp,          // the footprint that we're building
                     const detection::Threshold& threshold, // Threshold
                     const float param = -1);           // parameter that Threshold may need
    private:
        const image::Image<ImagePixelT> *_image; // the Image we're searching
        image::Mask<MaskPixelT> *_mask;          // the mask that tells us where we've got to
        std::vector<typename Startspan<MaskPixelT>::Ptr>& _spans; // list of Startspans
    };

    //
    // Add a new Startspan to a StartspansSet.  Iff we see a stop bit, return true
    //
    template<typename ImagePixelT, typename MaskPixelT>
    bool StartspanSet<ImagePixelT, MaskPixelT>::add(detection::Span *span, // the span in question
                                                    const DIRECTION dir, // the desired direction to search
                                                    bool addToMask) { // should I add the Span to the mask?
        if (dir == RESTART) {
            if (add(span,  UP) || add(span, DOWN, false)) {
                return true;
            }
        } else {
            typename Startspan<MaskPixelT>::Ptr sspan(new Startspan<MaskPixelT>(span, dir));
            if (sspan->stop()) {        // we detected a stop bit
                return true;
            } else {
                _spans->push_back(sspan);
            }
        }

        return false;
    }

    /************************************************************************************************************/
    /*
     * Search the image for pixels above threshold, starting at a single Startspan.
     * We search the array looking for one to process; it'd be better to move the
     * ones that we're done with to the end, but it probably isn't worth it for
     * the anticipated uses of this routine.
     *
     * This is the guts of pmFindFootprintAtPoint
     */
    template<typename ImagePixelT, typename MaskPixelT>
    bool StartspanSet<ImagePixelT, MaskPixelT>::process(
		detection::Footprint *fp,              // the footprint that we're building
                const detection::Threshold& threshold, // Threshold
                const float param                      // parameter that Threshold may need
                                                             ) {
        const int row0 = _image->getY0();
        const int col0 = _image->getOffsetCols();
        const int height = _image->getHeight();
        const int width = _image->getWidth();
    
        /********************************************************************************************************/
        
        typedef typename std::vector<typename Startspan<MaskPixelT>::Ptr> StartspanListT;
        typedef typename std::vector<typename Startspan<MaskPixelT>::Ptr>::iterator StartspanListIterT;

        Startspan<MaskPixelT> *sspan = NULL;
        for (StartspanListIterT iter = _spans->begin(); iter != _spans->end(); iter++) {
            *sspan = *iter;
            if (sspan->getDirection() != DONE) {
                break;
            }
            if (sspan->Stop()) {
                break;
            }
        }
        if (sspan == NULL || sspan->getDirection() == DONE) { // no more Startspans to process
            return false;
        }
        if (sspan->Stop()) {			// they don't want any more spans processed
            return false;
        }
        /*
         * Work
         */
        const DIRECTION dir = sspan->getDirection();
        /*
         * Set initial span to the startspan
         */
        int x0 = sspan->getSpan()->getX0() - col0, x1 = sspan->getSpan()->getX1() - col0;
        /*
         * Go through image identifying objects
         */
        int nx0, nx1 = -1;			// new values of x0, x1
        const int di = (dir == UP) ? 1 : -1; // how much i changes to get to the next row
        bool stop = false;			// should I stop searching for spans?

        typedef typename image::Image<ImagePixelT>::pixel_accessor pixAccessT;
        const float thresholdVal = threshold.getValue(param);
	const bool polarity = threshold.getPolarity();
        
        for (int i = sspan->span->y -row0 + di; i < height && i >= 0; i += di) {
            pixAccessT imgRow = _image->origin().advance(0, i); // row pointer
            //maskPixAccessT maskRow = _mask->origin.advance(0, i);  //  masks's row pointer
            //
            // Search left from the pixel diagonally to the left of (i - di, x0). If there's
            // a connected span there it may need to grow up and/or down, so push it onto
            // the stack for later consideration
            //
            nx0 = -1;
            for (int j = x0 - 1; j >= -1; j--) {
		ImagePixelT pixVal = (j < 0) ? thresholdVal - 100 : (polarity ? imgRow[j] : -imgRow[j]);
                if (_mask(j, i, Startspan<MaskPixelT>::detectedPlane) || pixVal < threshold) {
                    if (j < x0 - 1) {	// we found some pixels above threshold
                        nx0 = j + 1;
                    }
                    break;
                }
            }
#if 0
            if (nx0 < 0) {			// no span to the left
                nx1 = x0 - 1;		// we're going to resume searching at nx1 + 1
            } else {
                //
                // Search right in leftmost span
                //
                //nx1 = 0;			// make gcc happy
                for (int j = nx0 + 1; j <= width; j++) {
		    ImagePixelT pixVal = (j >= width) ? threshold - 100 : 
			  (polarity ? (F32 ? imgRowF32[j] : imgRowS32[j]) : (F32 ? -imgRowF32[j] : -imgRowS32[j]));
                    if ((maskRow[j] & DETECTED) || pixVal < threshold) {
                        nx1 = j - 1;
                        break;
                    }
                }
	    
                const pmSpan *sp = pmFootprintAddSpan(fp, i + row0, nx0 + col0, nx1 + col0);
	    
                if (add_startspan(startspans, sp, mask, RESTART)) {
                    stop = true;
                    break;
                }
            }
            //
            // Now look for spans connected to the old span.  The first of these we'll
            // simply process, but others will have to be deferred for later consideration.
            //
            // In fact, if the span overhangs to the right we'll have to defer the overhang
            // until later too, as it too can grow in both directions
            //
            // Note that column width exists virtually, and always ends the last span; this
            // is why we claim below that sx1 is always set
            //
            bool first = false;		// is this the first new span detected?
            for (int j = nx1 + 1; j <= x1 + 1; j++) {
		ImagePixelT pixVal = (j >= width) ? threshold - 100 : 
		     (polarity ? (F32 ? imgRowF32[j] : imgRowS32[j]) : (F32 ? -imgRowF32[j] : -imgRowS32[j]));
                if (!(maskRow[j] & DETECTED) && pixVal >= threshold) {
                    int sx0 = j++;		// span that we're working on is sx0:sx1
                    int sx1 = -1;		// We know that if we got here, we'll also set sx1
                    for (; j <= width; j++) {
			 ImagePixelT pixVal = (j >= width) ? threshold - 100 : 
			      (polarity ? (F32 ? imgRowF32[j] : imgRowS32[j]) : (F32 ? -imgRowF32[j] : -imgRowS32[j]));
                        if ((maskRow[j] & DETECTED) || pixVal < threshold) { // end of span
                            sx1 = j;
                            break;
                        }
                    }
                    assert (sx1 >= 0);

                    const pmSpan *sp;
                    if (first) {
                        if (sx1 <= x1) {
                            sp = pmFootprintAddSpan(fp, i + row0, sx0 + col0, sx1 + col0 - 1);
                            if (add_startspan(startspans, sp, mask, DONE)) {
                                stop = true;
                                break;
                            }
                        } else {		// overhangs to right
                            sp = pmFootprintAddSpan(fp, i + row0, sx0 + col0, x1 + col0);
                            if (add_startspan(startspans, sp, mask, DONE)) {
                                stop = true;
                                break;
                            }
                            sp = pmFootprintAddSpan(fp, i + row0, x1 + 1 + col0, sx1 + col0 - 1);
                            if (add_startspan(startspans, sp, mask, RESTART)) {
                                stop = true;
                                break;
                            }
                        }
                        first = false;
                    } else {
                        sp = pmFootprintAddSpan(fp, i + row0, sx0 + col0, sx1 + col0 - 1);
                        if (add_startspan(startspans, sp, mask, RESTART)) {
                            stop = true;
                            break;
                        }
                    }
                }
            }

            if (stop || first == false) {	// we're done
                break;
            }

            x0 = nx0; x1 = nx1;
#endif
        }
        /*
         * Cleanup
         */

        sspan->_direction = DONE;
        return stop ? false : true;
    }
    /// \endcond
} 
#if 0
    

/*
 * Go through an image, starting at (row, col) and assembling all the pixels
 * that are connected to that point (in a chess kings-move sort of way) into
 * a pmFootprint.
 *
 * This is much slower than pmFindFootprints if you want to find lots of
 * footprints, but if you only want a small region about a given point it
 * can be much faster
 *
 * N.b. The returned pmFootprint is not in "normal form"; that is the pmSpans
 * are not sorted by increasing y, x0, x1.  If this matters to you, call
 * pmFootprintNormalize()
 */
pmFootprint *
pmFindFootprintAtPoint(const psImage *img,	// image to search
		       const Threshold& threshold, // Threshold
		       const psArray *peaks, // array of peaks; finding one terminates search for footprint
		       int row, int col) { // starting position (in img's parent's coordinate system)
   assert(img != NULL);

   bool F32 = false;			// is this an F32 image?
   if (img->type.type == PS_TYPE_F32) {
       F32 = true;
   } else if (img->type.type == PS_TYPE_S32) {
       F32 = false;
   } else {				// N.b. You can't trivially add more cases here; F32 is just a bool
       psError(PS_ERR_UNKNOWN, true, "Unsupported psImage type: %d", img->type.type);
       return NULL;
   }
   psF32 *imgRowF32 = NULL;		// row pointer if F32
   psS32 *imgRowS32 = NULL;		//  "   "   "  "  !F32
   
   const int row0 = img->row0;
   const int col0 = img->col0;
   const int height = img->getHeight();
   const int width = img->getWidth();
/*
 * Is point in image, and above threshold?
 */
   row -= row0; col -= col0;
   if (row < 0 || row >= height ||
       col < 0 || col >= width) {
        psError(PS_ERR_BAD_PARAMETER_VALUE, true,
                "row/col == (%d, %d) are out of bounds [%d--%d, %d--%d]",
		row + row0, col + col0, row0, row0 + height - 1, col0, col0 + width - 1);
       return NULL;
   }

   ImagePixelT pixVal = F32 ? img->data.F32[row][col] : img->data.S32[row][col];
   if (pixVal < threshold) {
       return pmFootprintAlloc(0, img);
   }
   
   pmFootprint *fp = pmFootprintAlloc(1 + img->getHeight()/10, img);
/*
 * We need a mask for two purposes; to indicate which pixels are already detected,
 * and to store the "stop" pixels --- those that, once reached, should stop us
 * looking for the rest of the pmFootprint.  These are generally set from peaks.
 */
   psImage *mask = psImageAlloc(width, height, PS_TYPE_MASK);
   P_PSIMAGE_SET_ROW0(mask, row0);
   P_PSIMAGE_SET_COL0(mask, col0);
   psImageInit(mask, INITIAL);
   //
   // Set stop bits from peaks list
   //
   assert (peaks == NULL || peaks->n == 0 || pmIsPeak(peaks->data[0]));
   if (peaks != NULL) {
       for (int i = 0; i < peaks->n; i++) {
	   pmPeak *peak = peaks->data[i];
	   mask->data.PS_TYPE_MASK_DATA[peak->y - mask->row0][peak->x - mask->col0] |= STOP;
       }
   }
/*
 * Find starting span passing through (row, col)
 */
   psArray *startspans = psArrayAllocEmpty(1); // spans where we have to restart the search
   
   imgRowF32 = img->data.F32[row];	// only one of
   imgRowS32 = img->data.S32[row];	//      these is valid!
   psMaskType *maskRow = mask->data.PS_TYPE_MASK_DATA[row];
   {
       int i;
       for (i = col; i >= 0; i--) {
	   pixVal = F32 ? imgRowF32[i] : imgRowS32[i];
	   if ((maskRow[i] & DETECTED) || pixVal < threshold) {
	       break;
	   }
       }
       int i0 = i;
       for (i = col; i < width; i++) {
	   pixVal = F32 ? imgRowF32[i] : imgRowS32[i];
	   if ((maskRow[i] & DETECTED) || pixVal < threshold) {
	       break;
	   }
       }
       int i1 = i;
       const pmSpan *sp = pmFootprintAddSpan(fp, row + row0, i0 + col0 + 1, i1 + col0 - 1);

       (void)add_startspan(startspans, sp, mask, RESTART);
   }
   /*
    * Now workout from those Startspans, searching for pixels above threshold
    */
   while (do_startspan(fp, img, mask, threshold, startspans)) continue;
   /*
    * Cleanup
    */
   psFree(mask);
   psFree(startspans);			// restores the image pixel

   return fp;				// pmFootprint really
}
#endif

/************************************************************************************************************/
/**
 * Grow all the Footprints in the input DetectionSet, returning a new DetectionSet
 *
 * The output DetectionSet may contain fewer Footprints, as some may well have been merged
 *
 * \todo Implement this.  There's RHL Pan-STARRS code to do it, but it isn't yet converted to LSST C++
 */
template<typename ImagePixelT, typename MaskPixelT>
detection::DetectionSet<ImagePixelT, MaskPixelT>::DetectionSet(
	const DetectionSet &set,
        int r)                          //!< Grow Footprints by r pixels
    : lsst::daf::data::LsstBase(typeid(this)),
      _footprints(*new FootprintList()) {
}

/************************************************************************************************************/
/**
 * Return the DetectionSet corresponding to the merge of two input DetectionSets
 *
 * \todo Implement this.  There's RHL Pan-STARRS code to do it, but it isn't yet converted to LSST C++
 */
template<typename ImagePixelT, typename MaskPixelT>
detection::DetectionSet<ImagePixelT, MaskPixelT>::DetectionSet(
	DetectionSet const& footprints1, DetectionSet const& footprints2,
        bool const includePeaks)
    : lsst::daf::data::LsstBase(typeid(this)),
      _footprints(*new FootprintList())
    {
}

/************************************************************************************************************/
/**
 * Return an Image with pixels set to the Footprint%s in the DetectionSet
 *
 * \returns an image::Image::Ptr
 */
template<typename ImagePixelT, typename MaskPixelT>
typename image::Image<boost::uint16_t>::Ptr detection::DetectionSet<ImagePixelT, MaskPixelT>::insertIntoImage(const bool relativeIDs) {
    typename image::Image<boost::uint16_t>::Ptr im(new image::Image<boost::uint16_t>(_region.getDimensions()));
    *im = 0;

    int id = 0;
    for (FootprintList::const_iterator fiter = _footprints.begin(); fiter != _footprints.end(); fiter++) {
        const Footprint::Ptr foot = *fiter;
        
        if (relativeIDs) {
            id++;
        } else {
            id = foot->getId();
        }
        
        foot->insertIntoImage(*im.get(), id);
    }
    
    return im;
}

/************************************************************************************************************/
//
// Explicit instantiations
//
template class detection::DetectionSet<int, image::MaskPixel>;
template class detection::DetectionSet<float, image::MaskPixel>;
template class detection::DetectionSet<double, image::MaskPixel>;
