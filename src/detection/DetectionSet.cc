#include <algorithm>
#include <cassert>
#include <string>
#include <typeinfo>
#include <boost/format.hpp>
#include <lsst/daf/base/DataProperty.h>
#include "lsst/afw/image/ImageExceptions.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"

#include "lsst/detection/Peak.h"
#include "lsst/detection/Footprint.h"

using namespace lsst::detection;

/************************************************************************************************************/

namespace {
/*
 * run-length code for part of object
 */
    class IdSpan {
    public:
        typedef boost::shared_ptr<IdSpan> PtrType;
        
        explicit IdSpan(int id, int y, int x0, int x1) : id(id), y(y), x0(x0), x1(x1) {}
        int id;                         /* ID for object */
        int y;				/* Row wherein IdSpan dwells */
        int x0, x1;                     /* inclusive range of columns */
    };
/*
 * comparison functor; sort by ID then row
 */
    struct IdSpanCompar : public std::binary_function<const IdSpan::PtrType, const IdSpan::PtrType, bool> {
        bool operator()(const IdSpan::PtrType a, const IdSpan::PtrType b) {
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
}

/************************************************************************************************************/
/**
 * Dtor for DetectionSet
 */
template<typename ImagePixelType, typename MaskPixelType>
DetectionSet<ImagePixelType, MaskPixelType>::~DetectionSet() {
    delete &_footprints;
}

/**
 * \brief Find a Detection Set given a MaskedImage and a threshold
 *
 * Go through an image, finding sets of connected pixels above threshold
 * and assembling them into Footprints;  the resulting set of objects
 * is returned as an array<Footprint::PtrType>
 *
 * If threshold.getPolarity() is false, pixels which are more negative than threshold are
 * assembled into Footprints.
 */
template<typename ImagePixelType, typename MaskPixelType>
DetectionSet<ImagePixelType, MaskPixelType>::DetectionSet(
	const lsst::afw::image::MaskedImage<ImagePixelType, MaskPixelType> &maskedImg, //!< MaskedImage to search for objects
        const Threshold& threshold,     //!< threshold to find objects
        const std::string& planeName,   //!< mask plane to set (if != "")
        const int npixMin)              //!< minimum number of pixels in an object
    : lsst::daf::data::LsstBase(typeid(this)),
      _footprints(*new std::vector<Footprint::PtrType>()),
      _region(*new vw::BBox2i(maskedImg.getOffsetCols(), maskedImg.getOffsetRows(),
                              maskedImg.getCols(), maskedImg.getRows())) {
    int id;				/* object ID */
    int in_span;                        /* object ID of current IdSpan */
    int nobj = 0;			/* number of objects found */
    int x0 = 0;			        /* unpacked from a IdSpan */

    const typename lsst::afw::image::Image<ImagePixelType>::ImagePtrT img = maskedImg.getImage();
    const int row0 = img->getOffsetRows();
    const int col0 = img->getOffsetCols();
    const int numRows = img->getRows();
    const int numCols = img->getCols();
    assert (row0 == 0 && col0 == 0);    // address previous comment

    float thresholdParam = -1;          // standard deviation of image (may be needed by Threshold)
    if (threshold.getType() == Threshold::STDEV || threshold.getType() == Threshold::VARIANCE) {
        float sd = stddev_channel_value(img->getIVw());
        if (threshold.getType() == Threshold::VARIANCE) {
            thresholdParam = sd*sd;
        } else {
            thresholdParam = sd;
        }
    }
/*
 * Storage for arrays that identify objects by ID. We want to be able to
 * refer to idp[-1] and idp[numCols], hence the (numCols + 2)
 */
    std::vector<int> id1(numCols + 2);
    std::fill(id1.begin(), id1.end(), 0);
    std::vector<int> id2(numCols + 2);
    std::fill(id2.begin(), id2.end(), 0);
    std::vector<int>::iterator idc = id1.begin() + 1; // object IDs in current/
    std::vector<int>::iterator idp = id2.begin() + 1; //                       previous row

    std::vector<int> aliases;           // aliases for initially disjoint parts of Footprints
    aliases.reserve(1 + numRows/20);	// initial size of aliases

    std::vector<IdSpan::PtrType> spans; // row:x0,x1 for objects
    spans.reserve(aliases.capacity());	// initial size of spans

    aliases.push_back(0);               // 0 --> 0
/*
 * Go through image identifying objects
 */
    typedef typename lsst::afw::image::Image<ImagePixelType>::pixel_accessor pixAccessT;
    const float thresholdVal = threshold.getValue(thresholdParam);
    const bool polarity = threshold.getPolarity();

    pixAccessT rowPtr = img->origin();   // row pointer
    in_span = 0;			// not in a span
    for (int y = 0; y < numRows; y++, rowPtr.next_row()) {
        if (idc == id1.begin() + 1) {
            idc = id2.begin() + 1;
            idp = id1.begin() + 1;
        } else {
            idc = id1.begin() + 1;
            idp = id2.begin() + 1;
        }
        std::fill_n(idc - 1, numCols + 2, 0);
        
        in_span = 0;			/* not in a span */
        pixAccessT pixPtr = rowPtr;
        for (int x = 0; x < numCols; x++, pixPtr.next_col()) {
	     ImagePixelType pixVal = (polarity ? *pixPtr : -(*pixPtr));

            if (pixVal < thresholdVal) {
                if (in_span) {
                    IdSpan *sp = new IdSpan(in_span, y, x0, x - 1);
                    IdSpan::PtrType spp(sp);
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
            IdSpan *sp = new IdSpan(in_span, y, x0, numCols - 1);
            IdSpan::PtrType spp(sp);
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
                lsst::detection::Footprint *fp = new Footprint(i - i0, _region);
	    
                for(; i0 < i; i0++) {
                    fp->addSpan(spans[i0]->y + row0, spans[i0]->x0 + col0, spans[i0]->x1 + col0);
                }

                if (fp->getNpix() < npixMin) {
                    delete fp;
                } else {
                    lsst::detection::Footprint::PtrType fpp(fp);
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
    const typename lsst::afw::image::Mask<MaskPixelType>::MaskPtrT mask = maskedImg.getMask();
    mask->addMaskPlane(planeName);

    MaskPixelType bitPlane = -1;
    mask->getPlaneBitMask(planeName, bitPlane);
    //
    // Set the bits where objects are detected
    //
    typedef typename lsst::afw::image::Mask<MaskPixelType>::pixel_accessor maskPixAccessT;

    for (std::vector<Footprint::PtrType>::const_iterator fiter = _footprints.begin(); fiter != _footprints.end(); fiter++) {
        const Footprint::PtrType foot = *fiter;

        for (std::vector<Span::PtrType>::const_iterator siter = foot->getSpans().begin();
             siter != foot->getSpans().end(); siter++) {
            const Span::PtrType span = *siter;
            maskPixAccessT spanPtr = mask->origin().advance(span->getX0(), span->getY());
            for (int x = span->getX0(); x <= span->getX1(); x++, spanPtr.next_col()) {
                *spanPtr |= bitPlane;
            }
        }
    }
}

/************************************************************************************************************/
/**
 * Return a DetectionSet consisting a Footprint containing the point (x, y) (if above threshold)
 */
template<typename ImagePixelType, typename MaskPixelType>
DetectionSet<ImagePixelType, MaskPixelType>::DetectionSet(
	const lsst::afw::image::MaskedImage<ImagePixelType, MaskPixelType> &img, //!< Image to search for objects
        const Threshold& threshold,          //!< threshold to find objects
        int x,                          //!< Footprint should include this pixel (column)
        int y,                          //!< Footprint should include this pixel (row) 
        const std::vector<Peak> *peaks)        //!< Footprint should include at most one of these peaks
    : lsst::daf::data::LsstBase(typeid(this)),
      _footprints(*new std::vector<Footprint::PtrType>())
    {
}

/************************************************************************************************************/
namespace {
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
    template<typename MaskPixelType>
    class Startspan {
    public:
        typedef std::vector<boost::shared_ptr<Startspan> > StartspanPtrT;
        
        Startspan(const Span *span, lsst::afw::image::Mask<MaskPixelType> *mask, const DIRECTION dir);
        ~Startspan() { delete _span; }

        bool getSpan() { return _span; }
        bool Stop() { return _stop; }
        DIRECTION getDirection() { return _direction; }

        static int detectedPlane;       // The MaskPlane to use for detected pixels
        static int stopPlane;           // The MaskPlane to use for pixels that signal us to stop searching
    private:
        const boost::shared_ptr<lsst::detection::Span> _span; // The initial Span
        DIRECTION _direction;		// How to continue searching for further pixels
        bool _stop;                      // should we stop searching?
    };

    template<typename MaskPixelType>
    Startspan<MaskPixelType>::Startspan(const Span *span, // The span in question
                         lsst::afw::image::Mask<MaskPixelType> *mask, // Pixels that we've already detected
                         const DIRECTION dir // Should we continue searching towards the top of the image?
                        ) :
        _span(span),
        _direction(dir),
        _stop(false) {

        if (mask != NULL) {			// remember that we've detected these pixels
            mask->setMaskPlaneValues(detectedPlane, span->getX0(), span->getX1(), span->getY());

            const int y = span->getY() - mask->getOffsetRows();
            for (int x = span->getX0() - mask->getOffsetCols(); x <= span->getX1() - mask->getOffsetCols(); x++) {
                if (mask(x, y, stopPlane)) {
                    _stop = true;
                    break;
                }
            }
        }
    }

    template<typename ImagePixelType, typename MaskPixelType>
    class StartspanSet {
    public:
        StartspanSet(lsst::afw::image::MaskedImage<ImagePixelType, MaskPixelType>& image) :
            _image(image->getImage()),
            _mask(image->getMask()),
            _spans(*new std::vector<typename Startspan<MaskPixelType>::StartspanPtrT>()) {}
        ~StartspanSet() { delete &_spans; }

        bool add(Span *span, const DIRECTION dir, bool addToMask = true);
        bool process(Footprint *fp,     // the footprint that we're building
                     const Threshold& threshold,	// Threshold
                     const float param = -1);   // parameter that Threshold may need
    private:
        const lsst::afw::image::Image<ImagePixelType> *_image; // the Image we're searching
        lsst::afw::image::Mask<MaskPixelType> *_mask; // the mask that tells us where we've got to
        std::vector<typename Startspan<MaskPixelType>::StartspanPtrT>& _spans; // list of Startspans
    };

    //
    // Add a new Startspan to a StartspansSet.  Iff we see a stop bit, return true
    //
    template<typename ImagePixelType, typename MaskPixelType>
    bool StartspanSet<ImagePixelType, MaskPixelType>::add(Span *span, // the span in question
                                                          const DIRECTION dir, // the desired direction to search
                                                          bool addToMask) { // should I add the Span to the mask?
        if (dir == RESTART) {
            if (add(span,  UP) || add(span, DOWN, false)) {
                return true;
            }
        } else {
            typename Startspan<MaskPixelType>::StartspanPtrT sspan(new Startspan<MaskPixelType>(span, dir));
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
    template<typename ImagePixelType, typename MaskPixelType>
    bool StartspanSet<ImagePixelType, MaskPixelType>::process(Footprint *fp,     // the footprint that we're building
                                                              const Threshold& threshold, // Threshold
                                                              const float param // parameter that Threshold may need
                                                             ) {
        const int row0 = _image->getOffsetRows();
        const int col0 = _image->getOffsetCols();
        const int numRows = _image->getRows();
        const int numCols = _image->getCols();
    
        /********************************************************************************************************/
        
        typedef typename std::vector<typename Startspan<MaskPixelType>::StartspanPtrT> StartspanListT;
        typedef typename std::vector<typename Startspan<MaskPixelType>::StartspanPtrT>::iterator StartspanListIterT;

        Startspan<MaskPixelType> *sspan = NULL;
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

        typedef typename lsst::afw::image::Image<ImagePixelType>::pixel_accessor pixAccessT;
        const float thresholdVal = threshold.getValue(param);
	const bool polarity = threshold.getPolarity();
        
        for (int i = sspan->span->y -row0 + di; i < numRows && i >= 0; i += di) {
            pixAccessT imgRow = _image->origin().advance(0, i); // row pointer
            //maskPixAccessT maskRow = _mask->origin.advance(0, i);  //  masks's row pointer
            //
            // Search left from the pixel diagonally to the left of (i - di, x0). If there's
            // a connected span there it may need to grow up and/or down, so push it onto
            // the stack for later consideration
            //
            nx0 = -1;
            for (int j = x0 - 1; j >= -1; j--) {
		ImagePixelType pixVal = (j < 0) ? thresholdVal - 100 : (polarity ? imgRow[j] : -imgRow[j]);
                if (_mask(j, i, Startspan<MaskPixelType>::detectedPlane) || pixVal < threshold) {
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
                for (int j = nx0 + 1; j <= numCols; j++) {
		    ImagePixelType pixVal = (j >= numCols) ? threshold - 100 : 
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
            // Note that column numCols exists virtually, and always ends the last span; this
            // is why we claim below that sx1 is always set
            //
            bool first = false;		// is this the first new span detected?
            for (int j = nx1 + 1; j <= x1 + 1; j++) {
		ImagePixelType pixVal = (j >= numCols) ? threshold - 100 : 
		     (polarity ? (F32 ? imgRowF32[j] : imgRowS32[j]) : (F32 ? -imgRowF32[j] : -imgRowS32[j]));
                if (!(maskRow[j] & DETECTED) && pixVal >= threshold) {
                    int sx0 = j++;		// span that we're working on is sx0:sx1
                    int sx1 = -1;		// We know that if we got here, we'll also set sx1
                    for (; j <= numCols; j++) {
			 ImagePixelType pixVal = (j >= numCols) ? threshold - 100 : 
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


} // XXX
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
   const int numRows = img->numRows;
   const int numCols = img->numCols;
/*
 * Is point in image, and above threshold?
 */
   row -= row0; col -= col0;
   if (row < 0 || row >= numRows ||
       col < 0 || col >= numCols) {
        psError(PS_ERR_BAD_PARAMETER_VALUE, true,
                "row/col == (%d, %d) are out of bounds [%d--%d, %d--%d]",
		row + row0, col + col0, row0, row0 + numRows - 1, col0, col0 + numCols - 1);
       return NULL;
   }

   ImagePixelType pixVal = F32 ? img->data.F32[row][col] : img->data.S32[row][col];
   if (pixVal < threshold) {
       return pmFootprintAlloc(0, img);
   }
   
   pmFootprint *fp = pmFootprintAlloc(1 + img->numRows/10, img);
/*
 * We need a mask for two purposes; to indicate which pixels are already detected,
 * and to store the "stop" pixels --- those that, once reached, should stop us
 * looking for the rest of the pmFootprint.  These are generally set from peaks.
 */
   psImage *mask = psImageAlloc(numCols, numRows, PS_TYPE_MASK);
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
       for (i = col; i < numCols; i++) {
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
 */
template<typename ImagePixelType, typename MaskPixelType>
DetectionSet<ImagePixelType, MaskPixelType>::DetectionSet(
	const DetectionSet &set,
        int r)                          //!< Grow Footprints by r pixels
    : lsst::daf::data::LsstBase(typeid(this)),
      _footprints(*new std::vector<Footprint::PtrType>()) {
}

/************************************************************************************************************/
/**
 * Return the DetectionSet corresponding to the merge of two input DetectionSets
 */
template<typename ImagePixelType, typename MaskPixelType>
DetectionSet<ImagePixelType, MaskPixelType>::DetectionSet(
	const DetectionSet &footprints1, const DetectionSet &footprints2,
        const int includePeaks)
    : lsst::daf::data::LsstBase(typeid(this)),
      _footprints(*new std::vector<Footprint::PtrType>())
    {
}

/************************************************************************************************************/
/**
 * Return an Image<boost::uint16_t> consisting of the Footprints in the DetectionSet
 */
template<typename ImagePixelType, typename MaskPixelType>
typename lsst::afw::image::Image<boost::uint16_t>::ImagePtrT DetectionSet<ImagePixelType, MaskPixelType>::insertIntoImage(const bool relativeIDs) {
    const unsigned int ncols = _region.width();
    const unsigned int nrows = _region.height();

    typename lsst::afw::image::Image<boost::uint16_t>::ImagePtrT im(new lsst::afw::image::Image<boost::uint16_t>(ncols, nrows));

    int id = 0;
    for (std::vector<Footprint::PtrType>::const_iterator fiter = _footprints.begin(); fiter != _footprints.end(); fiter++) {
        const lsst::detection::Footprint::PtrType foot = *fiter;
        
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
// Implicit instantiations
//
template class DetectionSet<int, lsst::afw::image::maskPixelType>;
template class DetectionSet<float, lsst::afw::image::maskPixelType>;
template class DetectionSet<double, lsst::afw::image::maskPixelType>;
