#if !defined(LSST_DETECTION_FOOTPRINT_H)
#define LSST_DETECTION_FOOTPRINT_H
/**
 * \file
 * \brief Represent a set of pixels of an arbitrary shape and size
 *
 * Footprint is fundamental in astronomical image processing, as it defines what
 * is meant by a Source.
 */
#include <list>
#include <cmath>
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include "lsst/pex/policy/Policy.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/detection/Peak.h"

namespace lsst { namespace afw { namespace detection {
/*!
 * \brief A range of pixels within one row of an Image
 *
 * \note This isn't really for public consumption, as it's the insides
 * of a Footprint --- it should be made a private class within
 * Footprint (but not until I'm fully checked in, which is hard
 * at 30000' over Peru).  I'm now at 30000' over the Atlantic,
 * but the same argument holds.
 */
class Span {
public:
    typedef boost::shared_ptr<Span> Ptr;
    typedef boost::shared_ptr<const Span> ConstPtr;

    Span(int y,                         //!< Row that Span's in
         int x0,                        //!< Starting column (inclusive)
         int x1)                        //!< Ending column (inclusive)
        : _y(y), _x0(x0), _x1(x1) {}
    ~Span() {}

    int getX0() const { return _x0; }         ///< Return the starting x-value
    int getX1() const { return _x1; }         ///< Return the ending x-value
    int getY()  const { return _y; }          ///< Return the y-value
    int getWidth() const { return _x1 - _x0 + 1; } ///< Return the number of pixels

    std::string toString() const;    

    void shift(int dx, int dy) { _x0 += dx; _x1 += dx; _y += dy; }

    friend class Footprint;
private:
    int _y;                             //!< Row that Span's in
    int _x0;                            //!< Starting column (inclusive)
    int _x1;                            //!< Ending column (inclusive)
};

/************************************************************************************************************/
/**
 * \brief A Threshold is used to pass a threshold value to the FootprintSet constructors
 *
 * The threshold may be a simple value (type == VALUE), or in units of the image
 * standard deviation; you may specify that you'll provide the standard deviation
 * (type == STDEV) or variance (type == VARIANCE)
 *
 * Note that the constructor is not declared explicit, so you may pass a bare
 * threshold, and it'll be interpreted as a VALUE.
 */
class Threshold {
public:
    /// Types of threshold:
    typedef enum { VALUE,               //!< Use pixel value
                   STDEV,               //!< Use number of sigma given s.d.
                   VARIANCE             //!< Use number of sigma given variance
    } ThresholdType;

    Threshold(const float value,        //!< desired value
              const ThresholdType type = VALUE, //!< interpretation of type
              const bool polarity = true        //!< Search for pixels above threshold? (Useful for -ve thresholds)
             ) : _value(value), _type(type), _polarity(polarity) {}

    //! return type of threshold
    ThresholdType getType() const { return _type; }
    //! return value of threshold, to be interpreted via type
    float getValue(const float param = -1 //!< value of variance/stdev if needed
                  ) const {
        switch (_type) {
          case STDEV:
            if (param <= 0) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                                  (boost::format("St. dev. must be > 0: %g") % param).str());
            }
            return _value*param;
          case VALUE:
            return _value;
          case VARIANCE:
            if (param <= 0) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                                  (boost::format("Variance must be > 0: %g") % param).str());
            }
            return _value*std::sqrt(param);
          default:
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              (boost::format("Unsopported type: %d") % _type).str());
        }
    }
    /// return Threshold's polarity
    bool getPolarity() const { return _polarity; }
private:
    float _value;                       //!< value of threshold, to be interpreted via _type
    ThresholdType _type;                //!< type of threshold
    bool _polarity;                     //!< true for positive polarity, false for negative
};

// brief Factory method for creating Threshold objects
Threshold createThreshold(const float value,
                          const std::string type = "value",
                          const bool polarity = true);
/************************************************************************************************************/
/*!
 * \brief A set of pixels in an Image
 *
 * A Footprint is a set of pixels, usually but not necessarily contiguous.
 * There are constructors to find Footprints above some threshold in an Image
 * (see FootprintSet), or to create Footprints in the shape of various
 * geometrical figures
 */
class Footprint : public lsst::daf::data::LsstBase {
public:
    typedef boost::shared_ptr<Footprint> Ptr;
    /// The Footprint's Span list
    typedef std::vector<Span::Ptr> SpanList;

    Footprint(int nspan = 0, const image::BBox region=image::BBox());
    Footprint(const image::BBox& bbox, const image::BBox region=image::BBox());
    Footprint(const image::BCircle& circle, const image::BBox region=image::BBox());

    ~Footprint();

    int getId() const { return _fid; }   //!< Return the Footprint's unique ID
    SpanList &getSpans() { return _spans; } //!< return the Span%s contained in this Footprint
    const SpanList &getSpans() const { return _spans; } //!< return the Span%s contained in this Footprint
    std::vector<Peak::Ptr> &getPeaks() { return _peaks; } //!< Return the Peak%s contained in this Footprint
    int getNpix() const { return _npix; }     //!< Return the number of pixels in this Footprint

    const Span& addSpan(const int y, const int x0, const int x1);
    const Span& addSpan(Span const& span);
    const Span& addSpan(Span const& span, int dx, int dy);

    void shift(int dx, int dy);

    const image::BBox& getBBox() const { return _bbox; } //!< Return the Footprint's bounding box
    /// Return the corners of the MaskedImage the footprints live in
    image::BBox const& getRegion() const { return _region; }
    /// Set the corners of the MaskedImage wherein the footprints dwell
    void setRegion(lsst::afw::image::BBox const& region) { _region = region; }
    
    void normalize();
    int setNpix();
    void setBBox();

    void insertIntoImage(lsst::afw::image::Image<boost::uint16_t>& idImage, int const id,
                         image::BBox const& region=image::BBox()) const;
private:
    Footprint(const Footprint &);                   //!< No copy constructor
    Footprint operator = (Footprint const &) const; //!< no assignment
    static int id;
    mutable int _fid;                    //!< unique ID
    int _npix;                           //!< number of pixels in this Footprint
    
    SpanList &_spans;                    //!< the Spans contained in this Footprint
    image::BBox _bbox;                   //!< the Footprint's bounding box
    std::vector<Peak::Ptr> &_peaks;      //!< the Peaks lying in this footprint
    mutable image::BBox _region;         //!< The corners of the MaskedImage the footprints live in
    bool _normalized;                    //!< Are the spans sorted? 
};

Footprint::Ptr growFootprint(Footprint const &foot, int ngrow, bool isotropic=true);
Footprint::Ptr growFootprint(Footprint::Ptr const &foot, int ngrow, bool isotropic=true);

std::vector<lsst::afw::image::BBox> footprintToBBoxList(Footprint const& foot);

template<typename ImageT>
typename ImageT::Pixel setImageFromFootprint(ImageT *image,
                                             Footprint const& footprint,
                                             typename ImageT::Pixel const value);
template<typename ImageT>
typename ImageT::Pixel setImageFromFootprintList(ImageT *image,
                                                 std::vector<Footprint::Ptr> const& footprints,
                                                 typename ImageT::Pixel  const value);
template<typename MaskT>
MaskT setMaskFromFootprint(lsst::afw::image::Mask<MaskT> *mask,
                           Footprint const& footprint,
                           MaskT const bitmask);
template<typename MaskT>
MaskT setMaskFromFootprintList(lsst::afw::image::Mask<MaskT> *mask,
                               std::vector<Footprint::Ptr> const& footprints,
                               MaskT const bitmask);
template<typename MaskT>
Footprint::Ptr footprintAndMask(Footprint::Ptr const & foot,
                                typename image::Mask<MaskT>::Ptr const & mask,
                                MaskT bitmask);
    
/************************************************************************************************************/
/*!
 * \brief A set of Footprints, associated with a MaskedImage
 *
 */
template<typename ImagePixelT, typename MaskPixelT=lsst::afw::image::MaskPixel>
class FootprintSet : public lsst::daf::data::LsstBase {
public:
    typedef boost::shared_ptr<FootprintSet> Ptr;
    /// The FootprintSet's set of Footprint%s
    typedef std::vector<Footprint::Ptr> FootprintList;

    FootprintSet(lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT> const& img,
                 Threshold const& threshold,
                 std::string const& planeName = "",
                 int const npixMin=1);
    FootprintSet(lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT> const& img,
                 Threshold const& threshold,
                 int x,
                 int y,
                 std::vector<Peak> const* peaks = NULL);
    FootprintSet(FootprintSet const&);
    FootprintSet(FootprintSet const& set, int r, bool isotropic=true);
    FootprintSet(FootprintSet const& footprints1, FootprintSet const& footprints2,
                 bool const includePeaks);
    ~FootprintSet();

    FootprintSet& operator=(FootprintSet const& rhs);

    template<typename RhsImagePixelT, typename RhsMaskPixelT>
    void swap(FootprintSet<RhsImagePixelT, RhsMaskPixelT> &rhs) {
        using std::swap;                    // See Meyers, Effective C++, Item 25
        
        swap(_footprints, rhs.getFootprints());
        image::BBox rhsRegion = rhs.getRegion();
        swap(_region, rhsRegion);
    }
    
    FootprintList& getFootprints() { return _footprints; } //!< Retun the Footprint%s of detected objects
    FootprintList const& getFootprints() const { return _footprints; } //!< Retun the Footprint%s of detected objects
    void setRegion(lsst::afw::image::BBox const& region);
    image::BBox const& getRegion() const { return _region; } //!< Return the corners of the MaskedImage

#if 0                                   // these are equivalent, but the former confuses swig
    typename image::Image<boost::uint16_t>::Ptr insertIntoImage(const bool relativeIDs);
#else
    typename boost::shared_ptr<image::Image<boost::uint16_t> > insertIntoImage(const bool relativeIDs);
#endif

    void setMask(lsst::afw::image::Mask<MaskPixelT> *mask, ///< Set bits in the mask
                 std::string const& planeName   ///< Here's the name of the mask plane to fit
                ) {
        detection::setMaskFromFootprintList(mask, getFootprints(),
                                            image::Mask<MaskPixelT>::getPlaneBitMask(planeName));        
    }
private:
    FootprintList & _footprints;        //!< the Footprints of detected objects
    image::BBox _region;                //!< The corners of the MaskedImage that the detections live in
};

#if 0
psErrorCode pmPeaksAssignToFootprints(psArray *footprints, const psArray *peaks);

psErrorCode pmFootprintArrayCullPeaks(const psImage *img, const psImage *weight, psArray *footprints,
                                 const float nsigma, const float threshold_min);
psErrorCode pmFootprintCullPeaks(const psImage *img, const psImage *weight, pmFootprint *fp,
                                 const float nsigma, const float threshold_min);

psArray *pmFootprintArrayToPeaks(const psArray *footprints);
#endif

/************************************************************************************************************/
/**
 * \brief A functor class to allow users to process all the pixels in a Footprint
 *
 * There's an annotated example of a FootprintFunctor in action
 * \link FootprintFunctorsExample FootprintFunctors here\endlink
 */
template <typename ImageT>
class FootprintFunctor {
public:
    FootprintFunctor(ImageT const& image    ///< The image that the Footprint lives in
                    ) : _image(image) {}

    virtual ~FootprintFunctor() = 0;

    /**
     * A function that's called at the beginning of apply; useful if apply
     * calculates a per-footprint quantity
     */
    virtual void reset() {}
    virtual void reset(Footprint const& foot) {}

    /**
     * \brief Apply operator() to each pixel in the Footprint
     */
    void apply(Footprint const& foot    ///< The Footprint in question
              ) {
        reset();
        reset(foot);

        if (foot.getSpans().empty()) {
            return;
        }

        image::BBox const& bbox = foot.getBBox();
        image::BBox region = foot.getRegion();
        if (region &&
            (!region.contains(bbox.getLLC()) || !region.contains(bbox.getURC()))) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                              (boost::format("Footprint with BBox (%d,%d) -- (%dx%d) doesn't fit in image with BBox (%d,%d) -- (%dx%d)") %
                               bbox.getX0() % bbox.getY0() % bbox.getX1() % bbox.getY0() %
                               region.getX0() % region.getY0() % region.getX1() % region.getY1()).str());
        }

        int ox1 = 0, oy = 0;            // Current position of the locator (in the SpanList loop)
        typename ImageT::xy_locator loc = _image.xy_at(-_image.getX0(), -_image.getY0()); // Origin of the Image's pixels

        for (Footprint::SpanList::const_iterator siter = foot.getSpans().begin();
             siter != foot.getSpans().end(); siter++) {
            Span::Ptr const span = *siter;

            int const y = span->getY();
            int const x0 = span->getX0();
            int const x1 = span->getX1();

            loc += lsst::afw::image::pair2I(x0 - ox1, y - oy);

            for (int x = x0; x <= x1; ++x, ++loc.x()) {
                operator()(loc, x, y);
            }

            ox1 = x1 + 1; oy = y;
        }
    }
    /// Return the image
    ImageT const& getImage() const { return _image; }    

    /// The operator to be applied to each pixel in the Footprint.
    ///
    /// N.b. the coordinates (x, y) are relative to the origin of the image's parent
    /// if it exists (i.e. they obey getX0/getY0)
    virtual void operator()(typename ImageT::xy_locator loc, int x, int y) = 0;
private:
    ImageT const& _image;               // The image that the Footprints live in
};

/************************************************************************************************************/

template<typename ImagePixelT, typename MaskPixelT>
typename detection::FootprintSet<ImagePixelT, MaskPixelT>::Ptr makeFootprintSet(
        image::MaskedImage<ImagePixelT, MaskPixelT> const& img,
        Threshold const& threshold,
        std::string const& planeName = "",
        int const npixMin=1) {
    return typename detection::FootprintSet<ImagePixelT, MaskPixelT>::Ptr(new FootprintSet<ImagePixelT, MaskPixelT>(img, threshold, planeName, npixMin));
}

template<typename ImagePixelT, typename MaskPixelT>
typename detection::FootprintSet<ImagePixelT, MaskPixelT>::Ptr makeFootprintSet(
        image::MaskedImage<ImagePixelT, MaskPixelT> const& img,
        Threshold const& threshold,
        int x,
        int y,
        std::vector<Peak> const* peaks = NULL) {
    return typename detection::FootprintSet<ImagePixelT, MaskPixelT>::Ptr(new FootprintSet<ImagePixelT, MaskPixelT>(img, threshold, x, y, peaks));
}

/************************************************************************************************************/
///
/// Although FootprintFunctor is pure virtual, this is needed by subclasses
///
/// It wasn't defined in the class body as I want swig to know that the class is pure virtual
///
template <typename ImageT>
FootprintFunctor<ImageT>::~FootprintFunctor() {}
            
}}}
#endif
