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
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/detection/Peak.h"

namespace lsst { namespace afw { namespace detection {
namespace image = lsst::afw::image;
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

    Span(int y,                         //!< Row that Span's in
         int x0,                        //!< Starting column (inclusive)
         int x1)                        //!< Ending column (inclusive)
        : _y(y), _x0(x0), _x1(x1) {}
    ~Span() {}

    int getX0() { return _x0; }         ///< Return the starting x-value
    int getX1() { return _x1; }         ///< Return the ending x-value
    int getWidth() { return _x1 - _x0 + 1; } ///< Return the number of pixels
    int getY() { return _y; }                ///< Return the y-value

    std::string toString();    

    void shift(int dx, int dy) { _x0 += dx; _x1 += dx; _y += dy; }

    int compareByYX(const void **a, const void **b);

    friend class Footprint;
private:
    int _y;                             //!< Row that Span's in
    int _x0;                            //!< Starting column (inclusive)
    int _x1;                            //!< Ending column (inclusive)
};

/************************************************************************************************************/
/**
 * \brief A Threshold is used to pass a threshold value to the DetectionSet constructors
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
                throw lsst::pex::exceptions::InvalidParameter(boost::format("St. dev. must be > 0: %g") % param);
            }
            return _value*param;
          case VALUE:
            return _value;
          case VARIANCE:
            if (param <= 0) {
                throw lsst::pex::exceptions::InvalidParameter(boost::format("Variance must be > 0: %g") % param);
            }
            return _value*std::sqrt(param);
          default:
            throw lsst::pex::exceptions::InvalidParameter(boost::format("Unsopported type: %d") % _type);
        }
    }
    /// return Threshold's polarity
    bool getPolarity() const { return _polarity; }
private:
    float _value;                       //!< value of threshold, to be interpreted via _type
    ThresholdType _type;                //!< type of threshold
    bool _polarity;                     //!< true for positive polarity, false for negative
};

/************************************************************************************************************/
/*!
 * \brief A set of pixels in an Image
 *
 * A Footprint is a set of pixels, usually but not necessarily contiguous.
 * There are constructors to find Footprints above some threshold in an Image
 * (see DetectionSet), or to create Footprints in the shape of various
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
    const image::BBox& getRegion() const { return _region; }
    
    void normalize();
    int setNpix();
    void setBBox();

    void insertIntoImage(image::Image<boost::uint16_t>& idImage, const int id) const;
private:
    Footprint(const Footprint &);       //!< No copy constructor
    Footprint operator = (Footprint const &) const; //!< no assignment
    static int id;
    mutable int _fid;                    //!< unique ID
    int _npix;                          //!< number of pixels in this Footprint
    
    SpanList &_spans; //!< the Spans contained in this Footprint
    image::BBox _bbox;                   //!< the Footprint's bounding box
    std::vector<Peak::Ptr> &_peaks; //!< the Peaks lying in this footprint
    const image::BBox _region;           //!< The corners of the MaskedImage the footprints live in
    bool _normalized;                   //!< Are the spans sorted? 
};

Footprint::Ptr growFootprint(Footprint::Ptr const &foot, int ngrow);

template<typename MaskT>
MaskT setMaskFromFootprint(typename image::Mask<MaskT>::Ptr mask,
                           Footprint::Ptr const footprint,
                           MaskT const bitmask);
template<typename MaskT>
MaskT setMaskFromFootprintList(typename lsst::afw::image::Mask<MaskT>::Ptr mask,
                               std::vector<detection::Footprint::Ptr> const& footprints,
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
class DetectionSet : public lsst::daf::data::LsstBase {
public:
    typedef boost::shared_ptr<DetectionSet> Ptr;
    /// The DetectionSet's set of Footprint%s
    typedef std::vector<Footprint::Ptr> FootprintList;

    DetectionSet(image::MaskedImage<ImagePixelT, MaskPixelT> const& img,
                 Threshold const& threshold,
                 std::string const& planeName = "",
                 int const npixMin=1);
    DetectionSet(image::MaskedImage<ImagePixelT, MaskPixelT> const& img,
                 Threshold const& threshold,
                 int x,
                 int y,
                 std::vector<Peak> const* peaks = NULL);
    DetectionSet(DetectionSet const& set, int r=0);
    DetectionSet(DetectionSet const& footprints1, DetectionSet const& footprints2,
                 bool const includePeaks);
    ~DetectionSet();

    FootprintList& getFootprints() { return _footprints; } //!< Retun the Footprint%s of detected objects
    image::BBox const& getRegion() const { return _region; } //!< Return the corners of the MaskedImage

#if 0                                   // these are equivalent, but the former confuses swig
    typename image::Image<boost::uint16_t>::Ptr insertIntoImage(const bool relativeIDs);
#else
    typename boost::shared_ptr<image::Image<boost::uint16_t> > insertIntoImage(const bool relativeIDs);
#endif
private:
    FootprintList & _footprints;  //!< the Footprints of detected objects
    const image::BBox _region;      //!< The corners of the MaskedImage that the detections live in
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
 */
template <typename ImageT>
class FootprintFunctor {
public:
    FootprintFunctor(ImageT const& image    ///< The image that the Footprint lives in
                    ) : _image(image) {}

    virtual ~FootprintFunctor() {}
    /**
     * \brief Apply operator() to each pixel in the Footprint
     */
    void apply(Footprint const& foot    ///< The Footprint in question
              ) {
        if (foot.getSpans().empty()) {
            return;
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
    /// if it exists.
    virtual void operator()(typename ImageT::xy_locator loc, int x, int y) = 0;
private:
    ImageT const& _image;               // The image that the Footprints live in
};
            
}}}
#endif
