/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#if !defined(LSST_DETECTION_THRESHOLD_H)
#define LSST_DETECTION_THRESHOLD_H
/**
 * \file
 * \brief Represent a detection threshold.
 */

namespace lsst { 
namespace afw {
namespace detection {

/**
 * \brief A Threshold is used to pass a threshold value to detection algorithms
 *
 * The threshold may be a simple value (type == VALUE), or in units of the image
 * standard deviation. Alternatively you may specify that you'll provide the 
 * standard deviation (type == STDEV) or variance (type == VARIANCE)
 *
 * Note that the constructor is not declared explicit, so you may pass a bare
 * threshold, and it'll be interpreted as a VALUE.
 */
class Threshold {
public:    
    /// Types of threshold:
    enum ThresholdType {
        VALUE,               //!< Use pixel value
        BITMASK,             //!< Use (pixels & (given mask))
        STDEV,               //!< Use number of sigma given s.d.
        VARIANCE,            //!< Use number of sigma given variance
        PIXEL_STDEV          //!< Use number of sigma given per-pixel s.d.
    }; 

    /** 
     * Threshold constructor
     */
    Threshold(
        double const value,               ///< desired threshold value
        ThresholdType const type = VALUE, ///< interpretation of type
        bool const polarity = true,       ///< search pixel above threshold? (useful for -ve thresholds)
        double const includeMultiplier = 1.0 ///< threshold multiplier for inclusion in FootprintSet
             ) : _value(value), _type(type), _polarity(polarity),
                 _includeMultiplier(includeMultiplier) {}

    //! return type of threshold
    ThresholdType getType() const { return _type; }

    static ThresholdType parseTypeString(std::string const & typeStr);
    static std::string getTypeString(ThresholdType const & type);

    /**
     * return value of threshold, to be interpreted via type
     * @param param value of variance/stdev if needed
     * @return value of threshold
     */
    double getValue(const double param = -1) const; 

    /**
     * return value of threshold by interrogating the image, if required
     * @param image Image to interrogate, if threshold type demands
     * @return value of threshold
     */
    template<typename ImageT>
    double getValue(ImageT const& image) const;

    /// return Threshold's polarity
    bool getPolarity() const { return _polarity; }
    /// set Threshold's polarity
    void setPolarity(bool const polarity ///< desired polarity
                    ) { _polarity = polarity; }

    /// return includeMultiplier
    double getIncludeMultiplier() const { return _includeMultiplier; }
    /// set includeMultiplier
    void setIncludeMultiplier(double const includeMultiplier ///< desired multiplier
                             ) { _includeMultiplier = includeMultiplier; }

private:
    double _value;                      //!< value of threshold, to be interpreted via _type
    ThresholdType _type;                //!< type of threshold
    bool _polarity;                     //!< true for positive polarity, false for negative
    double _includeMultiplier;          //!< multiplier for threshold needed for inclusion in FootprintSet
};

// brief Factory method for creating Threshold objects
Threshold createThreshold(const double value,
                          const std::string type = "value",
                          const bool polarity = true);
}}}

#endif
