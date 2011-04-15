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
        STDEV,               //!< Use number of sigma given s.d.
        VARIANCE             //!< Use number of sigma given variance
    }; 

    /** 
     * Threshold constructor
     *
     * @param value desired threshold value
     * @param type interpretation of type
     * @param polarity search pixel above threshold? (useful for -ve thresholds)
     */
    Threshold(
        float const value,
        ThresholdType const type = VALUE,
        bool const polarity = true
    ) : _value(value), _type(type), _polarity(polarity) {}

    //! return type of threshold
    ThresholdType getType() const { return _type; }

    static ThresholdType parseTypeString(std::string const & typeStr);
    static std::string getTypeString(ThresholdType const & type);

    /**
     * return value of threshold, to be interpreted via type
     * @param param value of variance/stdev if needed
     * @return value of threshold
     */
    float getValue(const float param = -1) const; 
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
}}}

#endif
