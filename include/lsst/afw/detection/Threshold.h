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
/*
 * Represent a detection threshold.
 */

namespace lsst {
namespace afw {
namespace detection {

/**
 * A Threshold is used to pass a threshold value to detection algorithms
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
        VALUE,       ///< Use pixel value
        BITMASK,     ///< Use (pixels & (given mask))
        STDEV,       ///< Use number of sigma given s.d.
        VARIANCE,    ///< Use number of sigma given variance
        PIXEL_STDEV  ///< Use number of sigma given per-pixel s.d.
    };

    /**
     * Threshold constructor
     */
    Threshold(double const value,                ///< desired threshold value
              ThresholdType const type = VALUE,  ///< interpretation of type
              bool const polarity = true,  ///< search pixel above threshold? (useful for -ve thresholds)
              double const includeMultiplier = 1.0  ///< threshold multiplier for inclusion in FootprintSet
              )
            : _value(value), _type(type), _polarity(polarity), _includeMultiplier(includeMultiplier) {}

    ~Threshold() = default;
    Threshold(Threshold const &) = default;
    Threshold(Threshold &&) = default;
    Threshold &operator=(Threshold const &) = default;
    Threshold &operator=(Threshold &&) = default;

    /// return type of threshold
    ThresholdType getType() const { return _type; }

    static ThresholdType parseTypeString(std::string const &typeStr);
    static std::string getTypeString(ThresholdType const &type);

    /**
     * return value of threshold, to be interpreted via type
     * @param param value of variance/stdev if needed
     * @returns value of threshold
     */
    double getValue(const double param = -1) const;

    /**
     * return value of threshold by interrogating the image, if required
     * @param image Image to interrogate, if threshold type demands
     * @returns value of threshold
     */
    template <typename ImageT>
    double getValue(ImageT const &image) const;

    /// return Threshold's polarity
    bool getPolarity() const { return _polarity; }
    /// set Threshold's polarity
    void setPolarity(bool const polarity  ///< desired polarity
    ) {
        _polarity = polarity;
    }

    /// return includeMultiplier
    double getIncludeMultiplier() const { return _includeMultiplier; }
    /// set includeMultiplier
    void setIncludeMultiplier(double const includeMultiplier  ///< desired multiplier
    ) {
        _includeMultiplier = includeMultiplier;
    }

private:
    double _value;              ///< value of threshold, to be interpreted via _type
    ThresholdType _type;        ///< type of threshold
    bool _polarity;             ///< true for positive polarity, false for negative
    double _includeMultiplier;  ///< multiplier for threshold needed for inclusion in FootprintSet
};

/**
 * Factory method for creating Threshold objects
 *
 * @param value value of threshold
 * @param type string representation of a ThresholdType. This parameter is
 *                optional. Allowed values are: "variance", "value", "stdev", "pixel_stdev"
 * @param polarity If true detect positive objects, false for negative
 *
 * @returns desired Threshold
 */
Threshold createThreshold(const double value, const std::string type = "value", const bool polarity = true);
}  // namespace detection
}  // namespace afw
}  // namespace lsst

#endif
