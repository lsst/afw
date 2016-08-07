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
#if !defined(LSST_DETECTION_FOOTPRINTCTRL_H)
#define LSST_DETECTION_FOOTPRINTCTRL_H
/**
 * \file
 * \brief Control Footprint-related algorithms
 */

namespace lsst {
namespace afw {
namespace detection {
/*!
 * \brief A Control Object for Footprints, controlling e.g. how they are grown
 *
 */
class FootprintControl {
    enum TBool { FALSE_=false, TRUE_=true, NONE_ }; // ternary boolean value. N.b. _XXX is reserved

    static std::pair<bool, bool> makePairFromTBool(TBool const val)
    {
        return (val == NONE_) ? std::make_pair(false, false) : std::make_pair(true, val == TRUE_);
    }
public:
    explicit FootprintControl() : _circular(NONE_), _isotropic(NONE_),
                               _left(NONE_), _right(NONE_), _up(NONE_), _down(NONE_) {}
    explicit FootprintControl(bool circular, bool isotropic=false) :
        _circular(circular ? TRUE_ : FALSE_), _isotropic(isotropic ? TRUE_ : FALSE_),
        _left(NONE_), _right(NONE_), _up(NONE_), _down(NONE_) {}
    explicit FootprintControl(bool left, bool right, bool up, bool down) :
        _circular(NONE_), _isotropic(NONE_),
        _left(left ? TRUE_ : FALSE_), _right(right ? TRUE_ : FALSE_),
        _up(up ? TRUE_ : FALSE_), _down(down ? TRUE_ : FALSE_) {}

#define DEFINE_ACCESSORS(NAME, UNAME)                                   \
    /** Set whether Footprint should be grown in a NAME sort of   */    \
    void grow ## UNAME(bool val         /**!< Should grow be of type NAME? */ \
                      ) { _ ## NAME = val ? TRUE_ : FALSE_; }           \
    /** Return <isSet, Value> for NAME grows */                         \
    std::pair<bool, bool> is ## UNAME() const {                         \
        return makePairFromTBool(_ ## NAME);                            \
    }

    DEFINE_ACCESSORS(circular, Circular)
    //DEFINE_ACCESSORS(isotropic, Isotropic) // special, as isotropic => circular
    DEFINE_ACCESSORS(left, Left)
    DEFINE_ACCESSORS(right, Right)
    DEFINE_ACCESSORS(up, Up)
    DEFINE_ACCESSORS(down, Down)

    /// Set whether Footprint should be grown isotropically
    void growIsotropic(bool val         //!< Should grow be isotropic?
                      ) {
        _circular = TRUE_;
        _isotropic = val ? TRUE_ : FALSE_;
    }
    /// Return <isSet, Value> for isotropic grows
    std::pair<bool, bool> isIsotropic() const {
        return makePairFromTBool(_isotropic);
    }

#undef DEFINE_ACCESSORS
private:
    TBool _circular;                    // grow in all directions ( == left & right & up & down)
    TBool _isotropic;                   // go to the expense of as isotropic a grow as possible
    TBool _left, _right, _up, _down;    // grow in selected directions?
};

/*!
 * \brief A control object for HeavyFootprint%s
 */
    class HeavyFootprintCtrl {
    public:
        enum ModifySource {NONE, SET,};

        explicit HeavyFootprintCtrl(ModifySource modifySource=NONE) :
            _modifySource(modifySource),
            _imageVal(0.0), _maskVal(0), _varianceVal(0.0)
            {}

        ModifySource getModifySource() const { return _modifySource; }
        void setModifySource(ModifySource modifySource) { _modifySource = modifySource; }

        double getImageVal() const { return _imageVal; }
        void setImageVal(double imageVal) { _imageVal = imageVal; }
        long getMaskVal() const { return _maskVal; }
        void setMaskVal(long maskVal) { _maskVal = maskVal; }
        double getVarianceVal() const { return _varianceVal; }
    void setVarianceVal(double varianceVal) { _varianceVal = varianceVal; }

private:
    ModifySource _modifySource;
    double _imageVal;
    long _maskVal;
    double _varianceVal;
};

}}}

#endif
