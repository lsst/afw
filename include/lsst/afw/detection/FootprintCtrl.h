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
