// -*- LSST-C++ -*-

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
 
#if !defined(LSST_AFW_IMAGE_DEFECT_H)
#define LSST_AFW_IMAGE_DEFECT_H
/**
 * A base class for image defects
 */
#include <limits>
#include <vector>
#include "lsst/afw/image/Utils.h"

namespace lsst {
namespace afw {
namespace image {
    
/**
 * \brief Encapsulate information about a bad portion of a detector
 */
class DefectBase {
public:
    typedef boost::shared_ptr<DefectBase> Ptr; //!< shared pointer to DefectBase
    
    explicit DefectBase(const BBox& bbox    //!< Bad pixels' bounding box
                   )
        : _bbox(bbox) { }
    virtual ~DefectBase() {}
    
    BBox const & getBBox() const { return _bbox; } //!< Return the Defect's bounding box
    int getX0() const { return _bbox.getX0(); }    //!< Return the Defect's left column
    void setX0(int x0) { _bbox.setX0(x0); }        //!< Set the Defect's left column
    int getX1() const { return _bbox.getX1(); }    //!< Return the Defect's right column
    void setX1(int x1) { _bbox.setX1(x1); }        //!< Set the Defect's right column
    int getY0() const { return _bbox.getY0(); }    //!< Return the Defect's bottom row
    int getY1() const { return _bbox.getY1(); }    //!< Return the Defect's top row    

    /**
     * Offset a Defect by <tt>(dx, dy)</tt>
     */
    void shift(int dx,                  //!< How much to move defect in column direction
               int dy                   //!< How much to move in row direction
              ) {
        _bbox.shift(dx, dy);
    }
private:
    BBox _bbox;                         //!< Bounding box for bad pixels
};

}}}

#endif
