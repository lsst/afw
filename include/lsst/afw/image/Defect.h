// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#if !defined(LSST_AFW_IMAGE_DEFECT_H)
#define LSST_AFW_IMAGE_DEFECT_H
/**
 * A base class for image defects
 */
#include <limits>
#include <vector>
#include "lsst/afw/geom.h"
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
    
    explicit DefectBase(
        const geom::Box2I & bbox    //!< Bad pixels' bounding box
    ) : _bbox(bbox) { }
    virtual ~DefectBase() {}
    
    geom::Box2I const & getBBox() const { return _bbox; } //!< Return the Defect's bounding box
    int getX0() const { return _bbox.getMinX(); }    //!< Return the Defect's left column
    int getX1() const { return _bbox.getMaxX(); }    //!< Return the Defect's right column
    int getY0() const { return _bbox.getMinY(); }    //!< Return the Defect's bottom row
    int getY1() const { return _bbox.getMaxY(); }    //!< Return the Defect's top row    
    

    void clip(geom::Box2I const & bbox) {_bbox.clip(bbox);}

    /**
     * Offset a Defect by <tt>(dx, dy)</tt>
     */
    void shift(int dx,                  //!< How much to move defect in column direction
               int dy                   //!< How much to move in row direction
              ) {
        _bbox.shift(geom::Extent2I(dx, dy));
    }
    void shift(geom::Extent2I const & d) {_bbox.shift(d);}
private:
    geom::Box2I _bbox;                         //!< Bounding box for bad pixels
};

}}}

#endif
