#if !defined(LSST_AFW_CAMERAGEOM_CAMERA_H)
#define LSST_AFW_CAMERAGEOM_CAMERA_H

#include <string>
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/cameraGeom/DetectorMosaic.h"

/**
 * @file
 *
 * Describe a Camera made of sets of Rafts
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * Describe a set of Detectors that are physically closely related (e.g. on the same invar support)
 */
class Camera : public DetectorMosaic {
public:
    typedef boost::shared_ptr<Camera> Ptr;
    typedef boost::shared_ptr<const Camera> ConstPtr;

    Camera(Id id,               ///< ID for Mosaic
           int const nCol,      ///< Number of columns of detectors
           int const nRow       ///< Number of rows of detectors
          )
        : DetectorMosaic(id, nCol, nRow) {}
    virtual ~Camera() {}
};

}}}

#endif
