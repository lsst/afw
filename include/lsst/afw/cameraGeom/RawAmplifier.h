/* 
 * LSST Data Management System
 * Copyright 2014 LSST Corporation.
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
 
#if !defined(LSST_AFW_CAMERAGEOM_RAWAMPLIFIER_H)
#define LSST_AFW_CAMERAGEOM_RAWAMPLIFIER_H

#include "lsst/daf/base/Citizen.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/Extent.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * Geometry and electronic information about raw amplifier images
 *
 * @note:
 * * All bounding boxes are parent boxes with respect to the raw image.
 * * The overscan and underscan bounding boxes are regions containing USABLE data,
 *   NOT the entire underscan and overscan region. These bounding boxes should exclude areas
 *   with weird electronic artifacts. Each bounding box can be empty (0 extent) if the corresponding
 *   region is not used for data processing.
 * * xyOffset is not used for instrument signature removal (ISR); it is intended for use by display
 *   utilities. It supports construction of a raw CCD image in the case that raw data is provided as
 *   individual amplifier images (which is uncommon):
 *   * Use 0,0 for cameras that supply raw data as a raw CCD image (most cameras)
 *   * Use nonzero for cameras that supply raw data as separate amplifier images with xy0=0,0 (LSST)
 */
class RawAmplifier : public lsst::daf::base::Citizen {
public:
    /** Construct a RawAmplfier */
    explicit RawAmplifier(
        geom::Box2I const &bbox,        ///< bounding box of all amplifier pixels on raw image
        geom::Box2I const &dataBBox,    ///< bounding box of amplifier image pixels on raw image
        geom::Box2I const &horizontalOverscanBBox, ///< bounding box of usable horizontal overscan pixels
            ///< on raw image; this is usually the region used for computing bias
        geom::Box2I const &verticalOverscanBBox,    ///< bounding box of usable vertical overscan pixels
            ///< on raw image
        geom::Box2I const &prescanBBox, ///< bounding box of usable (horizontal) prescan pixels on raw image
        bool flipX,                     ///< flip row order to make assembled image?
        bool flipY,                     ///< flip column order to make an assembled image?
        geom::Extent2I const &xyOffset   ///< offset for assembling a raw CCD image: desired xy0 - raw xy0
    );

    ~RawAmplifier() {}

    geom::Box2I const getBBox() { return _bbox; }
    geom::Box2I const getDataBBox() { return _dataBBox; }
    geom::Box2I const getHorizontalOverscanBBox() { return _horizontalOverscanBBox; }
    geom::Box2I const getVerticalOverscanBBox() { return _verticalOverscanBBox; }
    geom::Box2I const getPrescanBBox() { return _prescanBBox; }
    bool getFlipX() { return _flipX; }
    bool getFlipY() { return _flipY; }
    geom::Extent2I const getXYOffset() { return _xyOffset; }

private:
    geom::Box2I _bbox;
    geom::Box2I _dataBBox;
    geom::Box2I _horizontalOverscanBBox;
    geom::Box2I _verticalOverscanBBox;
    geom::Box2I _prescanBBox;
    bool _flipX;
    bool _flipY;
    geom::Extent2I _xyOffset;
};

}}}

#endif
