#include "lsst/afw/cameraGeom/RawAmplifier.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

    RawAmplifier::RawAmplifier(
        geom::Box2I const &bbox,
        geom::Box2I const &dataBBox,
        geom::Box2I const &horizontalOverscanBBox,
        geom::Box2I const &verticalOverscanBBox,
        geom::Box2I const &prescanBBox,
        bool flipX,
        bool flipY,
        geom::Extent2I const &xyOffset
    ) :
        Citizen(typeid(this)),
        _bbox(bbox),
        _dataBBox(dataBBox),
        _horizontalOverscanBBox(horizontalOverscanBBox),
        _verticalOverscanBBox(verticalOverscanBBox),
        _prescanBBox(prescanBBox),
        _flipX(flipX),
        _flipY(flipY),
        _xyOffset(xyOffset)
    {}

}}}
