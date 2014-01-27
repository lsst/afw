#include "lsst/afw/cameraGeom/Amplifier.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

    Amplifier::Amplifier(
        std::string const &name,
        geom::Box2I const &bbox,
        double gain,
        double readNoise,
        CONST_PTR(RawAmplifier) rawAmplifierPtr
    ) :
        Citizen(typeid(this)),
        _name(name),
        _bbox(bbox),
        _gain(gain),
        _readNoise(readNoise),
        _rawAmplifierPtr(rawAmplifierPtr)
    {}

}}}
