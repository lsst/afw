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

#if !defined(LSST_AFW_CAMERAGEOM_AMPLIFIER_H)
#define LSST_AFW_CAMERAGEOM_AMPLIFIER_H

#include <string>
#include <sstream>
#include "lsst/base.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/cameraGeom/RawAmplifier.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * Information about the amplifier region of an assembled image.
 *
 * Optionally contains information about the raw amplifier, as well.
 * This is usually only omitted if data processing begins with data that has already had
 * the instrument signature removed (e.g. the way LSST processes SDSS data).
 */
class Amplifier : public lsst::daf::base::Citizen {
public:
    explicit Amplifier(
        std::string const &name,    ///< name of amplifier location in camera
        geom::Box2I const &bbox,    ///< bounding box of amplifier pixels in assembled image
        double gain,                ///< amplifier gain in e-/ADU
        double readNoise,           ///< amplifier read noise, in e-
        CONST_PTR(RawAmplifier) rawAmplifierPtr ///< data about raw amplifier image, if known and relevant
    );

    std::string const getName() { return _name; }
    geom::Box2I const getBBox() { return _bbox; }
    double getGain() { return _gain; }
    double getReadNoise() { return _readNoise; }
    CONST_PTR(RawAmplifier) getRawAmplifierPtr() { return _rawAmplifierPtr; }
    bool hasRawAmplifierPtr() { return bool(_rawAmplifierPtr); }

private:
    std::string _name;
    geom::Box2I _bbox;
    double _gain;
    double _readNoise;
    CONST_PTR(RawAmplifier) _rawAmplifierPtr;
};

}}}

#endif
