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
 
#if !defined(LSST_AFW_CAMERAGEOM_CAMERASYS_H)
#define LSST_AFW_CAMERAGEOM_CAMERASYS_H

#include <string>

namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * Coordinate system and coordinate frame (if needed to disambiguate).
 */
class CameraSys {
public:
    /**
     * Construct a CameraSys
     */
    explicit CameraSys(
        std::string const &coordSys,    ///< coordinate system
        std::string const &frameName    ///< frame name, if needed to disambiguate
            ///< (typically detector name or "")
    ) : _coordSys(coordSys), _frameName(frameName) {}

    std::string getCoordSys() const { return _coordSys; }

    std::string getFrameName() const { return _frameName; }
 
private:
    std::string _coordSys;  ///< coordinate system
    std::string _frameName; ///< frame name, if needed to disambiguate (typically detector name or "")
};

}}}

#endif
