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
 
#if !defined(LSST_AFW_CAMERAGEOM_ID_H)
#define LSST_AFW_CAMERAGEOM_ID_H

#include <string>
#include "boost/format.hpp"
#include "lsst/pex/exceptions.h"

/**
 * @file
 *
 * Describe the physical layout of pixels in the focal plane
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * An ID for something; a unique serial number and a name.  Either may be omitted.
 *
 * @note Serial numbers must be non-negative
 */
class Id {
public:
    Id(long serial=-1, std::string name="", int ix=0, int iy=0) :
        _serial(serial), _name(name), _ix(ix), _iy(iy) {}
    Id(std::string name) : _serial(-1), _name(name) {}
    
    long getSerial() const { return _serial; }
    std::string getName() const { return _name; }
    std::pair<int, int> getIndex() const { return std::make_pair(_ix, _iy); }

    bool operator==(Id const& rhs) const;
    bool operator<(Id const& rhs) const;
private:
    long _serial;                       // unique serial number
    std::string _name;                  // name for device
    int _ix, _iy;                       // two ints to identify Device in its parent;  maybe indexes?
};

inline std::ostream& operator<<(std::ostream& os, Id const& id) {
    return os << "(" << id.getSerial() << ", " << id.getName() << ")";
}
}}}
#endif
