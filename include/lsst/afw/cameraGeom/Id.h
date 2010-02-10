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
    Id(long serial=-1, std::string name="") : _serial(serial), _name(name) {}
    Id(std::string name) : _serial(-1), _name(name) {}
    
    long getSerial() const { return _serial; }
    std::string getName() const { return _name; }
    bool operator==(Id const& rhs) const;
    bool operator<(Id const& rhs) const;
private:
    long _serial;
    std::string _name;
};

inline std::ostream& operator<<(std::ostream& os, Id const& id) {
    return os << "(" << id.getSerial() << ", " << id.getName() << ")";
}
}}}
#endif
