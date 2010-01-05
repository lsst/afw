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
    Id(long serial, std::string name="") : _serial(serial), _name(name) {
        if (serial < 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              (boost::format("Saw invalid serial %d; must be >= 0") % serial).str());
        }
        
    }
    Id(std::string name) : _serial(-1), _name(name) {}
    
    long getSerial() const { return _serial; }
    std::string getName() const { return _name; }
    /// Test for equality of two Ids; ignore serial if < 0 and name if == ""
    bool operator==(Id const& rhs) const {
        if (_serial >= 0 && rhs._serial >= 0) {
            bool serialEq = (_serial == rhs._serial);
            if (serialEq) {
                if (_name != "" && rhs._name != "") {
                    return _name == rhs._name;
                }
            }

            return serialEq;
        } else {
            return _name == rhs._name;
        }
    }
private:
    long _serial;
    std::string _name;
};

inline std::ostream& operator<<(std::ostream& os, Id const& id) {
    return os << "(" << id.getSerial() << ", " << id.getName() << ")";
}
}}}
#endif
