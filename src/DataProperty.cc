// -*- lsst-c++ -*-
#include <sstream>
#include "lsst/fw/Trace.h"
#include "lsst/fw/DataProperty.h"

using namespace lsst;
using boost::any_cast;

DataProperty::DataProperty(std::string name, boost::any value) :
    fw::Citizen(typeid(this)),    
    _name(name),
    _value(value)
{
};

DataProperty::DataProperty(const DataProperty& orig) :
    fw::Citizen(typeid(this)) {
    using lsst::fw::Trace::trace;
    trace("fw.DataProperty", 1,
          boost::format("copying DataProperty %s") % orig._name);
    _name = orig._name;
    _value = orig._value;
    
    if (orig._properties.size() > 0) {
        std::list<DataPropertyPtrT>::const_iterator pos;
        for (pos = orig._properties.begin(); pos != orig._properties.end(); pos++) {
            const DataPropertyPtrT origPtr = *pos;
            trace("fw.DataProperty", 2,
                  "(in loop) copying DataProperty " + origPtr->_name);
            DataPropertyPtrT dpPtr(new DataProperty(*origPtr));
            _properties.push_back(dpPtr);
        }
    }
};

// Find the property with the given name.   If reset==true, start from the beginning.
// If reset==false, advance the pointer from the previous find, and continue the find.

DataProperty::DataPropertyPtrT DataProperty::find(const std::string name, bool reset) {

    if (reset) {
        _pos = _properties.begin();
    } else {
        _pos++;
    }

    for ( ; _pos != _properties.end(); _pos++) {
        if ((*_pos)->getName() == name) {
            return *_pos;
        }
    }
    
    return DataPropertyPtrT();
}


DataProperty::DataPropertyPtrT DataProperty::find(const boost::regex pattern, bool reset) {
    return DataPropertyPtrT();
}

void DataProperty::addProperty(DataPropertyPtrT propertyPtr) {
    _properties.push_back(propertyPtr);
}

void DataProperty::addProperty(const DataProperty& dp) {
    DataPropertyPtrT dpPtr(new DataProperty(dp));
    addProperty(dpPtr);
}

std::string DataProperty::repr(const std::string& prefix) const {
    std::ostringstream sout;
    sout << prefix << _name;
    if (_value.type() == typeid(int)) {
        int tmp = any_cast<const int>(_value);
        sout << " " << tmp;
    } else if (_value.type() == typeid(std::string)) {
        std::string tmp = any_cast<const std::string>(_value);
        sout << " " << tmp;
    }
    sout << std::endl;

    if (_properties.size() > 0) {
        sout << prefix << "Nested property list: " << std::endl;
        DataPropertyContainerT::const_iterator pos;
        for (pos = _properties.begin(); pos != _properties.end(); pos++) {
            sout << prefix << (*pos)->repr(prefix);
        }
    }

    return sout.str();
}

void DataProperty::print(const std::string& prefix) const {
    std::cout << repr(prefix);
}

DataProperty::~DataProperty() {
    using lsst::fw::Trace::trace;

    trace("fw.DataProperty", 1, "Destroying DataProperty " + _name);
#if 0
    if (_properties.size() > 0) {
        trace("fw.DataProperty", 1, "Destroying nested property list:");
        std::list<DataPropertyPtrT>::iterator pos;
        for (pos = _properties.begin(); pos != _properties.end(); pos++) {
            DataPropertyPtrT dpPtr = *pos;
            trace("fw.DataProperty", 10,
                  boost::format("Use count prior to destructor: %d") %
                  dpPtr.use_count());

            trace("fw.DataProperty", 2,
                  "(in loop) Destroying DataProperty " + dpPtr->getName());
            dpPtr.~DataPropertyPtrT();
            trace("fw.DataProperty", 10,
                  boost::format("Use count after destructor: %d") %
                  dpPtr.use_count());
        }
    }
#endif
}
