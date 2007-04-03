// -*- lsst-c++ -*-
#include <sstream>
#include "lsst/fw/Trace.h"
#include "lsst/fw/DataProperty.h"

using namespace lsst;
using boost::any_cast;

DataProperty::DataProperty(std::string name, boost::any value):
    _name(name),
    _value(value)
{
};

DataProperty::DataProperty(const DataProperty& orig)
{
    using lsst::fw::Trace::trace;
    trace("fw.DataProperty", 1,
          boost::format("copying DataProperty %s") % orig._name);
    _name = orig._name;
    _value = orig._value;
    
    if (orig._properties.size() > 0) {
        std::list<DataPropertyPtrT>::const_iterator pos;
        for (pos = orig._properties.begin(); pos != orig._properties.end(); pos++) {
            const DataPropertyPtrT origPtr = *pos;
            DataPropertyPtrT dpPtr(new DataProperty(*origPtr));
            _properties.push_back(dpPtr);
        }
    }
};

DataProperty::DataPropertyPtrT DataProperty::find(const std::string name, bool reset) {

    if (reset) {
        pos = _properties.begin();
    }

    for ( ; pos != _properties.end(); pos++) {
        if ((*pos)->getName() == name) {
            return *pos;
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


std::string DataProperty::repr() const {
    std::ostringstream sout;
    sout << _name;
    if (_value.type() == typeid(int)) {
        int tmp = any_cast<const int>(_value);
        sout << " " << tmp;
    } else if (_value.type() == typeid(std::string)) {
        std::string tmp = any_cast<const std::string>(_value);
        sout << " " << tmp;
    }
    sout << std::endl;

    if (_properties.size() > 0) {
        sout << "Nested property list: " << std::endl;
        ContainerT::const_iterator pos;
        for (pos = _properties.begin(); pos != _properties.end(); pos++) {
            sout << (*pos)->repr();
        }
    }

    return sout.str();
}

void DataProperty::print() const {
    std::cout << repr();
}

DataProperty::~DataProperty() {
    using lsst::fw::Trace::trace;

    trace("fw.DataProperty", 1,
          boost::format("Destroying DataProperty %s") % _name);

    if (_properties.size() > 0) {
        trace("fw.DataProperty", 1,
              boost::format("Destroying nested property list: "));
        std::list<DataPropertyPtrT>::iterator pos;
        for (pos = _properties.begin(); pos != _properties.end(); pos++) {
            DataPropertyPtrT dpPtr = *pos;
            trace("fw.DataProperty", 1,
                  boost::format("Use count prior to destructor: %d") %
                  dpPtr.use_count());
            trace("fw.DataProperty", 1,
                  boost::format("(in loop) Destroying DataProperty %s") %
                  dpPtr->getName());
            dpPtr.~DataPropertyPtrT();
        trace("fw.DataProperty", 1,
              boost::format("Use count after destructor: %s") %
              dpPtr.use_count());
        }
    }
}



