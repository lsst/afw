// -*- lsst-c++ -*-
#include <sstream>
#include "lsst/fw/Trace.h"
#include "lsst/fw/DataProperty.h"

using namespace lsst::fw;
using boost::any_cast;

DataProperty::DataProperty(std::string name, boost::any value) :
    fw::Citizen(typeid(this)),    
    _name(name),
    _value(value)
{
};

DataProperty::DataProperty(const DataProperty& orig) :
    fw::Citizen(typeid(this)) {
    fw::Trace("fw.DataProperty", 1,
              boost::format("copying DataProperty %s") % orig._name);
    _name = orig._name;
    _value = orig._value;
    
    if (orig._properties.size() > 0) {
        std::list<DataPropertyPtrT>::const_iterator pos;
        for (pos = orig._properties.begin(); pos != orig._properties.end(); pos++) {
            const DataPropertyPtrT origPtr = *pos;
            fw::Trace("fw.DataProperty", 1,
                      "(in loop) copying DataProperty " + origPtr->_name);
            DataPropertyPtrT dpPtr(new DataProperty(*origPtr));
            _properties.push_back(dpPtr);
        }
    }
};

// Find the next property with the given name.   If reset==true, start from the beginning.
// If reset==false, advance the pointer from the previous find, and continue the find.

DataPropertyPtrT DataProperty::find(const std::string name, bool reset) {

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

    _pos = _properties.end();
    return DataPropertyPtrT();
}


// Find the next property whose name matches the given regex .   If reset==true, start from the beginning.
// If reset==false, advance the pointer from the previous find, and continue the find.

DataPropertyPtrT DataProperty::find(const boost::regex pattern, bool reset) {
    if (reset) {
        _pos = _properties.begin();
    } else {
        _pos++;
    }

    for ( ; _pos != _properties.end(); _pos++) {
        std::string name = (*_pos)->getName();
        if (boost::regex_match(name, pattern)) {
            return *_pos;
        }
    }

    _pos = _properties.end();
    return DataPropertyPtrT();
}

// Delete the property that was found by DataProperty::find

void DataProperty::deleteFoundProperty() {
    std::list<DataPropertyPtrT>::iterator _posCopy = _pos;
    _pos--;   // ensures subsequent find with reset=false will work properly
    _properties.erase(_posCopy);
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
        sout << "(int) " << tmp;
    } else if (_value.type() == typeid(double)) {
        double tmp = any_cast<const double>(_value);
        sout << "(double) " << tmp;
    } else if (_value.type() == typeid(std::string)) {
        std::string tmp = any_cast<const std::string>(_value);
        sout << "(string) " << tmp;
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
    fw::Trace("fw.DataProperty", 1, "Destroying DataProperty " + _name);

    if (_properties.size() > 0) {
        fw::Trace("fw.DataProperty", 1, "Destroying nested property list:");
        std::list<DataPropertyPtrT>::iterator pos;
        for (pos = _properties.begin(); pos != _properties.end(); pos = _properties.erase(pos)) {
            fw::Trace("fw.DataProperty", 10,
                      boost::format("Use count prior to destructor: %d") %
                  pos->use_count());
        }
    }
}

// Free utility function

#include <sstream>

boost::any lsst::fw::stringToAny(std::string valueString)
{
    const boost::regex intRegex("(\\Q+\\E|\\Q-\\E){0,1}[0-9]+");
    const boost::regex doubleRegex("(\\Q+\\E|\\Q-\\E){0,1}[0-9]*\\.{0,1}[0-9]+((e|E)(\\Q+\\E|\\Q-\\E){0,1}[0-9]+){0,1}");
    const boost::regex FitsStringRegex("'(.*)'");

    boost::smatch matchStrings;

    std::istringstream converter(valueString);

    if (boost::regex_match(valueString, intRegex)) {
        // convert the string to an int
        int intVal;
        converter >> intVal;
        return boost::any(intVal);
    }
        
    if (boost::regex_match(valueString, doubleRegex)) {
        // convert the string to a double
        double doubleVal;
        converter >> doubleVal;
        return boost::any(doubleVal);
    }

    if (boost::regex_match(valueString, matchStrings, FitsStringRegex)) {
        // strip off the enclosing single quotes and return the string
        return boost::any(matchStrings[1].str());
    }
        
    return boost::any(valueString);
}
