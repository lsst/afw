// -*- lsst-c++ -*-
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
    std::cout << "copying DataProperty " << orig._name << std::endl;
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


void DataProperty::print() {
    std::cout << _name;
    if (_value.type() == typeid(int)) {
        int tmp = any_cast<const int>(_value);
        std::cout << " " << tmp;
    } else if (_value.type() == typeid(std::string)) {
        std::string tmp = any_cast<const std::string>(_value);
        std::cout << " " << tmp;
    }
    std::cout << std::endl;

    if (_properties.size() > 0) {
        std::cout << "Nested property list: " << std::endl;
        std::list<DataPropertyPtrT>::iterator pos;
        for (pos = _properties.begin(); pos != _properties.end(); pos++) {
            (*pos)->print();
        }
    }

}


DataProperty::~DataProperty() {
    std::cout << "Destroying DataProperty " << _name << std::endl;
    if (_properties.size() > 0) {
        std::cout << "Destroying nested property list: " << std::endl;
        std::list<DataPropertyPtrT>::iterator pos;
        for (pos = _properties.begin(); pos != _properties.end(); pos++) {
            DataPropertyPtrT dpPtr = *pos;
            std::cout << "Use count prior to destructor: " << dpPtr.use_count() << std::endl;
            std::cout << "(in loop) Destroying DataProperty " << dpPtr->getName() << std::endl;
            dpPtr.~DataPropertyPtrT();
            std::cout << "Use count after destructor: " << dpPtr.use_count() << std::endl;
        }
    }
}



