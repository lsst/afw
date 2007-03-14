// -*- lsst-c++ -*-
#include "lsst/DataProperty.h"

using namespace lsst;
using boost::any_cast;

DataProperty::DataProperty(std::string name, boost::any value):
    _name(name),
    _value(value)
{
};

DataProperty* DataProperty::find(const std::string name) {

    std::list<DataProperty>::iterator pos;
    for (pos = _properties.begin(); pos != _properties.end(); pos++) {
        if (pos->getName() == name) {
            return &(*pos);
        }
    }
    
    return NULL;
}

void DataProperty::addProperty(DataProperty &property) {
    _properties.push_back(property);
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
        std::list<DataProperty>::iterator pos;
        for (pos = _properties.begin(); pos != _properties.end(); pos++) {
            pos->print();
        }
    }

          
}
