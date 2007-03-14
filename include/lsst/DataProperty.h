// -*- lsst-c++ -*-
///////////////////////////////////////////////////////////
//  DataProperty.h
//  Implementation of the Class DataProperty
//  Created on:      13-Mar-07
//  Original author: Tim Axelrod
///////////////////////////////////////////////////////////

#ifndef LSST_DATAPROPERTY_H
#define LSST_DATAPROPERTY_H

#include "boost/any.hpp"
#include <string>
#include <list>
#include <iostream>

namespace lsst {

    
    class DataProperty {
    public:
        DataProperty(std::string name, boost::any value = boost::any());
        DataProperty* find(const std::string name);
        void addProperty(DataProperty &property);
        std::string getName() {return _name; }
        boost::any getValue() {return _value; }
        void print();
        
    private:
        typedef std::list<DataProperty> ContainerT;

        std::string _name;
        boost::any _value;
        ContainerT _properties;
    };

};

#endif // LSST_DATAPROPERTY_H
