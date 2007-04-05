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
#include "boost/shared_ptr.hpp"
#include "boost/regex.hpp"
#include <string>
#include <list>
#include <iostream>


namespace lsst {

    class DataProperty {
    public:
        typedef boost::shared_ptr<DataProperty> DataPropertyPtrT;
        typedef std::list<DataPropertyPtrT> DataPropertyContainerT;

        DataProperty(std::string name, boost::any value = boost::any());
        DataProperty(const DataProperty& orig);
        DataPropertyPtrT find(const std::string name, bool reset=true);
        DataPropertyPtrT find(const boost::regex pattern, bool reset=true);
        void addProperty(DataPropertyPtrT property);
        std::string getName() const {return _name; }
        boost::any getValue() const {return _value; }
        DataPropertyContainerT getContents() const {return _properties; }
        std::string repr() const;
        void print() const;
        ~DataProperty();
        
    private:
        std::string _name;
        boost::any _value;
        DataPropertyContainerT _properties;
        std::list<DataPropertyPtrT>::iterator _pos;
    };

};

#endif // LSST_DATAPROPERTY_H
