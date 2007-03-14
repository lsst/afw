// -*- lsst-c++ -*-
#include "lsst/DataProperty.h"
#include <typeinfo>

using namespace lsst;

int main()
{
     DataProperty root("root");

     DataProperty prop1("name1", std::string("value1"));
     DataProperty prop2("name2", 2);

     root.addProperty(prop1);
     root.addProperty(prop2);

     DataProperty *dpPtr = root.find("name2");
     dpPtr->print();
}     
