// -*- lsst-c++ -*-
#include "lsst/fw/DataProperty.h"
#include <typeinfo>

using namespace lsst;

class Foo {
     int gurp;
     std::string murp;
     double durp;
};

int main()
{
     DataProperty root("root");

     DataProperty prop1("name1", std::string("value1"));
     DataProperty prop2("name2", 2);
     DataProperty prop2a("name2", 4);
     

     root.addProperty(prop1);
     root.addProperty(prop2);

     Foo foo1;
     DataProperty prop3("name3", foo1);
     root.addProperty(prop3);

     root.addProperty(prop2a);

     DataProperty *dpPtr = root.find("name2");
     dpPtr->print();

     // check find without reset to beginning
     dpPtr = root.find("name2", false);
     dpPtr->print();

     dpPtr = root.find("name1");
     dpPtr->print();
     dpPtr = root.find("name3");
     dpPtr->print();

     // Try nested property list
     
     DataProperty nested("nested");

     DataProperty nprop1("name1n", std::string("value1"));
     DataProperty nprop2("name2n", 2);
     

     nested.addProperty(nprop1);
     nested.addProperty(nprop2);

     root.addProperty(nested);

     root.print();

}     
