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
    DataProperty::DataPropertyPtrT root(new DataProperty("root"));

     DataProperty::DataPropertyPtrT prop1(new DataProperty("name1", std::string("value1")));
     DataProperty::DataPropertyPtrT prop2(new DataProperty("name2", 2));
     DataProperty::DataPropertyPtrT prop2a(new DataProperty("name2", 4));
     

     root->addProperty(prop1);
     root->addProperty(prop2);

     Foo foo1;
     DataProperty::DataPropertyPtrT prop3(new DataProperty("name3", foo1));
     root->addProperty(prop3);

     root->addProperty(prop2a);

//      DataProperty::DataPropertyPtrT dpPtr = root->find("name2");
//      dpPtr->print();

//      // check find without reset to beginning
//      dpPtr = root->find("name2", false);
//      dpPtr->print();

//      dpPtr = root->find("name1");
//      dpPtr->print();
//      dpPtr = root->find("name3");
//      dpPtr->print();

     // Try nested property list
     
     DataProperty::DataPropertyPtrT nested(new DataProperty("nested"));

     DataProperty::DataPropertyPtrT nprop1(new DataProperty("name1n", std::string("value1")));
     DataProperty::DataPropertyPtrT nprop2(new DataProperty("name2n", 2));
     

     nested->addProperty(nprop1);
     nested->addProperty(nprop2);

     root->addProperty(nested);

     root->print();

     // Check copy constructor

     DataProperty::DataPropertyPtrT rootCopy(new DataProperty(*root));

     // Explicitly destroy root

     root.~shared_ptr<DataProperty>();

     std::cout << "Explicit destruction done" << std::endl;

     // Check that rootCopy is still OK...

     rootCopy->print();
     
}     
