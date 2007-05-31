// -*- lsst-c++ -*-
#include <iostream>
#include <typeinfo>
#include "lsst/fw/DataProperty.h"


#include <boost/shared_ptr.hpp>

using namespace lsst::fw;

class Foo {
     int gurp;
     std::string murp;
     double durp;
};

int main()
{
     DataPropertyPtrT root(new DataProperty("root"));
     std::cout<< "Initial ROOT use count: " << root.use_count() << std::endl;

     DataPropertyPtrT prop1(new DataProperty("name1", std::string("value1")));
     DataPropertyPtrT prop2(new DataProperty("name2", 2));
     DataPropertyPtrT prop2a(new DataProperty("name2", 4));

     root->addProperty(*prop1);
     root->addProperty(*prop2);

     Foo foo1;
     DataPropertyPtrT prop3(new DataProperty("name3", foo1));
     root->addProperty(*prop3);

     root->addProperty(*prop2a);
     DataPropertyPtrT dpPtr;

//      dpPtr = root->find("name2");
//      dpPtr->print();

     // check find without reset to beginning
     dpPtr = root->find("name2", false);
     dpPtr->print();

     dpPtr = root->find("name1");
     dpPtr->print();
     dpPtr = root->find("name3");
     dpPtr->print();

     {
        std::cout<< "===========================================" << std::endl;
        std::cout<< "Entered new block; test nested property add:  incoming root use: " << root.use_count() << std::endl;
        // Try nested property list
     
        DataPropertyPtrT nested(new DataProperty("nested"));

        DataPropertyPtrT nprop1(new DataProperty("name1n", std::string("value1")));
        DataPropertyPtrT nprop2(new DataProperty("name2n", 2));
     

        nested->addProperty(*nprop1);
        nested->addProperty(*nprop2);

        root->addProperty(*nested);
        std::cout<< "Nested property added; root use: " << root.use_count() << std::endl;

        std::cout << "Root print" << std::endl;
        root->print();
        std::cout << "Exiting new block" << std::endl;
    }
    std::cout << "Exited block, root print should NOT have nested props" << std::endl;
    root->print();

        { 
        std::cout<< "===========================================" << std::endl;
        DataPropertyPtrT secondroot = root;
        std::cout<< "Entered new block; test root copy"<< std::endl<< "ROOT use count: " << root.use_count() << std::endl;
        std::cout<< "SECONDROOT use count: " << secondroot.use_count() << std::endl;
        }
     std::cout<< "===========================================" << std::endl;
     std::cout<< "Exited block; ROOT use count: " << root.use_count() << std::endl;
     
     //std::cout<< "NESTED use count: " << nested.use_count() << std::endl;

     std::cout << "Exiting the code" << std::endl;

}     
