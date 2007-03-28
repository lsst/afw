// -*- lsst-c++ -*-
#include <iostream>
#include <typeinfo>
#include "lsst/fw/DataProperty.h"


#include <boost/shared_ptr.hpp>
typedef boost::shared_ptr<lsst::DataProperty> DataPropertyPtr;

using namespace lsst;

class Foo {
     int gurp;
     std::string murp;
     double durp;
};

int main()
{
     DataPropertyPtr root(new DataProperty("root"));
     std::cout<< "Initial ROOT use count: " << root.use_count() << std::endl;

     DataPropertyPtr prop1(new DataProperty("name1", std::string("value1")));
     DataPropertyPtr prop2(new DataProperty("name2", 2));
     DataPropertyPtr prop2a(new DataProperty("name2", 4));

     root->addProperty(*prop1);
     root->addProperty(*prop2);

     Foo foo1;
     DataPropertyPtr prop3(new DataProperty("name3", foo1));
     root->addProperty(*prop3);

     root->addProperty(*prop2a);

     DataProperty *dpPtr = root->find("name2");
     dpPtr->print();

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
     
        DataPropertyPtr nested(new DataProperty("nested"));

        DataPropertyPtr nprop1(new DataProperty("name1n", std::string("value1")));
        DataPropertyPtr nprop2(new DataProperty("name2n", 2));
     

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
        DataPropertyPtr secondroot = root;
        std::cout<< "Entered new block; test root copy"<< std::endl<< "ROOT use count: " << root.use_count() << std::endl;
        std::cout<< "SECONDROOT use count: " << secondroot.use_count() << std::endl;
        }
     std::cout<< "===========================================" << std::endl;
     std::cout<< "Exited block; ROOT use count: " << root.use_count() << std::endl;
     
     //std::cout<< "NESTED use count: " << nested.use_count() << std::endl;

     std::cout << "Exiting the code" << std::endl;

}     
