// -*- lsst-c++ -*-
#include "lsst/fw/DataProperty.h"
#include <lsst/fw/Citizen.h>
#include <lsst/fw/Trace.h>

using namespace std;
using namespace lsst::fw;

class Foo {
     int gurp;
     std::string murp;
     double durp;
};

void test() {
    DataPropertyPtrT root(new DataProperty("root"));

     DataPropertyPtrT prop1(new DataProperty("name1", std::string("value1")));
     DataPropertyPtrT prop2(new DataProperty("name2", 2));
     DataPropertyPtrT prop2a(new DataProperty("name2", 4));
     

     root->addProperty(prop1);
     root->addProperty(prop2);

     Foo foo1;
     DataPropertyPtrT prop3(new DataProperty("name3", foo1));
     root->addProperty(prop3);

     root->addProperty(prop2a);

//      DataPropertyPtrT dpPtr = root->find("name2");
//      dpPtr->print("\t");

//      // check find without reset to beginning
//      dpPtr = root->find("name2", false);
//      dpPtr->print("\t");

//      dpPtr = root->find("name1");
//      dpPtr->print("\t");
//      dpPtr = root->find("name3");
//      dpPtr->print("\t");

     // Try nested property list
     
     DataPropertyPtrT nested(new DataProperty("nested"));

     DataPropertyPtrT nprop1(new DataProperty("name1n", std::string("value1")));
     DataPropertyPtrT nprop2(new DataProperty("name2n", 2));
     

     nested->addProperty(nprop1);
     nested->addProperty(nprop2);

     root->addProperty(nested);

     root->print("\t");

     // Check copy constructor

     DataPropertyPtrT rootCopy(new DataProperty(*root));

     // Explicitly destroy root

     root.reset();

     std::cout << "Explicit destruction done" << std::endl;

     // Check that rootCopy is still OK...

     rootCopy->print("\t");
     
}     

void test2()
{
    boost::any foo = stringToAny("-1234");
    boost::any foo2 = stringToAny("1.234e-1");
    boost::any foo3 = stringToAny("'This is a Fits string'");
    
    DataPropertyPtrT fooProp(new DataProperty("foo", foo));
    DataPropertyPtrT fooProp2(new DataProperty("foo2", foo2));
    DataPropertyPtrT fooProp3(new DataProperty("foo3", foo3));
    fooProp->print();
    fooProp2->print();
    fooProp3->print();
}

int main() {
    Trace::setVerbosity("fw.DataProperty", 10);

    test();
     //
     // Check for memory leaks
     //
     if (Citizen::census(0) == 0) {
         cerr << "No leaks detected" << endl;
     } else {
         cerr << "Leaked memory blocks:" << endl;
         Citizen::census(cerr);
     }

     test2();
}
