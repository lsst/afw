// -*- lsst-c++ -*-

#include <string>
#include <iostream>
#include "lsst/fw/Exception.h"

using namespace lsst;
using boost::any_cast;
using boost::shared_ptr;

int main(int argc, char *argv[])
{
// ------------- Test constructors
     {
        std::cout << "Testing creation of Data Property." << std::endl;
        DataPropertyPtr cProperty(new DataProperty("DummyProperty",(int)4));
        std::cout << ".... Data Property created." << std::endl;
        std::cout << "Testing deletion of Data Property by leaving scope." 
               << std::endl << "....Just before leaving scope" << std::endl;
     }
     std::cout << std::endl << "....Just after leaving scope" << std::endl;

     try {
        std::cout << "Testing NoMaskPlane exception handler." << std::endl;
        std::string s("exception msg: Threw a  NoMaskPlane exception");
        NoMaskPlane nmp(s);
        throw nmp;
     }
     catch(NoMaskPlane &e){
        std::cout << "....Arrived in exception handler"<<std::endl;
        std::cout << "...." <<  e.what() << std::endl;
        std::cout << "Testing deletion of Null Data Property by leaving scope." 
               << std::endl << "....Just before leaving scope" << std::endl;
     }
     std::cout << std::endl << "....Just after leaving scope" << std::endl;

     try {
        std::cout << "Testing OutOfPlane exception handler." << std::endl;
        std::string s("exception msg: Threw an OutOfPlane exception with arguments");

        // allocate the list items from the heap 
        DataPropertyPtr exceptionPropertyList(new DataProperty("root",(int)0));
        DataPropertyPtr aProperty(new DataProperty("NPlane",(int)5));
        DataPropertyPtr bProperty(new DataProperty("MaxPlane",(int)3));
        DataPropertyPtr nProperty(new DataProperty("sonofMaxPlane",(int)12));
        bProperty->addProperty(*nProperty);
        exceptionPropertyList->addProperty(*aProperty);
        exceptionPropertyList->addProperty(*bProperty);

        std::cout<< "aProperty use count: " << aProperty.use_count() << std::endl;
        std::cout<< "bProperty use count: " << bProperty.use_count() << std::endl;
        std::cout<< "nProperty use count: " << nProperty.use_count() << std::endl;

        std::cout << "....Created new propertyList"<<std::endl;
        exceptionPropertyList->print();
        OutOfPlaneSpace oops = OutOfPlaneSpace(s, exceptionPropertyList);
        std::cout << "....Created new Exception Object"<<std::endl;
        oops.print();
        std::cout << "....Just before throwing exception"<<std::endl;
        throw oops;
     }
     catch(OutOfPlaneSpace &e){
        std::cout << "....Just arrived in exception handler"<<std::endl;
        std::cout << "...." <<  e.what() << std::endl;

        // acquire the list of variables used by this exception handler

        std::cout << "....Extracting Property list from Exception Object"<<std::endl;
        DataPropertyPtr exceptionPropertyList = e.propertyList();

        exceptionPropertyList->print();

        std::cout << "....Extracting properties from PropertyList"<<std::endl;
        DataProperty *aProperty = exceptionPropertyList->find("NPlane");
        int testNPlane = any_cast<const int>(aProperty->getValue());
        DataProperty *bProperty = exceptionPropertyList->find("MaxPlane");
        int testMaxPlane = any_cast<const int>(bProperty->getValue());
        //DataProperty *cProperty = exceptionPropertyList->find("sonofMaxPlane");
        //int testSonOfMaxPlane = any_cast<const int>(cProperty->getValue());

        std::cout << "....No space to add new CR plane: number of Planes: " 
             <<  testNPlane
             << "  max Planes: " 
             <<  testMaxPlane
           //  << " sonofMaxPlane "
           //  << testSonOfMaxPlane
             << std::endl;
        //should exception be passed up the chain for more processing? 
        // throw;
        
        std::cout<< "exceptionPropertyList use count: " << exceptionPropertyList.use_count() << std::endl;
     }
     std::cout << "Left exception scoping block, was anything additional deleted?" << std::endl;

}



/*
     {
        std::cout << "Testing ~deletion of Data Property." << std::endl;
        DataPropertyPtr cProperty(new DataProperty("DummyProperty",(int)4));
        std::cout << ".... Data Property created." << std::endl;
        std::cout<< "cProperty use count: " << cProperty.use_count() << std::endl;
        //~cProperty();
        std::cout << ".... Data Property  deleted";
        std::cout<< "cProperty use count: " << cProperty.use_count() << std::endl;
     }

*/
