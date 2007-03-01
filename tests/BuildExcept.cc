// -*- lsst-c++ -*-

#include <string>
#include <iostream>
#include "lsst/MaskExceptions.h"

using namespace lsst;

int main(int argc, char *argv[])
{
// ------------- Test constructors
     try {
        std::string s("Threw an exception");
        const int testNPlane = 2;
        const int testMaxPlane = 3;

        OutOfPlaneSpace oops(s, testNPlane,  testMaxPlane);
        throw oops;
     }
     catch(OutOfPlaneSpace &e){
        std::cout <<  e.what() << std::endl;
        std::cout << "Ran out of space to add new CR plane: number of Planes: " 
             << e.nPlane() << "  max Planes: " << e.maxPlane() << std::endl;
        throw;
     }

}

/* 
g++ -Wall -g -DNDEBUG  -I../include -I../src -I/usr/include/boost -c BuildExcept.cc

g++ -Wall -g -DNDEBUG  -I../include -I../src -I/usr/include/boost -c ../src/MaskExceptions.cc

g++ -o BuildExcept -Wall -g -DNDEBUG  MaskExceptions.o BuildExcept.o MaskExceptions.o

BuildExcept.o: In function `main':/home/robyn/LsstDC2/fw/scons/tests/BuildExcept.cc:17: undefined reference to 
`lsst::OutOfPlaneSpace::OutOfPlaneSpace(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int, int)'
:/home/robyn/LsstDC2/fw/scons/tests/BuildExcept.cc:18: undefined reference to 
`lsst::OutOfPlaneSpace::OutOfPlaneSpace(lsst::OutOfPlaneSpace const&)'

collect2: ld returned 1 exit status

make: *** [BuildExcept] Error 1

robyn@gumtree.tuc.noao.edu%
*/

