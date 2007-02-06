#include <iostream>
#include "Exception.h"

int main(int ac, char **av) {
    try {
        throw lsst::Memory("Hello World");
    } catch(lsst::Exception &e) {
        std::cerr << "Caught exception: " << e.what() << "\n";
    }

    try {
        throw lsst::Memory(boost::format("(This is an boost::format) %s") % "Goodbye World");
    } catch(lsst::Exception &e) {
        std::cerr << "Caught exception: " << e.what() << "\n";
    }

    return 0;
}
