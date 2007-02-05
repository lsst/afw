#include <iostream>
#include "Exception.h"

int main(int ac, char **av) {
    try {
        throw lsst::BadAlloc(boost::format("This is an exception: %s") % "Hello World");
    } catch(lsst::Exception &e) {
        std::cerr << "Caught exception: " << e.what() << "\n";
    }

    return 0;
}
