#include <iostream>
#include "Exception.h"

int main(int ac, char **av) {
    try {
        throw lsst::Exception(boost::format("This is an exception: %s") % "Hello World");
    } catch(lsst::Exception &e) {
        std::cerr << "Caught exception: " << e.getMsg() << "\n";
    }

    return 0;
}
