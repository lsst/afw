#include "lsst/Trace.h"

using namespace lsst::utils;

int main() {
    Trace::setVerbosity("", 1);
    Trace::setVerbosity("foo", 2);
    Trace::setVerbosity("foo.bar", 3);
    Trace::setVerbosity("foo.bar.goo", 10);

    std::cout << "Try some traces" << "\n";
    Trace::printVerbosity(std::cout);
    Trace::trace("foo", 2, "Hello world");
    Trace::trace("foo.bar", 2, boost::format("bar %d") % 10);
    Trace::trace("foo.bar.goo", 2, boost::format("goo %d") % 20);
    Trace::trace("foo.bar.goo.hoo", 4, boost::format("goo %d") % 30);

    Trace::reset();

    std::cout << "Reset; some more traces" << "\n";
    Trace::printVerbosity(std::cout);
    Trace::trace("foo", 1, "Hello world");

    Trace::setVerbosity("", 1);

    std::cout << "Try some more traces" << "\n";
    Trace::printVerbosity(std::cout);
    Trace::trace("foo", 1, "Hello world");
    Trace::trace("foo", 2, "Hello world II");
    Trace::trace("foo.bar", 3, boost::format("bar x%d") % 10);
    Trace::trace("foo.bar.goo", 2, boost::format("goo x%d") % 20);
    Trace::trace("foo.bar.goo.hoo", 4, boost::format("goo x%d") % 20);

    return 0;
}
