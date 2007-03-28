#include "lsst/Trace.h"

using namespace lsst::fw;

static void work() {
    std::cout << "\nVerbosity levels:\n";
    Trace::printVerbosity(std::cout);
    std::cout << "Traces:\n";

    Trace::trace("foo", 1, "foo 1");
    Trace::trace("foo.bar", 2, boost::format("foo.bar %d") % 2);
    Trace::trace("foo.bar.goo", 4, "foo.bar.goo 4");
    Trace::trace("foo.bar.goo.hoo", 3, boost::format("foo.bar.goo.hoo %d") % 3);

    Trace::trace("foo.tar", 5, "foo.tar 5");
}

int main() {
    Trace::setDestination(std::cout);
    
    Trace::setVerbosity(".", 100);
    work();

    Trace::setVerbosity(".", 0);
    Trace::setVerbosity("foo.bar", 3);
    Trace::setVerbosity("foo.bar.goo", 10);
    Trace::setVerbosity("foo.tar", 5);
    work();

    Trace::setVerbosity("foo.tar");
    Trace::setVerbosity("foo.bar");
    work();
    
    std::cout << "\nReset.";
    Trace::reset();
    work();

    Trace::setVerbosity("", 1);
    Trace::setVerbosity("foo.bar.goo.hoo", 10);
    work();

    Trace::setVerbosity("", 2);
    work();

    Trace::setVerbosity("");
    Trace::setVerbosity("foo.bar.goo.hoo");
    Trace::setVerbosity("foo.bar.goo.hoo.joo", 10);
    Trace::setVerbosity("foo.bar.goo", 3);
    work();
    
    return 0;
}
