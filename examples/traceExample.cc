#include "lsst/fw/Trace.h"

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
    using namespace Trace;
    
    setDestination(std::cout);
    
    setVerbosity(".", 100);
    work();

    setVerbosity(".", 0);
    setVerbosity("foo.bar", 3);
    setVerbosity("foo.bar.goo", 10);
    setVerbosity("foo.tar", 5);
    work();

    setVerbosity("foo.tar");
    setVerbosity("foo.bar");
    work();
    
    std::cout << "\nReset.";
    reset();
    work();

    setVerbosity("", 1);
    setVerbosity("foo.bar.goo.hoo", 10);
    work();

    setVerbosity("", 2);
    work();

    setVerbosity("");
    setVerbosity("foo.bar.goo.hoo");
    setVerbosity("foo.bar.goo.hoo.joo", 10);
    setVerbosity("foo.bar.goo", 3);
    work();
    
    return 0;
}
