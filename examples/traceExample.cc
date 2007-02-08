#include "Trace.h"

using namespace lsst::utils;

int main() {
    Trace::setVerbosity("foo", 1);
    Trace::setVerbosity("foo.bar", 2);
    Trace::setVerbosity("foo.bar.goo", 10);

    Trace::printVerbosity();

    Trace::trace("foo", 2, "Hello world");
    Trace::trace("foo.bar", 2, boost::format("bar x%d") % 10);
    Trace::trace("foo.bar.goo", 2, boost::format("goo x%d") % 20);
    Trace::trace("foo.bar.goo.hoo", 4, boost::format("goo x%d") % 20);

    return 0;
}
