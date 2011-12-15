

#include "boost/make_shared.hpp"

#include "lsst/afw/table/IdFactory.h"

namespace lsst { namespace afw { namespace table {

namespace {

class SimpleIdFactory : public IdFactory {
public:

    virtual RecordId operator()() { return ++_current; }

    virtual void notify(RecordId id) { _current = id; }

    virtual PTR(IdFactory) clone() const { return boost::make_shared<SimpleIdFactory>(*this); }

    SimpleIdFactory() : _current(0) {}

private:
    RecordId _current;
};

} // anonymous

PTR(IdFactory) IdFactory::makeSimple() {
    return boost::make_shared<SimpleIdFactory>();
}

}}} // namespace lsst::afw::table
