#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include "lsst/fw/Citizen.h"
#include "lsst/fw/LsstBase.h"
#include "lsst/fw/Exception.h"

//
// We'll fully qualify LsstBasehere in the class definitions;
// when we come to exercise the classes we'll use lsst::fw

LSST_START_NAMESPACE(rhl)

class Shoe : private lsst::fw::LsstBase{
public:
    Shoe(int i = 0) : LsstBase(typeid(this)), _i(i) { }
    ~Shoe() { }
private:
    int _i;
};

LSST_END_NAMESPACE(rhl)

using namespace rhl;

class MyClass : private lsst::fw::LsstBase{
  public:
    MyClass(const char *typeName = 0) :
        LsstBase(typeid(this)),
        ptr(new int) {
        *ptr = 0;
    }
    int add_one() { return ++*ptr; }
private:
    boost::scoped_ptr<int> ptr;         // no need to track this alloc
};

using namespace lsst::fw;

MyClass *foo() {
    boost::scoped_ptr<Shoe> x(new Shoe(1));
    MyClass *my_instance = new MyClass();

    std::cout << "In foo\n";
    Citizen::census(std::cout);

    return my_instance;
}

Citizen::memId newCallback(const Citizen *ptr) {
    std::cout << boost::format("\tRHL Allocating memId %s\n") % ptr->repr();
    
    return 2;                           // trace every other subsequent allocs
}

Citizen::memId deleteCallback(const Citizen *ptr) {
    std::cout << boost::format("\tRHL deleting memId %s\n") % ptr->repr();
    
    return 0;
}

int main() {
#if 1
    (void)Citizen::setNewCallbackId(2);
    (void)Citizen::setDeleteCallbackId(3);
    (void)Citizen::setNewCallback(newCallback);
    (void)Citizen::setDeleteCallback(deleteCallback);
#endif
    // x isn't going to be deleted until main exists, so I don't want to see it
    // listed as a memory leak; I accordingly save the memId just _after_
    // it was allocated, and will only ask for leaks that occurred after
    // this point    
    Shoe x;
    const Citizen::memId firstId = Citizen::getNextMemId();
    
    boost::scoped_ptr<Shoe> y(new Shoe);
    boost::scoped_ptr<Shoe> z(new Shoe(10));
    
    MyClass *mine = foo();

    std::cout << boost::format("In main (%d objects)\n") % Citizen::census(0);

    boost::scoped_ptr<const std::vector<const Citizen *> > leaks(Citizen::census());
    for (std::vector<const Citizen *>::const_iterator cur = leaks->begin();
         cur != leaks->end(); cur++) {
        std::cerr << boost::format("    %s\n") % (*cur)->repr();
    }

    z.reset();                          // i.e. delete pointed-to object
    delete mine;

    ((int *)y.get())[0] = 0;            // deliberately corrupt the block
    try {
        std::cerr << "Checking corruption\n";
        (void)Citizen::checkCorruption();
    } catch(lsst::Memory &e) {
        std::cerr << "Memory check: " << e.what() <<
            "; proceeding with trepidation\n";
        ((int *)y.get())[0] = 0xdeadbeef; // uncorrupt the block
    }

    y.reset();

    std::cout << boost::format("In main (%d objects)\n") % Citizen::census(0, firstId);
    Citizen::census(std::cout, firstId);
    
    return 0;
}
