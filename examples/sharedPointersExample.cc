#include <iostream>
#include "Citizen.h"
#include "Exception.h"

class Shoe : lsst::Citizen {
public:
    Shoe() {
    }
    Shoe(const char *file, int line) : lsst::Citizen(file, line) {
    }
    Shoe(const char *file, int line, int) : lsst::Citizen(file, line) {
    }
    ~Shoe() { }
};

class MyClass : lsst::Citizen {
  public:
    MyClass(const char *file, int line) :
        lsst::Citizen(file, line),
        ptr(new int) {
        *ptr = 0;
    }
    MyClass() : ptr(new int) {
        *ptr = 0;
    }
    int add_one() { return ++*ptr; }
private:
    boost::scoped_ptr<int> ptr;         // no need to track this alloc
};

using namespace lsst;

MyClass *foo() {
    SCOPED_PTR(Shoe, x, NEW(Shoe, 1));
#if 1
    MyClass *my_instance = NEW(MyClass);
#else
    MyClass *my_instance = new MyClass;
#endif

    std::cout << "In foo\n";
    Citizen::census(std::cout);

    return my_instance;
}

Citizen::memId newCallback(const Citizen *ptr) {
    std::cout << boost::format("\tRHL Allocating memId %s\n") % ptr->repr();
    
    return 1;                           // trace all subsequent allocs
}

Citizen::memId deleteCallback(const Citizen *ptr) {
    std::cout << boost::format("\tRHL Freeing memId %s\n") % ptr->repr();
    
    return 0;
}

int main() {
    (void)lsst::Citizen::setNewCallbackId(6);
    (void)lsst::Citizen::setDeleteCallbackId(3);
    (void)lsst::Citizen::setNewCallback(newCallback);
    (void)lsst::Citizen::setDeleteCallback(deleteCallback);

    SCOPED_PTR(Shoe, x, NEW(Shoe));
    SHARED_PTR(Shoe, y, new Shoe);
    Shoe *z = NEW(Shoe);

    MyClass *my_instance = foo();

    std::cout << boost::format("In main (%d objects)\n") % Citizen::census(0);

    //boost::scoped_ptr<const std::vector<const Citizen *> > leaks(Citizen::census());
    const std::vector<const Citizen *> *leaks = Citizen::census();
    for (std::vector<const Citizen *>::const_iterator cur = leaks->begin();
         cur != leaks->end(); cur++) {
        std::cerr << boost::format("    %s\n") % (*cur)->repr();
    }
    delete leaks;                       // not needed with scoped_ptr

    x.reset();
    y.reset();
    delete my_instance;

#if 1                                   // Try out the corruption detection
    ((char *)z)[0] = 0;                 // corrupt the block
    
    try {
        std::cerr << "Checking corruption\n";
        (void)Citizen::checkCorruption();
    } catch(lsst::BadAlloc &e) {
        std::cerr << "Memory check: " << e.what() << "; exiting\n";
        return 1;
    }
#endif
    
    delete z;

    std::cout << boost::format("In main (%d objects)\n") % Citizen::census(0);
    Citizen::census(std::cout);
    
    return 0;
}
