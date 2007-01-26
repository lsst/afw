#include <iostream>
#include "sharedPointers.h"

class Shoe : lsst::citizen {
public:
    Shoe() {
    }
    Shoe(const char *file, int line) : lsst::citizen(file, line) {
    }
    Shoe(const char *file, int line, int) : lsst::citizen(file, line) {
    }
    ~Shoe() { }
};

class MyClass : lsst::citizen {
  public:
    MyClass(const char *file, int line) :
        lsst::citizen(file, line),
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
    lsst::SCOPED_PTR(Shoe, x, NEW(Shoe, 1));
#if 1
    MyClass *my_instance = NEW(MyClass);
#else
    MyClass *my_instance = new MyClass;
#endif

    std::cout << "In foo\n";
    citizen::census(std::cout);

    return my_instance;
}

citizen::memId newCallback(const citizen *ptr) {
    std::cout << boost::format("\tRHL Allocating memId %s\n") % ptr->repr();
    
    return 1;                           // trace all subsequent allocs
}

citizen::memId deleteCallback(const citizen *ptr) {
    std::cout << boost::format("\tRHL Freeing memId %s\n") % ptr->repr();
    
    return 0;
}

int main() {
    (void)lsst::citizen::setNewCallbackId(6);
    (void)lsst::citizen::setDeleteCallbackId(3);
    (void)lsst::citizen::setNewCallback(newCallback);
    (void)lsst::citizen::setDeleteCallback(deleteCallback);

    lsst::SCOPED_PTR(Shoe, x, NEW(Shoe));
    lsst::SCOPED_PTR(Shoe, y, new Shoe);
    Shoe *z = NEW(Shoe);

    MyClass *my_instance = foo();

    std::cout << boost::format("In main (%d objects)\n") % citizen::census(0);

    //boost::scoped_ptr<const std::vector<const citizen *> > leaks(citizen::census());
    const std::vector<const citizen *> *leaks = citizen::census();
    for (std::vector<const citizen *>::const_iterator cur = leaks->begin();
         cur != leaks->end(); cur++) {
        std::cerr << boost::format("    %s\n") % (*cur)->repr();
    }
    delete leaks;

    x.reset();
    y.reset();
    delete my_instance;
    delete z;

    std::cout << boost::format("In main (%d objects)\n") % citizen::census(0);
    citizen::census(std::cout);
    
    return 0;
}
