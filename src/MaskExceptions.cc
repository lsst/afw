// -*- lsst-c++ -*-
// Implementations of Mask Exceptions class methods

#include  "lsst/MaskExceptions.h"

using namespace lsst;

const char* 
NoMaskPlane::what() throw() {
    return "NoMaskPlane exception: ";
}

OutOfPlaneSpace::OutOfPlaneSpace(void) throw():  
    std::exception(), 
    _message("Out of Plane Space"),
    _nPlane(-1), 
    _maxPlane(-1) {
}

OutOfPlaneSpace::OutOfPlaneSpace(const OutOfPlaneSpace& oops) throw():
    std::exception(oops), 
    _message(oops._message), 
    _nPlane(oops._nPlane), 
    _maxPlane(oops._maxPlane) {
}

OutOfPlaneSpace::OutOfPlaneSpace(std::string const& message, const int nPlane, \
    const int maxPlane) throw(): 
    std::exception(), 
    _message(message), 
    _nPlane(nPlane), 
    _maxPlane(maxPlane) {
}

OutOfPlaneSpace::~OutOfPlaneSpace() throw() {
}

OutOfPlaneSpace::OutOfPlaneSpace& OutOfPlaneSpace::OutOfPlaneSpace::operator= (const OutOfPlaneSpace& oops) throw() {
    std::exception::operator= (oops);
    _message=oops._message;
    _nPlane=oops._nPlane;
    _maxPlane=oops._maxPlane;
    return *this;
}

const char* 
OutOfPlaneSpace::what() throw() {
    return _message.c_str();
}

const int 
OutOfPlaneSpace::nPlane() throw() {
    return _nPlane;
}

const int 
OutOfPlaneSpace::maxPlane() throw() {
    return _maxPlane;
}

