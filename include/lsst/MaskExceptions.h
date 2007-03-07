// -*- lsst-c++ -*-
///////////////////////////////////////////////////////////
//  MaskExceptions.h
//  Implementation of the Class MaskExceptions
//  Created on:      28-Feb-2007 
//  Original author: Roberta Allsman
///////////////////////////////////////////////////////////

#ifndef LSST_MASK_EXCEPTIONS_H
#define LSST_MASK_EXCEPTIONS_H

#include <string>
#include <stdexcept>

namespace lsst {

    class NoMaskPlane: public std::exception {
    public:
        const char* what() throw();
    };
    
    class OutOfPlaneSpace: public std::exception{
    private:
        std::string _message;
        int _nPlane;
        int _maxPlane;
    public:
        OutOfPlaneSpace(void) throw();
        OutOfPlaneSpace(const OutOfPlaneSpace& ) throw();
        OutOfPlaneSpace(std::string const&, const int nPlane, \
                        const int maxPlane) throw();
        ~OutOfPlaneSpace() throw();
        virtual OutOfPlaneSpace& operator= (const OutOfPlaneSpace & ) throw();
        const char* what() throw();
        const int nPlane() throw();
        const int maxPlane() throw();
    };

} // namespace lsst

#endif // LSST_MASK_EXCEPTIONS_H


