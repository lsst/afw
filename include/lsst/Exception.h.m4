// -*- lsst-c++ -*-
#if !defined(LSST_EXCEPTION)      //! multiple inclusion guard macro
#define LSST_EXCEPTION 1
dnl "dnl" is the m4 comment syntax
dnl
undefine(`format')dnl ' Stop m4 expanding format
//
dnl comment to go into the output file
// This file is machine generated from Exception.h.m4. Please do not edit directly
//         
dnl
//! \file
//! \brief An exception class that remembers a string with details of the problem
//
//! Exception is able to accept either a std::string or a boost::format; unlike
//! the VisionWorkbench exception classes it doesn't overload operator<<
#include <exception>
#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>
#include "lsst/DataProperty.h"

namespace lsst {
typedef boost::shared_ptr<DataProperty> DataPropertyPtr;

    //! An exception that saves a string or boost::format
    class Exception : std::runtime_error {
    public:
        explicit Exception(const std::string msg) : std::runtime_error(msg) { }
        explicit Exception(const boost::format msg) : std::runtime_error(msg.str()) { }

        //! Return the details of the exception
        const char *what() const throw() { return std::runtime_error::what(); }
    };

    dnl
    dnl define(name, body) defines an m4 macro
    dnl Define a new subclass $1 of Exception without added functionality; docstring $2
define(LSST_NEW_EXCEPTION,
    `//! $2
    class $1 : public Exception {
    public:
        $1(std::string const& msg ) throw() : \
            Exception(msg),_propertyList(new DataProperty("root",int(0))){};
        $1(std::string const& msg, DataPropertyPtr propertyList) throw() : \
            Exception(msg), _propertyList(propertyList) {};
        $1(boost::format const& msg ) throw() : \
            Exception(msg),_propertyList(new DataProperty("root",int(0))){};
        $1(boost::format const& msg, DataPropertyPtr propertyList) throw() : \
            Exception(msg), _propertyList(propertyList) {};
        $1(const $1 & oops) throw() : \
            Exception(oops.what()), _propertyList(oops._propertyList){};
        ~$1() throw() { \
            std::cout << "----Destroy ExceptObj" << std::endl; \
            _propertyList->print();  \
            };
        DataPropertyPtr propertyList() throw() { return _propertyList;};
        $1 & operator= (const $1 & oops) throw() { \
            Exception::operator= (oops); _propertyList=oops._propertyList; \
            return *this;}; 
        void print(){ \
            std::cout << "----ExceptObj: msg: "<< std::endl; \
            this->what();\
            std::cout << "----ExceptObj: Proplist: " << std::endl; \
            _propertyList->print(); };
    private:
        DataPropertyPtr _propertyList;
    }')

    LSST_NEW_EXCEPTION(Memory,
                       An Exception due to a problem in the memory system);

    LSST_NEW_EXCEPTION(NoMaskPlane,
                      An Exception due to failure to find specified Mask Plane);

    LSST_NEW_EXCEPTION(OutOfPlaneSpace,
                       An Exception due to an insufficient Plane allocation);
}
#endif
