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

namespace lsst {
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
           //! $2
           class $1 : public Exception {
           public:
               $1(const std::string msg) throw() : Exception(msg) {};
               $1(const boost::format msg) throw() : Exception(msg) {};
           })
        
    LSST_NEW_EXCEPTION(Memory,
                       An Exception due to a problem in the memory system);
}
#endif
