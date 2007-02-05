// -*- lsst-c++ -*-
#if !defined(LSST_EXCEPTION)      //! multiple inclusion guard macro
#define LSST_EXCEPTION 1

//! \file
//! \brief An exception class that provides a string with details

#include <boost/format.hpp>

namespace lsst {
    class Exception : std::exception {
    public:
        Exception(const std::string msg) throw() {
            _msg = new std::string(msg);
        }
        Exception(const boost::format msg) throw() {
            _msg = new std::string(msg.str());
        }
        ~Exception() throw() {};

        //! Return the details of the exception
        const std::string &getMsg() const { return *_msg; }
    private:
        const std::string *_msg;
    };

    //! A bad_alloc class parallel to std::bad_alloc
    class bad_alloc : public Exception {
    public:
        bad_alloc(const std::string msg) throw() : Exception(msg) {};
    };
}
#endif
