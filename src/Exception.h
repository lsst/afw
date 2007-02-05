// -*- lsst-c++ -*-
#if !defined(LSST_EXCEPTION)      //! multiple inclusion guard macro
#define LSST_EXCEPTION 1

//! \file
//! \brief An exception class that provides a string with details

#include <exception>
#include <boost/format.hpp>

namespace lsst {
    //! Define an exception that saves a string or boost::format
    class Exception : std::exception {
    public:
        Exception(const std::string msg) throw() {
            _what = new std::string(msg);
        }
        Exception(const boost::format msg) throw() {
            _what = new std::string(msg.str());
        }
        ~Exception() throw() {};

        //! Return the details of the exception
        const char *what() const throw() { return _what->c_str(); }
    private:
        const std::string *_what;
    };

    //! Define a new subclass NAME of Exception without added functionality
#define LSST_NEW_EXCEPTION(NAME) \
    class NAME : public Exception { \
    public: \
        NAME(const std::string msg) throw() : Exception(msg) {}; \
        NAME(const boost::format msg) throw() : Exception(msg) {}; \
    }
    
    //! \brief \name BadAlloc Problem in allocation
    LSST_NEW_EXCEPTION(BadAlloc);
    
#undef LSST_NEW_EXCEPTION
}
#endif
