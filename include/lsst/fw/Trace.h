// -*- lsst-c++ -*-
/*
 * \file
 * basic run-time trace facilities
 *
 *  \author Robert Lupton, Princeton University
 */
#if !defined(LSST_FW_TRACE_H)
#define LSST_FW_TRACE_H 1

#include <iostream>
#include <string>
#include <sstream>
#include <boost/format.hpp>

#include "Utils.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

#if !defined(LSST_NO_TRACE)
#  define LSST_NO_TRACE 0               //!< True => turn off all tracing
#endif

class Trace {
public:
#if !LSST_NO_TRACE
    Trace(const std::string& name, const int verbosity) :
        _print(check_level(name, verbosity)), _verbosity(verbosity) {
        ;
    }

    Trace(const std::string& name, const int verbosity,
          const std::string& msg) :
        _print(check_level(name, verbosity)), _verbosity(verbosity) {
        if (_print) {
            trace(msg, true);
        }
    }

    Trace(const std::string& name, const int verbosity,
          const boost::format& msg) :
        _print(check_level(name, verbosity)), _verbosity(verbosity) {
        if (_print) {
            trace(msg.str(), true);
        }
    }

    template<typename T>
    Trace& operator<<(T v) {
        if (_print) {
            std::ostringstream s;
            s << v;
            trace(s.str());
        }
        return *this;
    }
#else
    Trace(const std::string& name, const int verbosity) {}
    Trace(const std::string& name, const int verbosity,
          const std::string& msg) {}
    Trace(const std::string& name, const int verbosity,
          const boost::format& msg) {}

    template<typename T>
    Trace& operator<<(T v) {
        return *this;
    }
#endif

    static void reset();
    static void setVerbosity(const std::string &name);
    static void setVerbosity(const std::string &name, const int verbosity);

    static int getVerbosity(const std::string &name);

    static void printVerbosity(std::ostream &fp = std::cout);

    static void setDestination(std::ostream &fp);
private:
    bool _print;
    int _verbosity;

    bool check_level(const std::string& name, const int verbosity);
    void trace(const std::string& msg);
    void trace(const std::string& msg, const bool add_newline);
};

LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
#endif
