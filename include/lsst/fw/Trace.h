// -*- lsst-c++ -*-
/** \file    Trace.h
  *
  * \ingroup fw
  *
  * \brief  Class providing basic run-time trace facilities.
  *
  * \author Robert Lupton, Princeton University
  */

#if !defined(LSST_FW_TRACE_H)        //!< multiple inclusion guard macro
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

/**
 * \brief  Class providing basic run-time trace facilities.
 *
 *      Tracing is controlled on a per "component" basis, where a "component" 
 *      is a name of the form aaa.bbb.ccc where aaa is the Most significant 
 *      part; for example, the utilities library might be called "utils", 
 *      the doubly-linked list "utils.dlist", and the code to destroy a 
 *      list "utils.dlist.del".
 *
 *      All tracing may be disabled by recompiling with LSST_NO_TRACE defined
 *      to be non-zero
 *
 * \see Component class for details on the verbosity tree which
 *      determines when a trace record will be emitted.
 */
class Trace {
public:
#if !LSST_NO_TRACE
    /**
     * Return a Trace object (which will later print if verbosity is high enough
     * for name) to which a message can be attached with <<
     */
    Trace(const std::string& name,      //!< Name of component
          const int verbosity           //!< Desired verbosity
         ) :
        _print(check_level(name, verbosity)), _verbosity(verbosity) {
        ;
    }

    /**
     * Print msg if verbosity is high enough for name
     */
    Trace(const std::string& name,      //!< Name of component
          const int verbosity,          //!< Desired verbosity
          const std::string& msg        //!< Message to write
         ) :
        _print(check_level(name, verbosity)), _verbosity(verbosity) {
        if (_print) {
            trace(msg, true);
        }
    }

    /**
     * Print msg if verbosity is high enough for name
     */
    Trace(const std::string& name,      //!< Name of component
          const int verbosity,          //!< Desired verbosity
          const boost::format& msg      //!< Message to write
         ) :
        _print(check_level(name, verbosity)), _verbosity(verbosity) {
        if (_print) {
            trace(msg.str(), true);
        }
    }

    /**
      * Add to a trace record being emitted.
      *
      */
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

    static void setDestination(std::ostream &fp);
    static void setVerbosity(const std::string &name);
    static void setVerbosity(const std::string &name, const int verbosity);
    static int  getVerbosity(const std::string &name);
    static void printVerbosity(std::ostream &fp = std::cout);
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
