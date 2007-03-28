// -*- lsst-c++ -*-
/*
 * \file
 * basic run-time trace facilities
 *
 *  \author Robert Lupton, Princeton University
 */
#if !defined(LSST_TRACE_H)
#define LSST_TRACE_H 1

#include <iostream>
#include <string>
#include <boost/format.hpp>

#include "Utils.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

#if !defined(LSST_NO_TRACE)
#  define LSST_NO_TRACE 0               //!< True => turn off all tracing
#endif
/*!
 * \brief A simple implementation of a tracing facility for LSST
 *
 * Tracing is controlled on a per "component" basis, where a "component" is a
 * name of the form aaa.bbb.ccc where aaa is the Most significant part; for
 * example, the utilities library might be called "utils", the doubly-linked
 * list "utils.dlist", and the code to destroy a list "utils.dlist.del" 
 */
class Trace {
public:
    Trace();

#if LSST_NO_TRACE
    static void trace(const std::string &comp, const int verbosity,
                      const std::string &msg) {}
    static void trace(const std::string &comp, const int verbosity,
                      const boost::format &msg) {}
#else
    static void trace(const std::string &comp, const int verbosity,
                      const std::string &msg);
    static void trace(const std::string &comp, const int verbosity,
                      const boost::format &msg);
#endif

    static void reset();
    static void setVerbosity(const std::string &name);
    static void setVerbosity(const std::string &name, const int verbosity);

    static void printVerbosity(std::ostream &fp = std::cout);

    static void setDestination(std::ostream &fp);
private:
    class Component;
    ~Trace() {}                         //!< no-one should delete the singleton

    enum { INHERIT_VERBOSITY = -9999};  //!< use parent's verbosity

    static Component *_root;            //!< the root of the Component tree

    static std::string _separator;      //!< path separation character
    static std::ostream *_traceStream;  //!< output stream for traces

    static int getVerbosity(const std::string &name);

    //! Properties cached for efficiency
    static int _HighestVerbosity;         //!< highest verbosity requested
    static bool _cacheIsValid;            //!< Is the cache valid?
    static std::string _cachedName;       //!< last name looked up
    static int _cachedVerbosity;          //!< verbosity of last looked up name
};

LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
#endif
