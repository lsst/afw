// -*- lsst-c++ -*-
/** \file 
  *
  * \brief Class providing basic run-time trace facilities.
  *
  * Tracing is controlled on a per "component" basis, where a "component" is a
  * name of the form aaa.bbb.ccc where aaa is the Most significant part; for
  * example, the utilities library might be called "utils", the doubly-linked
  * list "utils.dlist", and the code to destroy a list "utils.dlist.del"
  *
  * \see Component class for details on the verbosity tree which 
  *      determines when a trace record will be emitted.
  *
  * \author Robert Lupton, Princeton University
  */

#include <map>
#include <boost/tokenizer.hpp>

#include "lsst/fw/Trace.h"
#include "lsst/fw/Component.h"

using namespace lsst::fw;

/*****************************************************************************/
/**
  * \brief TraceImpl is a singleton class to which all Trace methods delegate their actual functionality.
  *
  *      By using a singleton class to orchestrate all Trace methods, a simple
  *      uniform interface across all classes is provided for the developers.
  */
class TraceImpl {
public:
    TraceImpl();

    friend class lsst::fw::Trace;

    static void reset();
    static void setVerbosity(const std::string &name);
    static void setVerbosity(const std::string &name, const int verbosity);

    static int getVerbosity(const std::string &name);

    static void printVerbosity(std::ostream &fp = std::cout);

    static void setDestination(std::ostream &fp);
private:

    virtual ~TraceImpl() {}             //!< no-one should delete the singleton

    static Component *_root;            //!< the root of the Component tree

    static std::string _separator;      //!< path separation character
    static std::ostream *_traceStream;  //!< output location for traces

    //! Properties cached for efficiency
    static int _HighestVerbosity;         //!< highest verbosity requested
    static bool _cacheIsValid;            //!< Is the cache valid?
    static std::string _cachedName;       //!< last name looked up
    static int _cachedVerbosity;          //!< verbosity of last looked up name
};


/*****************************************************************************/
/** Create the one true trace tree
  */
TraceImpl::TraceImpl() {
    if (_root == 0) {
        _root = new Component(".", 0);
        _traceStream = &std::cerr;
        _separator = ".";
    }
}

/*****************************************************************************/
/*  Order is important below
 */
Component*    TraceImpl::_root      = 0;
std::string   TraceImpl::_separator;
std::ostream* TraceImpl::_traceStream;

static TraceImpl::TraceImpl* _singleton  = new TraceImpl(); // the singleton

/* The trace verbosity cache
 */
int         TraceImpl::_HighestVerbosity = 0;
bool        TraceImpl::_cacheIsValid     = false;
std::string TraceImpl::_cachedName       = "";
int         TraceImpl::_cachedVerbosity;

/******************************************************************************/
/** Reset the entire trace system
  */
void TraceImpl::reset() {
    delete _root;
    _root = new Component;
    setVerbosity("");
}


/** Set a component's verbosity.
  *
  * If no verbosity is specified, inherit from parent
  */
void TraceImpl::setVerbosity(const std::string &name, //!< component of interest
                             const int verbosity      //!< desired trace verbosity
                        ) {
    _cacheIsValid = false;
    
    _root->add(name, verbosity, _separator);

    if (verbosity > _HighestVerbosity) {
        _HighestVerbosity = verbosity;
    } else {
        _HighestVerbosity = _root->highestVerbosity();
    }
}


/** \overload TraceImpl::setVerbosity(const std::string &name)
  */
void TraceImpl::setVerbosity(const std::string &name //!< component of interest
                        ) {
    int verbosity = Component::INHERIT_VERBOSITY;
    if (name == "" || name == ".") {
        verbosity = 0;
    }
    setVerbosity(name, verbosity);
}


/**  Return a component's verbosity
  */
int TraceImpl::getVerbosity(const std::string &name   //!< component of interest
                           ) {
    //
    // Is name cached?
    //
    if (_cacheIsValid && name == _cachedName) {
        return _cachedVerbosity;
    }

    const int verbosity = _root->getVerbosity(name, _separator);

    _cachedName = name;
    _cachedVerbosity = verbosity;
    _cacheIsValid = true;
    
    return verbosity;
}


/** Print all the trace verbosities
  */
void TraceImpl::printVerbosity(std::ostream &fp     //!< output location
                               ) {
    _root->printVerbosity(fp);
}


/** Change location where traces output
  *
  * Close previous file descriptor if it isn't stdout/stderr
  */
void TraceImpl::setDestination(std::ostream &fp     //!< new output location
                               ) {
    if (*_traceStream != std::cout && *_traceStream != std::cerr) {
        delete _traceStream;
    }
    
    _traceStream = &fp;
}

/******************************************************************************/
LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

/** Generate the trace message.
  */
void Trace::trace(const std::string &msg     //!< Trace message
                 ) {
    *TraceImpl::_traceStream << msg;
}


/** Generate the trace message.
  */
void Trace::trace(const std::string &msg,    //!< Trace message
                  const bool add_newline     //!< Should newline be added?
                 ) {
    *TraceImpl::_traceStream << msg;

    if (msg.substr(msg.size() - 1) != "\n") {
        *TraceImpl::_traceStream << "\n";
    }
}


/** Check that component should be output
  */
bool Trace::check_level(const std::string& name,  //!< Component of interest
                        const int verbosity       //!< Trace request verbosity
                       ) {
    bool print = (verbosity <= TraceImpl::_HighestVerbosity &&
                  TraceImpl::getVerbosity(name) >= verbosity) ? true : false;

    if (print) {
        for (int i = 0; i < verbosity; i++) {
            *TraceImpl::_traceStream << ' ';
        }
    }

    return print;
}


/** Reset the entire Trace system to default values.
  */
void Trace::reset() {
    TraceImpl::reset();
}


/** Set component's verbosity level.
  *
  * If no verbosity is specified, inherit from parent.
  */
void Trace::setVerbosity(const std::string &name,   //!< Component of interest
                         const int verbosity        //!< Component's verbosity
                        ) {
    TraceImpl::setVerbosity(name, verbosity);
}


/** \overload Trace::setVerbosity(const std::string &name)
  */
void Trace::setVerbosity(const std::string &name    //!< Component of interest
                        ) {
    TraceImpl::setVerbosity(name);
}


/** Fetch component's verbosity.
  */
int Trace::getVerbosity(const std::string &name     //!< Component of interest
                       ) {
    return TraceImpl::getVerbosity(name);
}


/** Print entire verbosity tree
  */
void Trace::printVerbosity(std::ostream &fp         //!< Output location
                          ) {
    TraceImpl::printVerbosity(fp);
}


/** Set output location 
  */
void Trace::setDestination(std::ostream &fp         //!< Output location
                          ) {
    TraceImpl::setDestination(fp);
}

LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
