/*!
 * \file
 */
/*!
 * \brief A simple implementation of a tracing facility for LSST
 *
 * Tracing is controlled on a per "component" basis, where a "component" is a
 * name of the form aaa.bbb.ccc where aaa is the Most significant part; for
 * example, the utilities library might be called "utils", the doubly-linked
 * list "utils.dlist", and the code to destroy a list "utils.dlist.del" 
 *
 * All tracing may be disabled by recompiling with LSST_NO_TRACE defined
 * to be non-zero
 */
#include <map>

#include <boost/tokenizer.hpp>

#include "lsst/fw/Trace.h"

using namespace lsst::fw;

class TraceImpl {
public:
    TraceImpl();

    friend void Trace::trace(const std::string &comp, const int verbosity,
                             const std::string &msg);
    friend void Trace::trace(const std::string &comp, const int verbosity,
                             const boost::format &msg);

    static void reset();
    static void setVerbosity(const std::string &name);
    static void setVerbosity(const std::string &name, const int verbosity);

    static int getVerbosity(const std::string &name);

    static void printVerbosity(std::ostream &fp = std::cout);

    static void setDestination(std::ostream &fp);
private:
    class Component;
    ~TraceImpl() {}                         //!< no-one should delete the singleton

    enum { INHERIT_VERBOSITY = -9999};  //!< use parent's verbosity

    static Component *_root;            //!< the root of the Component tree

    static std::string _separator;      //!< path separation character
    static std::ostream *_traceStream;  //!< output stream for traces

    //! Properties cached for efficiency
    static int _HighestVerbosity;         //!< highest verbosity requested
    static bool _cacheIsValid;            //!< Is the cache valid?
    static std::string _cachedName;       //!< last name looked up
    static int _cachedVerbosity;          //!< verbosity of last looked up name
};

/*****************************************************************************/
/*!
 * \brief A node in the Trace system
 */
class TraceImpl::Component {
public:
    Component(const std::string &name = ".", int verbosity=INHERIT_VERBOSITY);
    ~Component();

    void add(const std::string &name, int verbosity,
             const std::string &separator);

    int getVerbosity(const std::string &name,
                     const std::string &separator);
    int highestVerbosity(int highest=0);
    void printVerbosity(std::ostream &fp, int depth = 0);
    void setVerbosity(int verbosity) { _verbosity = verbosity; }
private:
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    typedef std::map<const std::string, Component *> comp_map; // subcomponents

    std::string *_name;			// last part of name of this component
    int _verbosity;                     // verbosity for this component
    comp_map &_subcomp;                 // next level of subcomponents

    void add(tokenizer::iterator token,
             const tokenizer::iterator end,
             int verbosity);
    int getVerbosity(tokenizer::iterator token,
                     const tokenizer::iterator end,
                     int defaultVerbosity);
};

/*****************************************************************************/
/*!
 * Create a component of the trace tree
 *
 * A name is a string of the form aaa.bbb.ccc, and may itself contain further
 * subcomponents. The Component structure doesn't in fact contain its full name,
 * but only the first part.
 *
 * The reason for this is inheritance --- verbosity is inherited, but may be
 * overriden.  For example, if "foo" is at level 2 then "foo.goo.hoo" is taken
 * to be at level 2 unless set specifically -- but this inheritance is dynamic
 * (so changing "foo" to 3 changes "foo.goo.hoo" too).  However, I may also set
 * "foo.goo.hoo" explicitly, in which case "foo"'s value is irrelevant --- but
 * "foo.goo" continues to inherit it.
 */
TraceImpl::Component::Component(const std::string &name, //!< name of component
                            int verbosity            //!< associated verbosity
                           ) : _name(new std::string(name)),
                               _verbosity(verbosity),
                               _subcomp(*new(comp_map)) {
}

TraceImpl::Component::~Component() {
    delete &_subcomp;
    delete _name;
}

/*!
 * Add a new component to the tree
 */
void TraceImpl::Component::add(const std::string &name,  //!< Component's name
                           int verbosity, //!< The component's verbosity
                           const std::string &separator //!< path separator
                          ) {
    //
    // Prepare to parse name
    //
    boost::char_separator<char> sep(separator.c_str());
    tokenizer components(name, sep);
    tokenizer::iterator token = components.begin();
    const tokenizer::iterator end = components.end();

    if (token == end) {                 // "" or "."
        _root->setVerbosity(verbosity);
    } else {
        add(token, end, verbosity);
    }
}

void TraceImpl::Component::add(tokenizer::iterator token, //!< parts of name
                           const tokenizer::iterator end, //!< end of name
                           int verbosity //!< The component's verbosity
                     ) {
    const std::string cpt0 = *token++;  // first component of name
    //
    // Does first part of path match this verbosity?
    //
    if (*_name == cpt0) {               // a match
	if (token == end) {             // name has no more components
	    _verbosity = verbosity;
	} else {
            add(token, end, verbosity);
	}
	
	return;
    }
    //
    // Look for a match for cpt0 in this verbosity's subcomps
    //
    comp_map::iterator iter = _subcomp.find(cpt0);
    if (iter != _subcomp.end()) {
        if (token == end) {
            iter->second->_verbosity = verbosity;
        } else {
            iter->second->add(token, end, verbosity);
        }

        return;
    }
    /*
     * No match; add cpt0 to this verbosity
     */
    Component *fcpt0 = new Component(cpt0);
    _subcomp[*fcpt0->_name] = fcpt0;

    if (token == end) {
	fcpt0->_verbosity = verbosity;
    } else {
        fcpt0->add(token, end, verbosity);
    }
}

/*****************************************************************************/
/*!
 * Return the highest verbosity rooted at comp
 */
int TraceImpl::Component::highestVerbosity(int highest //!< minimum verbosity to return
                           ) {
    if (_verbosity > highest) {
	highest = _verbosity;
    }
    
    for (comp_map::iterator iter = _subcomp.begin();
         iter != _subcomp.end(); iter++) {
	highest = iter->second->highestVerbosity(highest);
    }

    return highest;
}

/*****************************************************************************/
/*!
 * Return a trace verbosity given a name
 */
int TraceImpl::Component::getVerbosity(tokenizer::iterator token,
                                       const tokenizer::iterator end,
                                       int defaultVerbosity
                                      ) {
    const std::string cpt0 = *token++;  // first component of name
    /*
     * Look for a match for cpt0 in this Component's subcomps
     */
    comp_map::iterator iter = _subcomp.find(cpt0);
    if (iter != _subcomp.end()) {
        int verbosity = iter->second->_verbosity;
        if (verbosity == INHERIT_VERBOSITY) {
            verbosity = defaultVerbosity;
        }

        if (token == end) {             // there was only one component
            ;                           // so save the function call
        } else {
            verbosity = iter->second->getVerbosity(token, end, verbosity);
        }

        return (verbosity == INHERIT_VERBOSITY) ? defaultVerbosity : verbosity;
    }
    /*
     * No match. This is as far as she goes
     */
    return _verbosity;
}

//!
// Return a component's verbosity, from the perspective of "this".
//
// \sa TraceImpl::Component::getVerbosity
//
int TraceImpl::Component::getVerbosity(const std::string &name, // component of interest
                                       const std::string &separator //!< path separator
                                      ) {
    //
    // Prepare to parse name
    //
    boost::char_separator<char> sep(separator.c_str());
    tokenizer components(name, sep);
    tokenizer::iterator token = components.begin();

    if (token == components.end()) {
        return _verbosity;
    }

    return getVerbosity(token, components.end(), _verbosity);
}

/*!
 * Print all the trace verbosities rooted at "this"
 */
void TraceImpl::Component::printVerbosity(std::ostream &fp,
                                      int depth
                                     ) {
    if (_subcomp.empty() && _verbosity == INHERIT_VERBOSITY) {
        return;
    }
    //
    // Print this verbosity
    //
    for (int i = 0; i < depth; i++) {
        fp << ' ';
    }

    const std::string &name = *_name;
    fp << name;
    for (int i = 0; i < 20 - depth - static_cast<int>(name.size()); i++) {
        fp << ' ';
    }
    
    if (_verbosity != INHERIT_VERBOSITY) {
        fp << _verbosity;
    }
    fp << "\n";
    //
    // And other levels of the hierarchy too
    //
    for (comp_map::iterator iter = _subcomp.begin();
         iter != _subcomp.end(); iter++) {
        iter->second->printVerbosity(fp, depth + 1);
    }
}

/*****************************************************************************/
//! Create the one true trace tree
    
TraceImpl::TraceImpl() {
    if (_root == 0) {
        _root = new Component(".", 0);
        _traceStream = &std::cerr;
        _separator = ".";
    }
}

TraceImpl::Component *TraceImpl::_root = 0;
std::string TraceImpl::_separator;
std::ostream *TraceImpl::_traceStream;

static TraceImpl::TraceImpl *_root = new TraceImpl(); // the singleton

/******************************************************************************/
/*
 * The trace verbosity cache
 */
int TraceImpl::_HighestVerbosity = 0;
bool TraceImpl::_cacheIsValid = false;
std::string TraceImpl::_cachedName = "";
int TraceImpl::_cachedVerbosity;

/******************************************************************************/
//! Reset the entire trace system

void TraceImpl::reset() {
    delete _root;
    _root = new Component;
    setVerbosity("");
}
/*!
 * Set a component's verbosity.
 *
 * If no verbosity is specified, inherit from parent
 */
void TraceImpl::setVerbosity(const std::string &name //!< component of interest
                        ) {

    int verbosity = INHERIT_VERBOSITY;
    if (name == "" || name == ".") {
        verbosity = 0;
    }
    setVerbosity(name, verbosity);
}

void TraceImpl::setVerbosity(const std::string &name, //!< component of interest
                         const int verbosity //!< desired trace verbosity
                        ) {
    _cacheIsValid = false;
    
    _root->add(name, verbosity, _separator);

    if (verbosity > _HighestVerbosity) {
	_HighestVerbosity = verbosity;
    } else {
	_HighestVerbosity = _root->highestVerbosity();
    }
}

//!
// Return a component's verbosity
//
int TraceImpl::getVerbosity(const std::string &name	// component of interest
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

/*!
 * Print all the trace verbosities
 */
void TraceImpl::printVerbosity(std::ostream &fp) {
    _root->printVerbosity(fp);
}

/*!
 * Change where traces go
 *
 * close previous file descriptor if it isn't stdout/stderr
 */
void TraceImpl::setDestination(std::ostream &fp) {
    if (*_traceStream != std::cout && *_traceStream != std::cerr) {
        delete _traceStream;
    }
    
    _traceStream = &fp;
}

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)
/*
 * \brief Document namespace Trace
 */
LSST_START_NAMESPACE(Trace)

/*!
 * Actually generate the trace message.
 */
#if !LSST_NO_TRACE
void trace(const std::string &name,	//!< component being traced
           const int verbosity,		//!< desired trace verbosity
           const boost::format &msg     //!< trace message
          ) {
    trace(name, verbosity, msg.str());
}

void trace(const std::string &name,	//!< component being traced
           const int verbosity,		//!< desired trace verbosity
           const std::string &msg       //!< trace message
          ) {
    if (verbosity <= TraceImpl::_HighestVerbosity &&
        TraceImpl::getVerbosity(name) >= verbosity) {
	for (int i = 0; i < verbosity; i++) {
            *TraceImpl::_traceStream << ' ';
	}
        *TraceImpl::_traceStream << msg;

        if (msg.substr(msg.size() - 1) != "\n") {
            *TraceImpl::_traceStream << "\n";
	}
    }
}
#endif

void reset() {
    TraceImpl::reset();
}

void setVerbosity(const std::string &name) {
    TraceImpl::setVerbosity(name);
}

void setVerbosity(const std::string &name, const int verbosity) {
    TraceImpl::setVerbosity(name, verbosity);
}

int getVerbosity(const std::string &name) {
    return TraceImpl::getVerbosity(name);
}

void printVerbosity(std::ostream &fp) {
    TraceImpl::printVerbosity(fp);
}

void setDestination(std::ostream &fp) {
    TraceImpl::setDestination(fp);
}

LSST_END_NAMESPACE(Trace)
LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
    
