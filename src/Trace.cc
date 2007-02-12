/*!
 * \file
 */
#include <map>

#include <boost/tokenizer.hpp>

#include "lsst/Trace.h"

/*
 * \brief Document namespace
 */
using namespace lsst::utils;

/*****************************************************************************/
/*!
 * \brief A node in the Trace system
 */
class Trace::Component {
public:
    Component(const std::string &name = "", int verbosity=0);
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

    const static int UNKNOWN_LEVEL;     // we don't know this name's verbosity

    std::string *_name;			// last part of name of this component
    int _verbosity;                     // verbosity for this component
    comp_map *_subcomp;                 // next level of subcomponents

    void add(tokenizer::iterator token,
             const tokenizer::iterator end,
             int verbosity);
    int getVerbosity(tokenizer::iterator token,
                     const tokenizer::iterator end);
};

/*****************************************************************************/

const int Trace::Component::UNKNOWN_LEVEL = -9999;

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
Trace::Component::Component(const std::string &name, //!< name of component
                            int verbosity            //!< associated verbosity
                ) {
    _name = new std::string(name);
    _verbosity = verbosity;
    _subcomp = new(comp_map);
}

Trace::Component::~Component() {
    delete _subcomp;
    delete _name;
}

/*!
 * Add a new component to the tree
 */
void Trace::Component::add(const std::string &name,  //!< Component's name
                           int verbosity, //!< The component's verbosity
                           const std::string &separator //!< path separator
                          ) {
    if (name == "") {
        _root->setVerbosity(verbosity);
        return;
    }
    //
    // Prepare to parse name
    //
    boost::char_separator<char> sep(separator.c_str());
    tokenizer components(name, sep);
    tokenizer::iterator token = components.begin();
    const tokenizer::iterator end = components.end();

    if (token != end) {
        add(token, end, verbosity);
    }
}

void Trace::Component::add(tokenizer::iterator token, //!< parts of name
                           const tokenizer::iterator end, //!< end of name
                           int verbosity                 //!< The component's verbosity
                     ) {
    const std::string cpt0 = *token++;  // first component of name
    //
    // Does first part of path match this verbosity?
    //
    if (*_name == cpt0) {          // a match
	if (token == end) {
	    _verbosity = verbosity;
	} else {
            add(token, end, verbosity);
	}
	
	return;
    }
    //
    // Look for a match for cpt0 in this verbosity's subcomps
    //
    comp_map::iterator iter = _subcomp->find(cpt0);
    if (iter != _subcomp->end()) {
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
    (*_subcomp)[*fcpt0->_name] = fcpt0;

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
int Trace::Component::highestVerbosity(int highest //!< minimum verbosity to return
                           ) {
    if (_verbosity > highest) {
	highest = _verbosity;
    }
    
    for (comp_map::iterator iter = _subcomp->begin();
         iter != _subcomp->end(); iter++) {
	highest = iter->second->highestVerbosity(highest);
    }

    return highest;
}

/*****************************************************************************/
/*!
 * Return a trace verbosity given a name
 */
int Trace::Component::getVerbosity(tokenizer::iterator token,
                                   const tokenizer::iterator end
                                  ) {
    const std::string cpt0 = *token++;  // first component of name
    /*
     * Look for a match for cpt0 in this Component's subcomps
     */
    comp_map::iterator iter = _subcomp->find(cpt0);
    if (iter != _subcomp->end()) {
        int verbosity;
        if (token == end) {
            verbosity = iter->second->_verbosity;
        } else {
            verbosity = iter->second->getVerbosity(token, end);
        }

        return (verbosity == UNKNOWN_LEVEL) ? _verbosity : verbosity;
    }
    /*
     * No match. This is as far as she goes
     */
    return _verbosity;
}

//!
// Return a component's verbosity, from the perspective of this.
//
// \sa Trace::Component::getVerbosity
//
int Trace::Component::getVerbosity(const std::string &name, // component of interest
                                   const std::string &separator //!< path separator
                                  ) {
    //
    // Prepare to parse name
    //
    boost::char_separator<char> sep(separator.c_str());
    tokenizer components(name, sep);
    tokenizer::iterator token = components.begin();

    return getVerbosity(token, components.end());
}

/*!
 * Print all the trace verbosities rooted at this
 */
void Trace::Component::printVerbosity(std::ostream &fp,
                                      int depth
                                     ) {
    //
    // Print this verbosity
    //
    for (int i = 0; i < depth; i++) {
        fp << ' ';
    }

    const std::string &name = (_verbosity == UNKNOWN_LEVEL) ? "." : *_name;
    fp << name;
    for (int i = 0; i < 20 - depth - static_cast<int>(name.size()); i++) {
        fp << ' ';
    }
    
    if (_verbosity != UNKNOWN_LEVEL) {
        fp << _verbosity;
    }
    fp << "\n";
    //
    // And other levels of the hierarchy too
    //
    for (comp_map::iterator iter = _subcomp->begin();
         iter != _subcomp->end(); iter++) {
        iter->second->printVerbosity(fp, depth + 1);
    }
}

/*****************************************************************************/
//! Create the one true trace tree
    
Trace::Trace() {
    if (_root == 0) {
        _root = new Component();
        _traceStream = &std::cerr;
        _separator = ".";
    }
}

Trace::Component *Trace::_root = 0;
std::string Trace::_separator;
std::ostream *Trace::_traceStream;

static Trace::Trace *_root = new Trace(); // the singleton

/******************************************************************************/
/*
 * The trace verbosity cache
 */
int Trace::_HighestVerbosity = 0;
bool Trace::_cacheIsValid = false;
std::string Trace::_cachedName = "";
int Trace::_cachedVerbosity;

/******************************************************************************/
//! Reset the entire trace system

void Trace::reset() {
    delete _root;
    _root = new Component;
}
/*!
 * Set a component's verbosity
 */
void Trace::setVerbosity(const char *name, //!< component of interest
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
int Trace::getVerbosity(const std::string &name	// component of interest
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
void Trace::printVerbosity(std::ostream &fp) {
    _root->printVerbosity(fp);
}

/*!
 * Change where traces go
 *
 * close previous file descriptor if it isn't stdout/stderr
 */
void Trace::setDestination(std::ostream &fp) {
    if (*_traceStream != std::cout && *_traceStream != std::cerr) {
        delete _traceStream;
    }
    
    _traceStream = &fp;
}

/*!
 * Actually generate the trace message. Note that psTrace dealt with
 * prepending appropriate indentation
 */
#if !LSST_NO_TRACE
void Trace::trace(const std::string &name,	//!< component being traced
                  const int verbosity,		//!< desired trace verbosity
                  const boost::format &msg      //!< trace message
                 ) {
    trace(name, verbosity, msg.str());
}

void Trace::trace(const std::string &name,	//!< component being traced
                  const int verbosity,		//!< desired trace verbosity
                  const std::string &msg        //!< trace message
                 ) {
    if (verbosity <= _HighestVerbosity &&
        Trace::getVerbosity(name) >= verbosity) {
	for (int i = 0; i < verbosity; i++) {
            *_traceStream << ' ';
	}
        *_traceStream << msg;

        if (msg.substr(msg.size() - 1) != "\n") {
            *_traceStream << "\n";
	}
    }
}
#endif
