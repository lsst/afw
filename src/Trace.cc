/*!
 * \brief A simple implementation of a tracing facility for LSST
 *
 * Tracing is controlled on a per "component" basis, where a "component" is a
 * name of the form aaa.bbb.ccc where aaa is the Most significant part; for
 * example, the utilities library might be called "utils", the doubly-linked
 * list "utils.dlist", and the code to destroy a list "utils.dlist.del" 
 */
#include "Trace.h"

using namespace lsst::utils;

/*****************************************************************************/
/*
 * A component is a string of the form aaa.bbb.ccc, and may itself
 * contain further subcomponents. The Trace structure doesn't
 * in fact contain its full name, but only the last part.
 */
Trace::Trace(const std::string &name,      //!< name of component
             int verbosity                     //!< associated verbosity
            ) {
    _name = new std::string(name);
    _verbosity = verbosity;
    _subcomp = new(comp_map);
}

Trace::~Trace() {
    delete _subcomp;
    delete _name;
}

//! Initialise the trace system
void Trace::init() {
    if (_root == 0) {
        _root = new Trace("");
    }
}

const int Trace::UNKNOWN_LEVEL = -9999;
const std::string Trace::NO_CACHE = "\a";

Trace *Trace::_root = 0;
std::ostream *Trace::_traceStream = 0;

/******************************************************************************/
/*
 * The trace verbosity cache
 */
int Trace::_highest_verbosity = 0;
std::string Trace::_cachedName = NO_CACHE;
int Trace::_cachedVerbosity;

/******************************************************************************/
//! Reset the entire trace system
void Trace::reset() {
    delete _root;
    _root = 0;
}

/*****************************************************************************/
/*
 * Add a new component to the tree
 */
void Trace::add(Trace &comp,          //!< Where to add component
                    const std::string &name,  //!< The name of the component
                    int verbosity                 //!< The component's verbosity verbosity
                   ) {
    //
    // Prepare to parse name
    //
    boost::char_separator<char> sep(".");
    tokenizer components(name, sep);
    tokenizer::iterator token = components.begin();
    const tokenizer::iterator end = components.end();

    if (token != end) {
        doAdd(comp, token, end, verbosity);
    }
}

void Trace::doAdd(Trace &comp,          //!< Where to add component
                      tokenizer::iterator token, //!< parts of name
                      const tokenizer::iterator end, //!< end of name
                      int verbosity                 //!< The component's verbosity verbosity
                     ) {
    const std::string cpt0 = *token++;  // first component of name
    //
    // Does first part of path match this verbosity?
    //
    if (*comp._name == cpt0) {          // a match
	if (token == end) {
	    comp._verbosity = verbosity;
	} else {
            doAdd(comp, token, end, verbosity);
	}
	
	return;
    }
    //
    // Look for a match for cpt0 in this verbosity's subcomps
    //
    comp_map::iterator iter = comp._subcomp->find(cpt0);
    if (iter->second != 0) {
        if (token == end) {
            iter->second->_verbosity = verbosity;
        } else {
            doAdd(*iter->second, token, end, verbosity);
        }

        return;
    }
    /*
     * No match; add cpt0 to this verbosity
     */
    Trace *fcpt0 = new Trace(cpt0);
    (*comp._subcomp)[*fcpt0->_name] = fcpt0;

    if (token == end) {
	fcpt0->_verbosity = verbosity;
    } else {
        doAdd(*fcpt0, token, end, verbosity);
    }
}

/*****************************************************************************/
/*
 * Find the highest verbosity present in the tree
 */
void Trace::setHighestVerbosity(const Trace *comp) {
    if (comp->_verbosity > _root->_highest_verbosity) {
	_root->_highest_verbosity = comp->_verbosity;
    }
    
    for (comp_map::iterator iter = comp->_subcomp->begin();
         iter != comp->_subcomp->end(); iter++) {
	setHighestVerbosity(iter->second);
    }
}

/*****************************************************************************/

void Trace::setVerbosity(const char *comp, //!< component of interest
                         const int verbosity //!< desired trace verbosity
                        ) {
    init();

    if (*comp == '.') {			// skip initial '.'
	comp++;
    }
    
    _cachedName = NO_CACHE;             // invalidate cache
    
    add(*_root, comp, verbosity);

    if (verbosity > _highest_verbosity) {
	_highest_verbosity = verbosity;
    } else {
	_highest_verbosity = Trace::UNKNOWN_LEVEL;
	setHighestVerbosity(_root);
    }
}

/*****************************************************************************/
/*
 * Return a trace verbosity given a name
 */
int Trace::doGetVerbosity(const Trace &comp, //!< end of component to search
                          tokenizer::iterator token, const tokenizer::iterator end
                         ) {
    const std::string cpt0 = *token++;				// first component of name
    /*
     * Look for a match for cpt0 in this verbosity's subcomps
     */
    comp_map::iterator iter = comp._subcomp->find(cpt0);
    if (iter->second != 0) {
        int verbosity;
        if (token == end) {
            verbosity = iter->second->_verbosity;
        } else {
            verbosity = doGetVerbosity(*iter->second, token, end);
        }

        return (verbosity == Trace::UNKNOWN_LEVEL) ? comp._verbosity : verbosity;
    }
    /*
     * No match. This is as far as she goes
     */
    return comp._verbosity;
}

int Trace::getVerbosity(const std::string &name	// component of interest
                       ) {
    init();
    //
    // Is name cached?
    //
    if (name == _cachedName) {
	return _cachedVerbosity;
    }
    //
    // Prepare to parse name
    //
    boost::char_separator<char> sep(".");
    tokenizer components(name, sep);
    tokenizer::iterator token = components.begin();

    const int verbosity = doGetVerbosity(*_root, token, components.end());

    _cachedName = name;
    _cachedVerbosity = verbosity;

    return verbosity;
}

/*****************************************************************************/
/*!
 * Print a tree of trace verbositys
 */
void Trace::printVerbosity() {
    init();

    doPrintVerbosity(*_root, 0);
}

void Trace::doPrintVerbosity(const Trace &comp,
                              int depth
                             ) {
    //
    // Print this verbosity
    //
    for (int i = 0; i < depth; i++) {
        std::cerr << ' ';
    }

    const std::string &name =
        (comp._verbosity == Trace::UNKNOWN_LEVEL) ? "." : *comp._name;
    std::cerr << name;
    for (int i = 0; i < 20 - depth - name.size(); i++) {
        std::cerr << ' ';
    }
    
    const int verbosity = comp._verbosity;
    if (verbosity != Trace::UNKNOWN_LEVEL) {
        std::cerr << verbosity;
    }
    std::cerr << "\n";
    //
    // And other verbositys  too
    //
    for (comp_map::iterator iter = comp._subcomp->begin();
         iter != comp._subcomp->end(); iter++) {
        doPrintVerbosity(*iter->second, depth + 1);
    }
}

/*****************************************************************************/
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

/*****************************************************************************/
/*!
 * Actually generate the trace message. Note that psTrace dealt with
 * prepending appropriate indentation
 */
#if !LSST_NO_TRACE
void Trace::trace(const std::string &comp,	//!< component being traced
                  const int verbosity,		//!< desired trace verbosity
                  const boost::format &msg      //!< trace message
                 ) {
    trace(comp, verbosity, msg.str());
}

void Trace::trace(const std::string &comp,	//!< component being traced
                  const int verbosity,		//!< desired trace verbosity
                  const std::string &msg        //!< trace message
                 ) {
    if (verbosity <= _highest_verbosity &&
        Trace::getVerbosity(comp) >= verbosity) {
        if (_traceStream == 0) {	// not initialised
            _traceStream = &std::cerr;
	}
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
