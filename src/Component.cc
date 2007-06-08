// -*- lsst-c++ -*-
/** \file Component.cc
  *
  * \brief Create a component in the verbosity tree. 
  *
  *        Both Trace and Log use a verbosity tree to determine if a 
  *        status message should be emitted.
  *
  * \author Robert Lupton, Princeton University
  */

#include <map>
#include <boost/tokenizer.hpp>

#include "lsst/fw/Trace.h"
#include "lsst/fw/Component.h"

using namespace lsst::fw;

/*****************************************************************************/
/** Initialize the component structure.
 */
Component::Component(const std::string &name,   //!< component's name
                    int verbosity               //!< component's verbosity
                    ) : _name(new std::string(name)),
                        _verbosity(verbosity),
                        _subcomp(*new(comp_map)) {
}

/** Destroy the component structure.
 */
Component::~Component() {
    delete &_subcomp;
    delete _name;
}

/** Add a new component to the tree
 */
void Component::add(const std::string &name,     //!< component's name
                    int verbosity,               //!< component's verbosity
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
        this->setVerbosity(verbosity);
    } else {
        add(token, end, verbosity);
    }
}

/** Add a new component to the tree
 */
void Component::add(tokenizer::iterator token,     //!< parts of name
                    const tokenizer::iterator end, //!< end of name
                    int verbosity                  //!< component's verbosity
                     ) {
    const std::string cpt0 = *token++;  // first component of name
    /*
     * Does first part of path match this verbosity?
     */
    if (*_name == cpt0) {               // a match
        if (token == end) {             // name has no more components
            _verbosity = verbosity;
        } else {
            add(token, end, verbosity);
        }
        
        return;
    }
    /*
    * Look for a match for cpt0 in this verbosity's subcomps
    */
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
/** Return the highest verbosity rooted at comp
 */
int Component::highestVerbosity(int highest //!< minimum verbosity to return
                           ) {
    if (_verbosity > highest) {
        highest = _verbosity;
    }
    
    for (comp_map::iterator iter = _subcomp.begin();
         iter != _subcomp.end(); iter++) {
        highest = iter->second->highestVerbosity(highest);
    }

    return highest;                     //!< return highest verbosity rooted at comp
}

/*****************************************************************************/
/** Return a trace verbosity given a name
 */
int Component::getVerbosity(tokenizer::iterator token,     //!< parts of name
                            const tokenizer::iterator end, //!< end of name
                            int defaultVerbosity           //!< default verbosity
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

        return (verbosity == INHERIT_VERBOSITY) ? defaultVerbosity : verbosity; //!< Return trace verbosity given name
    }
    /*
     * No match. This is as far as she goes
     */
    return _verbosity;          //!< Return trace verbosity given name
}

/**  Return a component's verbosity, from the perspective of "this".
  *
  * \sa Component::getVerbosity
  */
int Component::getVerbosity(const std::string &name,     //!< component of interest
                            const std::string &separator //!< path separator
                            ) {
    //
    // Prepare to parse name
    //
    boost::char_separator<char> sep(separator.c_str());
    tokenizer components(name, sep);
    tokenizer::iterator token = components.begin();

    if (token == components.end()) {
        return _verbosity;      //!< Return a component's verbosity
    }

    return getVerbosity(token, components.end(), _verbosity); //!< Return a component's verbosity
}

/** Print all the trace verbosities rooted at "this"
  */
void Component::printVerbosity(std::ostream &fp,    //!< Output stream
                               int depth            //!< Tree depth to recurse
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

