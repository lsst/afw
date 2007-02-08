// -*- lsst-c++ -*-
/*!
 *  \brief basic run-time trace facilities
 *
 *  A class to print trace messages from code, with controllable verbosity
 *
 *  \author Robert Lupton, Princeton University
 */
#if !defined(LSST_TRACE_H)
#define LSST_TRACE_H 1

#include <iostream>
#include <string>
#include <map>

#include <boost/tokenizer.hpp>
#include <boost/format.hpp>

#include "Utils.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(utils)

#if !defined(LSST_NO_TRACE)
#  define LSST_NO_TRACE 0               //!< to turn off all tracing
#endif
    
class Trace {
public:
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
    static int getVerbosity(const std::string &name);
    static void setVerbosity(const char *comp,
                             const int verbosity);

    static void printVerbosity();

    static void setDestination(std::ostream &fp);
private:
    Trace(const std::string &name,  //!< name of component
          int verbosity = Trace::UNKNOWN_LEVEL //!< associated verbosity
         );
    ~Trace();

    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    typedef std::map<const std::string, Trace *> comp_map; //!< sub components

    std::string *_name;			//!< last part of name of this component
    int _verbosity;                     //!< verbosity for this component
    comp_map *_subcomp;                 //!< next level of subcomponents

    const static std::string Trace::NO_CACHE; //!< There is no valid cache
    const static int UNKNOWN_LEVEL;     //!< we don't know this name's verbosity

    static Trace *_root;                //!< root of the trace component tree

    static std::ostream *_traceStream;  //!< output stream for traces
    //! Properties cached for efficiency
    static int _highest_verbosity;              //!< highest verbosity requested
    static std::string _cachedName;             //!< last name looked up
    static int _cachedVerbosity;        //!< verbosity of last looked up name

    static void init();
    static void setHighestVerbosity(const Trace *comp);
    static void add(Trace &comp, const std::string &name, int verbosity);
    static void doAdd(Trace &comp,
                      tokenizer::iterator token, const tokenizer::iterator end,
                      int verbosity);
    static void doPrintVerbosity(const Trace &comp, int depth);
    static int doGetVerbosity(const Trace &comp,
                              tokenizer::iterator token,
                              const tokenizer::iterator end);
};

LSST_END_NAMESPACE(utils)
LSST_END_NAMESPACE(lsst)
#endif
