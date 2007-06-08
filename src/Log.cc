// -*- lsst-c++ -*-
/** \file 
  *
  * \brief Implementation of a logging facility for LSST
  *
  * Logging is controlled on a per "component" basis, where a "component" is a
  * name of the form aaa.bbb.ccc where aaa is the Most significant part; for
  * example, the utilities library might be called "utils", the doubly-linked
  * list "utils.dlist", and the code to destroy a list "utils.dlist.del" 
  *
  * \see Component class for details on the verbosity tree which
  *      determines when a log record will be emitted.
  *
  * Author: Robert Lupton, Princeton University
  * Author: Roberta Allsman, LSST Corp
  */

#include <time.h>
#include <sys/time.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/types.h>
#include <unistd.h>

#include "lsst/fw/Log.h"
#include "lsst/fw/Component.h"

using namespace lsst::fw;

/*****************************************************************************/
/**
 * \brief LogImpl is a singleton class to which all Log methods delegate their actual functionality.
 *
 *      By using a singleton class to orchestrate all Log methods, a simple
 *      uniform interface across all classes is provided for the developers.
 */
class LogImpl {
public:
    LogImpl();

    friend class lsst::fw::Log;

    static void reset();
    static void setVerbosity(const std::string &name);
    static void setVerbosity(const std::string &name, const int verbosity);
    static void setDestination(std::ostream &fp);

    static int  getVerbosity(const std::string &name);

    static void printVerbosity(std::ostream &fp = std::cout);

private:
    virtual     ~LogImpl() {}           //!< no-one should delete the singleton
    void        setPreamble();          //!< initialize Log record preamble
    void        logPreamble();          //!< output Log record preamble

    static      Component*    _root;    //!< the root of the Component tree

    static      std::string   _separator;    //!< path separation character
    static      std::ostream* _logStream;    //!< output stream for logs
    static      std::string   _preamble;     //!< Log record preamble

    //! Properties cached for efficiency
    static int         _HighestVerbosity; //!< highest verbosity requested
    static bool        _cacheIsValid;     //!< Is the cache valid?
    static std::string _cachedName;       //!< last name looked up
    static int         _cachedVerbosity;  //!< verbosity of last looked up name
};

/*****************************************************************************/
/** Create the one true log tree
  */
LogImpl::LogImpl() {
    if (_root == 0) {
        _root = new Component(".", 0);
        _logStream = &std::cerr;
        _separator = ".";
        setPreamble();
    }
}

/******************************************************************************/
/* Order is important below
 */

Component*    LogImpl::_root       = 0;
std::string   LogImpl::_separator;
std::ostream* LogImpl::_logStream;
std::string   LogImpl::_preamble;

static LogImpl::LogImpl* _singleton     = new LogImpl(); // the singleton

// The log verbosity cache
int           LogImpl::_HighestVerbosity = 0;
bool          LogImpl::_cacheIsValid     = false;
std::string   LogImpl::_cachedName       = "";
int           LogImpl::_cachedVerbosity;


/******************************************************************************/
/** Reset the entire log system
  */
void LogImpl::reset() {
    delete _root;
    _root = new Component;
    setVerbosity("");
}


/** Set a component's verbosity.
  *
  *    If no verbosity is specified, inherit from parent
  */
void LogImpl::setVerbosity(const std::string &name, //!< Component of interest
                           const int verbosity      //!< Desired log verbosity
                           ) {
    _cacheIsValid = false;
    
    _root->add(name, verbosity, _separator);

    if (verbosity > _HighestVerbosity) {
        _HighestVerbosity = verbosity;
    } else {
        _HighestVerbosity = _root->highestVerbosity();
    }
}


/** \overload LogImpl::setVerbosity(const std::string &name)
  */
void LogImpl::setVerbosity(const std::string &name //!< Component of interest
                        ) {
    int verbosity = Component::INHERIT_VERBOSITY;
    if (name == "" || name == ".") {
        verbosity = 0;
    }
    setVerbosity(name, verbosity);
}


/** Return a component's verbosity
  */
int LogImpl::getVerbosity(const std::string &name   //!<  Component of interest
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


/** Print all the log verbosities
  */
void LogImpl::printVerbosity(std::ostream &fp       //!< Output stream
                            ) {
    _root->printVerbosity(fp);
}


/** Change where logs go
  *
  *   Close previous file descriptor if it isn't stdout/stderr
  */
void LogImpl::setDestination(std::ostream &fp       //!< Output stream
                            ) {
    if (*_logStream != std::cout && *_logStream != std::cerr) {
        delete _logStream;
    }
    
    _logStream = &fp;
}


/** Generate fixed environmental preamble required for each Log record.
  */
void LogImpl::setPreamble() {
    const int maxNameLength = 255;
    char name[maxNameLength];
    struct hostent *hostinfo;
    const int maxIpLength = 17;
    char ip[maxIpLength];
    std::string unknownNameConstant("unknown");
    std::string ipConstant("111.111.111.111");
    
    int foundHostName = gethostname( name, maxNameLength) ;
    if( foundHostName != 0) {
        strncpy(name,unknownNameConstant.c_str(),maxNameLength -1);
        strncpy(ip, ipConstant.c_str(),sizeof(ip) -1);
    } else {
        hostinfo = gethostbyname(name);
        if (hostinfo != 0) {
            strncpy( ip, inet_ntoa(*(struct in_addr *)*hostinfo->h_addr_list),
                     sizeof(ip) - 1 );
        } else {
            strncpy(ip, ipConstant.c_str(),sizeof(ip) -1);
        }
    }

    _preamble = str(boost::format("PID: %d\nHOST: %s\nIP: %s\n") % getpid() % std::string(name) % std::string(ip) );
}


/** Output the environmental preamble required for each Log record.
  *
  * Preamble includes: Current date and time, Process Id, Hostname, IP Address.
  */
void LogImpl::logPreamble(){
    
    struct timeval tv;      
    struct timezone tz;     
    gettimeofday(&tv,&tz);
    // _tv.tv_sec = seconds since the epoch
    // _tv.tv_usec = microseconds since tv.tv_sec

    time_t rawtime;
    struct tm timeinfo;

    // TODO: a future efficiency:  if (last time's tv == this time's tv), then
    // don't rebuild whole datestring--just modify the fractional seconds.
    // see NL_fmt_iso8601 in netlogger/nl_log.c

    char xxxx[40];

    time(&rawtime);
    gmtime_r(&rawtime,&timeinfo);

    if ( 0 == strftime(xxxx,39,"\nDATE: %Y-%m-%dT%H:%M:%S.",&timeinfo)) {
        //TODO: What to do if failed to convert time for some reason?????
        throw;
    }

    *_logStream << str(boost::format("%s%d\n") % std::string(xxxx) % tv.tv_usec) << _preamble;
}

/************************************************************************/

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

/** Generate a minimal Log record
  *
  * Log record contains date, time, host system details and logging component name.
  */
Log::Log(const std::string& name,           //!< Log record component name
         const int verbosity                //!< Component's Verbosity level
         ) :
    _print(check_level(name, verbosity)), _verbosity(verbosity) {
    if ( _print) {
        _singleton->LogImpl::logPreamble();
        log(boost::format("EVNT: %s\n") %(name) );
    }
}


/** Generate Log record
  *
  * Log record contains date, time, host system details,
  * logging component name and user provided attributes.
  */
Log::Log(const std::string& name,                //!< Log record component name
         const int verbosity,                    //!< Component's Verbosity level
         const DataPropertyPtrT dp //!< (keyword, value) attributes
         ) :
    _print(check_level(name, verbosity)), _verbosity(verbosity) {
    if (_print) {
        _singleton->LogImpl::logPreamble();
        log(boost::format("EVNT: %s\n") %(name) );
        log(dp->treeNode("", ": "));
    }
}


/** Generate Log record
  *
  * Log record contains date, time, host system details,
  * logging component name and user provided attributes.
  */
Log::Log(const std::string& name,           //!< Log record component name
         const int verbosity,               //!< Component's Verbosity level
         const DataProperty& dp             //!< (keyword, value) attributes
         ) :
    _print(check_level(name, verbosity)), _verbosity(verbosity) {
    if (_print) {
        _singleton->LogImpl::logPreamble();
        log(boost::format("EVNT: %s\n") %(name) );
        log(dp.treeNode("", ": "));
    }
}


/** Generate minimal Log record
  *
  * The minmal Log record contains date, time, host system details,
  * logging  component name.
  */
void Log::log(const std::string &msg            //!< Log message
              ) {
    *LogImpl::_logStream << msg;
}


/** \overload Log::log(const boost::format &msg)
  */
void Log::log(const boost::format &msg          //!< Log message
              ) {
    *LogImpl::_logStream << msg.str();
}


bool Log::check_level(const std::string& name,  //!< Component name
                      const int verbosity       //!< Component's verbosity
                      ) {
    bool print = (verbosity <= LogImpl::_HighestVerbosity &&
                  LogImpl::getVerbosity(name) >= verbosity) ? true : false;
    return print;
}


/** Reset entire Log system to default values.
  */
void Log::reset() {
    LogImpl::reset();
}


/** Set verbosity level of indicated Component name.
  *
  * Use parent Component's verbosity level, if not specified.
  */
void Log::setVerbosity(const std::string &name, //!< Component's name
                       const int verbosity      //!< Component's verbosity
                       ) {
    LogImpl::setVerbosity(name, verbosity);
}


/** \overload void Log::setVerbosity(const std::string &name)
  */
void Log::setVerbosity(const std::string &name  //!< Component's name
                       ) {
    LogImpl::setVerbosity(name);
}


/** Retrieve verbosity level associated with specified logging component.
  */
int Log::getVerbosity(const std::string &name   //!< Component's name
                      ) {
    return LogImpl::getVerbosity(name);
}


/** Output all defined (component,verbosity) pairs.
  */
void Log::printVerbosity(std::ostream &fp       //!< Output stream
                         ) {
    LogImpl::printVerbosity(fp);
}


/** Set destination output stream for Log records.
  */
void Log::setDestination(std::ostream &fp       //!< Output stream
                         ) {
    LogImpl::setDestination(fp);
}

LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
    
