// -*- lsst-c++ -*-
/** \file Log.h
  *
  * \ingroup fw
  *
  * \brief Log provides basic run-time logging facilities.
  *
  * \author Robert Lupton, Princeton University
  * \author Robyn Allsman, LSST Corp
  */

#if !defined(LSST_FW_LOG_H)                      //!< multiple inclusion guard macro
#define LSST_FW_LOG_H 1

#include <iostream>
#include <string>
#include <sstream>
#include <boost/format.hpp>

#include "Utils.h"

#include "lsst/fw/DataProperty.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

/**
 * \brief  Log provides basic run-time logging facilities.
 *
 *        Log creates run-time status records for post-processing by
 *        performance analysis tools.  The structured format of Log records 
 *        is more easily parsed than the free-format Trace records.
 *
 *        Generation of a log record is controlled on a per "component" 
 *        basis, where a "component" is a name of the form aaa.bbb.ccc 
 *        where aaa is the Most significant part; for example, 
 *        the utilities library might be called "utils", the doubly-linked 
 *        list "utils.dlist", and code to destroy a list "utils.dlist.del". 
 *
 * \see Component class for details on the verbosity tree which
 *      determines when a trace record will be emitted.
 */
class Log {
public:
    Log(const std::string& name, const int verbosity) ;
    Log(const std::string& name, const int verbosity, const DataPropertyPtrT dp) ;
    Log(const std::string& name, const int verbosity, const DataProperty& dp) ;

    /** Add user attributes to current Log record
      */
    Log& operator<<(DataPropertyPtrT dp) {
        if (_print) {
            log(dp->treeNode("", ": "));
        }
        return *this;
    }

    /** \overload Log& operator<< (DataProperty::DataProperty dp)
      */
    Log& operator<< (DataProperty dp) {
        if (_print) {
            log(dp.treeNode("", ": "));
        }
        return *this;
    }

    static void reset();
    static void setVerbosity(const std::string &name);
    static void setVerbosity(const std::string &name, const int verbosity);
    static void setDestination(std::ostream &fp);

    static int  getVerbosity(const std::string &name);
    static void printVerbosity(std::ostream &fp = std::cout);

private:
    bool    _print;
    int     _verbosity;

    bool check_level(const std::string& name, const int verbosity);
    void log(const std::string& msg);
    void log(const boost::format &msg);
    void logDate();
};

LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
#endif
