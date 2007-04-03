// -*- lsst-c++ -*-
/*
 * \file
 * basic run-time trace facilities
 *
 *  \author Robert Lupton, Princeton University
 */
#if !defined(LSST_FW_TRACE_H)
#define LSST_FW_TRACE_H 1

#include <iostream>
#include <string>
#include <boost/format.hpp>

#include "Utils.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)
LSST_START_NAMESPACE(Trace)

#if !defined(LSST_NO_TRACE)
#  define LSST_NO_TRACE 0               //!< True => turn off all tracing
#endif

#if LSST_NO_TRACE
inline void trace(const std::string &comp, const int verbosity,
                  const std::string &msg) {}
inline void trace(const std::string &comp, const int verbosity,
                  const boost::format &msg) {}
#else
void trace(const std::string &comp, const int verbosity,
           const std::string &msg);
void trace(const std::string &comp, const int verbosity,
           const boost::format &msg);
#endif

void reset();
void setVerbosity(const std::string &name);
void setVerbosity(const std::string &name, const int verbosity);

int getVerbosity(const std::string &name);

void printVerbosity(std::ostream &fp);

void setDestination(std::ostream &fp);

LSST_END_NAMESPACE(Trace)
LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
#endif
