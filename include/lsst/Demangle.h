// -*- lsst-c++ -*-
#if !defined(LSST_DEMANGLE)             //! multiple inclusion guard macro
#define LSST_DEMANGLE 1

#include "Utils.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

std::string demangleType(const std::string _typeName);
    
LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
#endif
