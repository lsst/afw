// -*- lsst-c++ -*-
#if !defined(LSST_LSST_BASE)            //! multiple inclusion guard macro
#define LSST_LSST_BASE 1

#include "lsst/Citizen.h"
#include "Utils.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

class LsstBase : private Citizen {
public:
    LsstBase(const std::type_info &type) : Citizen(type) {}
};
    
LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
#endif
