// -*- lsst-c++ -*-
/**
  * \class Policy
  *
  * \ingroup fw
  *
  * \brief A set of rules and parameters that govern the processing or 
  * handling of an LsstData realization.
  *
  * \note This version is a stub implementation of the Class Policy
  * 
  * \author  $Author::                                                        $
  * \version $Rev::                                                           $
  * \date    $Date::                                                          $
  * 
  * $Id::                                                                     $
  * 
  * Contact: Jeff Bartels (jeffbartels@usa.net)
  * 
  * Created:
  * 
  */

#ifndef LSST_FW_POLICY_H
#define LSST_FW_POLICY_H

#include <boost/shared_ptr.hpp>
#include <string>
using namespace std;

#include "lsst/fw/Utils.h"
#include "lsst/fw/Citizen.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

class Policy : private Citizen {
public:
    /// Default constructor
    Policy();
    /// Copy initialization semantics (NIL in this revision)
    Policy(const Policy&);
    /// Copy assignment semantics (NIL in this revision)
    Policy& operator= (const Policy&);
     
    /// Virtual destructor, class may be specialized (see Stroustrup 12.4.2)
    virtual ~Policy();

    /// Reference-counted pointer typedef forinstances of this class
    typedef boost::shared_ptr<Policy> sharedPtrType;
    
    /// A short string representation of an instance
    std::string toString();
};
    
LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)

#endif // LSST_FW_POLICY_H
