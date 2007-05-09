// -*- lsst-c++ -*-
/**
  * \class Security
  *
  * \ingroup fw
  *
  * \brief Implements the logic that enforces the access and authorization 
  *  rules that apply to an LsstData Realization.
  * 
  * 
  * \note OUT OF SCOPE FOR DC2 - stub implementation
  * 
  * \author  $Author::                                                        $
  * \version $Rev::                                                           $
  * \date    $Date::                                                          $
  * 
  * $Id::                                                                     $
  * 
  * Contact: Jeff Bartels (jeffbartels@usa.net)
  * 
  * Created: 03-Apr-2007 5:30:00 PM
  * 
  */

#ifndef LSST_FW_SECURITY_H
#define LSST_FW_SECURITY_H

#include <boost/shared_ptr.hpp>

#include "lsst/fw/Utils.h"
#include "lsst/fw/Citizen.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

class Security : private Citizen {
public:
    /// Default constructor
    Security();
    /// Copy initialization semantics (NIL in this revision)
    Security(const Security&);
    /// Copy assignment semantics (NIL in this revision)
    Security& operator= (const Security&);
    /// Virtual destructor, class may be specialized (see Stroustrup 12.4.2)
    virtual ~Security();

    /// Reference-counted pointer typedef forinstances of this class
    typedef boost::shared_ptr<Security> sharedPtrType;
    
    /// A short string representation of an instance
    std::string toString();
};
    
LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)

#endif // LSST_FW_SECURITY_H
