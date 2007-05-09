// -*- lsst-c++ -*-
/**
  * \class Provenance
  *
  * \ingroup fw
  *
  * \brief A type of Metadata that captures the processing history 
  * of an LsstData realization.
  * 
  * \note This version is a stub implementation of the class
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

#ifndef LSST_FW_PROVENANCE_H
#define LSST_FW_PROVENANCE_H

#include <boost/shared_ptr.hpp>

#include "lsst/fw/Utils.h"
#include "lsst/fw/Citizen.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

class Provenance : private Citizen {

public:
    /// Default constructor
    Provenance();
    /// Copy initialization semantics (NIL in this revision)
    Provenance(const Provenance&);
    /// Copy assignment semantics (NIL in this revision)
    Provenance& operator= (const Provenance&);
    /// Virtual destructor, class may be specialized (see Stroustrup 12.4.2)
    virtual ~Provenance();

    /// Reference-counted pointer typedef forinstances of this class
    typedef boost::shared_ptr<Provenance> sharedPtrType;
    
    /// A short string representation of an instance
    std::string toString();
};
    
LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)

#endif // LSST_FW_PROVENANCE_H
