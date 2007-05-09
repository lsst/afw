// -*- lsst-c++ -*-
/**
  * \class Persistence
  *
  * \ingroup fw
  *
  * \brief Provides the abstract interface for saving and retrieving LsstData 
  * to and from storage.  
  *
  * Persistence schedules the writing of an LsstData realization in one or 
  * more formats according to a pre-defined Policy and ensures its completion 
  * as a transaction.
  * 
  * \note This version is a stub implementation of the Class Persistence
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

#ifndef LSST_FW_PERSISTENCE_H
#define LSST_FW_PERSISTENCE_H

#include <boost/shared_ptr.hpp>

#include "lsst/fw/Utils.h"
#include "lsst/fw/Citizen.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

class Persistence : private Citizen {
public:
    /// Default constructor
    Persistence();
    /// Copy initialization semantics (NIL in this revision)
    Persistence(const Persistence&);
    /// Copy assignment semantics (NIL in this revision)
    Persistence& operator= (const Persistence&);
    /// Virtual destructor, class may be specialized (see Stroustrup 12.4.2)
    virtual ~Persistence();

    /// Reference-counted pointer typedef forinstances of this class
    typedef boost::shared_ptr<Persistence> sharedPtrType;
    
    /// A short string representation of an instance
    std::string toString();
};
    
LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)

#endif // LSST_FW_PERSISTENCE_H
