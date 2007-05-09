// -*- lsst-c++ -*-
/**
  * \class Metadata
  *
  * \ingroup fw
  *
  * \brief  Metadata is a means of recording/retrieving Metadata for an LsstData 
  * realization.
  * 
  * \note This verision is a stub implementation of the Metadata class
  * 
  * \author  $Author::                                                    $
  * \version $Rev::                                                       $
  * \date    $Date::                                                      $
  * 
  * $Id::                                                                 $
  * 
  * Contact: Jeff Bartels (jeffbartels@usa.net)
  * 
  * Created:
  * 
  */

#ifndef LSST_FW_METADATA_H
#define LSST_FW_METADATA_H

#include <boost/shared_ptr.hpp>

#include "lsst/fw/Utils.h"
#include "lsst/fw/Citizen.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

class Metadata : private Citizen {
public:
    /// Default constructor
    Metadata();
    /// Copy initialization semantics (NIL in this revision)
    Metadata(const Metadata&);
    /// Copy assignment semantics (NIL in this revision)
    Metadata& operator= (const Metadata&);
   /// Virtual destructor, class may be specialized (see Stroustrup 12.4.2)
    virtual ~Metadata();

    /// Reference-counted pointer typedef forinstances of this class
    typedef boost::shared_ptr<Metadata> sharedPtrType;
    
    /// A short string representation of an instance
    std::string toString();
};
    
LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)

#endif // LSST_FW_METADATA_H

