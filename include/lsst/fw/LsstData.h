// -*- lsst-c++ -*-
/**
  * \class LsstData
  *
  * \ingroup fw
  *
  * \brief LsstData is the pure abstract base type (interface) for all core 
  *        LSST data types
  *
  * LsstData is the pure abstract base type for all core LSST data types that
  * have provenance and meta-data associated with them.  Classes that realize
  * LsstData are the core astronomical objects of the Framework.
  *
  * See LsstData package for realizations of this type.
  * 
  * \note 
  * 
  * \author  $Author::                                                        $
  * \version $Rev::                                                           $
  * \date    $Date::                                                          $
  * 
  * $Id::                                                                     $
  * 
  * Contact: Jeff Bartels (jeffbartels@usa.net)
  * 
  * Created: 24-Mar-2007 10:20:57 AM
  * 
  */

#ifndef LSST_FW_LSSTDATA_H
#define LSST_FW_LSSTDATA_H

#include <list>
#include <string>

#include "lsst/fw/Utils.h"

#include "lsst/fw/Metadata.h"
#include "lsst/fw/Persistence.h"
#include "lsst/fw/Policy.h"
#include "lsst/fw/Provenance.h"
#include "lsst/fw/ReleaseProcess.h"
#include "lsst/fw/Security.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

class LsstData {
public:
    /// Reference-counted pointer typedef for LsstData instances
    typedef boost::shared_ptr<LsstData *> sharedPtrType;
    /// The returned child collection type for all LsstData realizations
    typedef std::list<sharedPtrType> childCollectionType;
    
    /// Virtual destructor, pure virtual base class (see Stroustrup 12.4.2)
    virtual ~LsstData() {};
    
    /**
      * \brief All classes implementing this interface will return a 
      *        child collection in the form of std::list.
      * \param depth How deep below the current object to go in gathering the
      *              list of children.
      *              1 = this object only
      *              2 = this object plus the children of any of its children
      *              ...
      *              UINT_MAX = completely recurse all children of this object
      *                  (see limits.h)
      * \return std::list Contains reference-counted pointers to the
      *        LsstData objects aggregated by a given object. The return
      *        value may be NULL if the object does not aggregate children
      *        (i.e. is simple) or an empty list if the object may aggregate 
      *        children but currently has none.
      */
    virtual childCollectionType* getChildren( unsigned depth = 1 ) = 0;

    /**
      * \brief   Accessor for object's Metadata
      * \return  see lsst::fw::Metadata
      */
    virtual Metadata::sharedPtrType getMetadata() const =0;

    /**
      * \brief   Accessor for an LsstData instance's Persistence
      *          see lsst::fw::Persistence
      * \return 
      */
    virtual Persistence::sharedPtrType getPersistence() const =0;

    /**
      * \brief   Accessor for an LsstData instance's Policy
      * \return  see lsst::fw::Policy
      */
    virtual Policy::sharedPtrType getPolicy() const =0;

    /**
      * \brief   Accessor for an LsstData instance's Provenance
      * \return  see lsst::fw::Provenance
      */
    virtual Provenance::sharedPtrType getProvenance() const =0;

    /**
      * \brief   Accessor for an LsstData instance's ReleaseProcess
      * \return  see lsst::fw::ReleaseProcess
      */
    virtual ReleaseProcess::sharedPtrType getReleaseProcess() const =0;

    /**
      * \brief   Accessor for an LsstData instance's Security
      * \return  see lsst::fw::Security
      */
    virtual Security::sharedPtrType getSecurity() const =0;


    /**
      * \brief   Store the given Metadata object in an LsstData instance
      */
    virtual void setMetadata(Metadata::sharedPtrType metadata) =0;

    /**
      * \brief   Store the given Persistence object in an LsstData instance
      */
    virtual void setPersistence(Persistence::sharedPtrType persistence) =0;

    /**
      * \brief   Store the given Policy object in an LsstData instance
      */
    virtual void setPolicy(Policy::sharedPtrType policy) =0;

    /**
      * \brief   Store the given Provenance object in an LsstData instance
      */
    virtual void setProvenance(Provenance::sharedPtrType provenance) =0;

    /**
      * \brief   Store the given ReleaseProcess object in an LsstData instance
      */
    virtual void setReleaseProcess(ReleaseProcess::sharedPtrType release) =0;

    /**
      * \brief   Store the given Security object in an LsstData instance
      */
    virtual void setSecurity(Security::sharedPtrType security) =0;
 
    /**
      * \brief   Return a short string representation of an instance
      */
    virtual std::string toString() =0;
};
    
LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)

#endif // LSST_FW_LSSTDATA_H
