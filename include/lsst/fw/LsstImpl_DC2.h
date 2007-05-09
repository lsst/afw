// -*- lsst-c++ -*-
/**
  * \class LsstImpl_DC2
  *
  * \ingroup fw
  *
  * \brief The implementation of LsstImpl for DC2.
  *        
  *        While publicly available, it is intended that LsstData realizations
  *        will derive from LsstBase, and not LsstImpl_DC2. This indirection
  *        will isolate LsstData realizations from the exact base 
  *        implementation chosen for a given release of the framework.
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
  * Created:
  * 
  */

#ifndef LSST_FW_LSSTIMPL_DC2_H
#define LSST_FW_LSSTIMPL_DC2_H

#include <typeinfo>

#include "lsst/fw/LsstData.h"
#include "lsst/fw/Citizen.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

class LsstImpl_DC2 : public LsstData, private Citizen {
public:
    LsstImpl_DC2(const std::type_info & type);
    /// Virtual destructor, class may be specialized (see Stroustrup 12.4.2)
    virtual ~LsstImpl_DC2();

    /**
      * \brief   Base implementation lsst::fw:getChildren. May be overridden.
      *          Classes deriving from LsstImpl need implement a getChildren
      *          method only if they collect children.
      * \param   depth Specifies how deep to recurse the collection of
      *                children objects when creating the returned collection.
      *                see lsst::fw::LsstData.getChildren
      * \return  childCollectionType Will always return a value of 0  
      */
    virtual LsstData::childCollectionType* getChildren( unsigned depth = 1 );


    /**
      * \brief   Base implementation of lsst::fw::LsstData.getMetadata().
      *          May be overridden. Base implementation returns a 
      *          reference-counted pointer to the base object's cached 
      *          instance of Metadata.
      * \return  see lsst::fw::Metadata
      */
    virtual Metadata::sharedPtrType getMetadata() const;

    /**
      * \brief   Base implementation of lsst::fw::LsstData.getPersistence().
      *          May be overridden. Base implementation returns a 
      *          reference-counted pointer to the base object's cached 
      *          instance of Persistence.
      * \return  see lsst::fw::Persistence
      */
    virtual Persistence::sharedPtrType getPersistence() const;

    /**
      * \brief   Base implementation of lsst::fw::LsstData.getPolicy().
      *          May be overridden. Base implementation returns a 
      *          reference-counted pointer to the base object's cached 
      *          instance of Policy.
      * \return  see lsst::fw::Policy
      */
    virtual Policy::sharedPtrType getPolicy() const;

    /**
      * \brief   Base implementation of lsst::fw::LsstData.getProvenance().
      *          May be overridden. Base implementation returns a 
      *          reference-counted pointer to the base object's cached 
      *          instance of Provenance.
      * \return  see lsst::fw::Provenance
      */
    virtual Provenance::sharedPtrType getProvenance() const;

    /**
      * \brief   Base implementation of lsst::fw::LsstData.getReleaseProcess().
      *          May be overridden. Base implementation returns a 
      *          reference-counted pointer to the base object's cached 
      *          instance of ReleaseProcess.
      * \return  see lsst::fw::ReleaseProcess
      */
    virtual ReleaseProcess::sharedPtrType getReleaseProcess() const;

    /**
      * \brief   Base implementation of lsst::fw::LsstData.getSecurity().
      *          May be overridden. Base implementation returns a 
      *          reference-counted pointer to the base object's cached 
      *          instance of Security.
      * \return  see lsst::fw::Security
      */
    virtual Security::sharedPtrType getSecurity() const;

    /**
      * \brief   Base implementation of lsst::fw::setMetadata(). 
      *          May be overridden. Assigns the given metadata
      *          object to the base object's private data member. May result in
      *          the destruction of the currently-cached member object since
      *          the data member is a reference-counted pointer.
      */
    virtual void setMetadata(Metadata::sharedPtrType metadata);

    /**
      * \brief   Base implementation of lsst::fw::setPersistence(). 
      *          May be overridden. Assigns the given Persistence
      *          object to the base object's private data member. May result in
      *          the destruction of the currently-cached member object since
      *          the data member is a reference-counted pointer.
      */
    virtual void setPersistence(Persistence::sharedPtrType persistence);

    /**
      * \brief   Base implementation of lsst::fw::setPolicy(). 
      *          May be overridden. Assigns the given Policy
      *          object to the base object's private data member. May result in
      *          the destruction of the currently-cached member object since
      *          the data member is a reference-counted pointer.
      */
    virtual void setPolicy(Policy::sharedPtrType policy);

    /**
      * \brief   Base implementation of lsst::fw::setProvenance(). 
      *          May be overridden. Assigns the given Provenance
      *          object to the base object's private data member. May result in
      *          the destruction of the currently-cached member object since
      *          the data member is a reference-counted pointer.
      */
    virtual void setProvenance(Provenance::sharedPtrType provenance);

    /**
      * \brief   Base implementation of lsst::fw::setReleaseProcess(). 
      *          May be overridden. Assigns the given ReleaseProcess
      *          object to the base object's private data member. May result in
      *          the destruction of the currently-cached member object since
      *          the data member is a reference-counted pointer.
      */
    virtual void setReleaseProcess(ReleaseProcess::sharedPtrType release);

    /**
      * \brief   Base implementation of lsst::fw::setSecurity(). 
      *          May be overridden. Assigns the given Security
      *          object to the base object's private data member. May result in
      *          the destruction of the currently-cached member object since
      *          the data member is a reference-counted pointer.
      */
    virtual void setSecurity(Security::sharedPtrType security);
 
    /**
      * \brief   Base implementation of lsst::fw::toString(). 
      *          May be overridden. Returns a short string representation
      *          of the object as implemented by Citizen::repr().
      */
    virtual std::string toString();

private:
    Metadata::sharedPtrType _metadata;
    Persistence::sharedPtrType _persistence;
    Policy::sharedPtrType _policy;
    Provenance::sharedPtrType _provenance;
    ReleaseProcess::sharedPtrType _releaseProcess;
    Security::sharedPtrType _security;

};

LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)

#endif // LSST_FW_LSSTIMPL_DC2_H
