// -*- lsst-c++ -*-
/**
  * \class SupportFactory
  *
  * \ingroup fw
  *
  * \brief Factory class for Support objects
  *
  * Implements a factory object to create instances of the various
  * framework support classes. 
  *  
  * Since this is a singleton class, it provides no
  * callable constructors nor can it be used in an assignment statement. It 
  * cannot be used to derive another class. (It is 'final' or 'sealed').
  *
  * The primary rationale for this object is the assumption that support object 
  * creation will likely require a variety of non-trivial processes and
  * techniques (e.g. return a pooled object, retrieve from a library, etc.). In
  * its initial implementation, this class functions primarily as a place-holder
  * for methods that are expected to be required, and to drive the syntactic
  * form of client code.
  *
  * Usage (assuming using namespace lsst::fw):
  *       x = SupportFactory::factoryMethod(...);
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

#ifndef LSST_FW_SUPPORTFACTORY_H
#define LSST_FW_SUPPORTFACTORY_H

#include "lsst/fw/Utils.h"

#include "lsst/fw/Metadata.h"
#include "lsst/fw/Persistence.h"
#include "lsst/fw/Policy.h"
#include "lsst/fw/Provenance.h"
#include "lsst/fw/ReleaseProcess.h"
#include "lsst/fw/Security.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

class SupportFactory {
public:
    /**
      * \brief Construct a default instance of a Metadata object
      * \return A reference-counted pointer to the instance
      */
    static Metadata::sharedPtrType createMetadata();
    /**
      * \brief Construct a default instance of a Persistence object
      * \return A reference-counted pointer to the instance
      */
    static Persistence::sharedPtrType createPersistence();
    /**
      * \brief Construct a default instance of a Policy object
      * \return A reference-counted pointer to the instance
      */
    static Policy::sharedPtrType createPolicy();
    /**
      * \brief Construct a default instance of a Provenance object
      * \return A reference-counted pointer to the instance
      */
    static Provenance::sharedPtrType createProvenance();
    /**
      * \brief Construct a default instance of a ReleaseProcess object
      * \return A reference-counted pointer to the instance
      */
    static ReleaseProcess::sharedPtrType createReleaseProcess();
    /**
      * \brief Construct a default instance of a Security object
      * \return A reference-counted pointer to the instance
      */
    static Security::sharedPtrType createSecurity();

    /// A short string representation of an instance
    std::string toString();

private:
    SupportFactory();
    SupportFactory(const SupportFactory&);
    SupportFactory& operator= (const SupportFactory&);
    ~SupportFactory();
    static SupportFactory* _singleton;
    static SupportFactory& the();
};


LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)

#endif // LSST_FW_SUPPORTFACTORY_H

