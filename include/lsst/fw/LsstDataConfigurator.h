// -*- lsst-c++ -*-
/**
  * \class   LsstDataConfigurator
  *
  * \ingroup fw
  *
  * \brief   Configure the content of LsstData objects
  *
  * Implements a factory object to configure instances of LsstData realizations. 
  *  
  * Since this is a singleton class, it provides no
  * callable constructors nor can it be used in an assignment statement. It 
  * cannot be used to derive another class. (It is 'final' or 'sealed').
  *
  * Usage (assuming using namespace lsst::fw):
  *       LsstDataConfigurator::method(...);
  * 
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

#ifndef LSST_FW_LSSTDATACONFIGURATOR_H
#define LSST_FW_LSSTDATACONFIGURATOR_H

#include "lsst/fw/Utils.h"

#include "lsst/fw/Policy.h"
#include "lsst/fw/LsstData.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

class LsstDataConfigurator {
public:

    /**
      * \brief   Initialize the given LsstData object according to the
      *          content of the given Policy object
      * \param   data The LsstData object to initialize 
      * \param   policy The controlling policy object
      */
    static void configureSupport( LsstData* data, Policy* policy);
private:
    // All constructors/destructor, copy constructors, assignment operators
    // are private to preclude specialization and explicit creation
    // of the class
    LsstDataConfigurator();
    LsstDataConfigurator(const LsstDataConfigurator&);
    LsstDataConfigurator& operator= (const LsstDataConfigurator&);
    ~LsstDataConfigurator();
    // The singleton instance (initialized during 1st call to the()
    static LsstDataConfigurator* _singleton;
    // Returns reference to the singleton, creates on first call
    static LsstDataConfigurator& the();
};


LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)

#endif // LSST_FW_LSSTDATACONFIGURATOR_H
