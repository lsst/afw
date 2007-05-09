// -*- lsst-c++ -*-
/*
  * \class LsstBase
  *
  * \ingroup fw
  *
  * \brief Base class implementation for all LsstData derived types
  *
  * LsstBase establishes the true base class implementation for given release
  * of the framework package. All LsstData relizations in that release of the
  * framework will derive from LsstBase.
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
  * Created: 03-Apr-2007 5:30:00 PM
  * 
  */

#ifndef LSST_FW_LSSTBASE_H            //! multiple inclusion guard macro
#define LSST_FW_LSSTBASE_H

#include "lsst/fw/LsstImpl_DC2.h"
#include "Utils.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

class LsstBase : public LsstImpl_DC2 {
public:
    /**
      * \brief Construct an instance of the concrete base class for the
      *        current revision of the framework
      * \param type A std::type_info required by the Citizen base class.
      *        (Obtain with a call to std::typeid(...) )
      */
    LsstBase(const std::type_info &type): LsstImpl_DC2(type) {};

    /// Virtual destructor, class may be specialized (see Stroustrup 12.4.2)
    virtual ~LsstBase() {};
};
    
LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)

#endif  // LSST_FW_LSSTBASE_H

