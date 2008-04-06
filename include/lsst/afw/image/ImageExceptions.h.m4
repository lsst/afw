// -*- lsst-c++ -*-
#ifndef LSST_AFW_IMAGE_EXCEPTIONS
#define LSST_AFW_IMAGE_EXCEPTIONS
dnl "dnl" is the m4 comment syntax
dnl
undefine(`format')dnl ' Stop m4 expanding format
//
dnl comment to go into the output file
// This file is machine generated from Runtime.h.m4. Please do not edit directly
//         
dnl
/** \file  fwExceptions.h
  * \brief LSST exception handler classes for generic runtime exception events
  *
  * When LSST Data Managment software forces (i.e. throws) an exception 
  * event, an LSST exception handler (e.g. OutOfRange) must be invoked.  
  * When DM software catches an exception event, the software either 
  * satisfactorily resolves the event and continues; or ensures that 
  * regardless of the type of event caught, the object re-thrown is an LSST 
  * exception handler.
  *
  * An LSST exception handler manages an LSST ExceptionStack composed of 
  * an ordered collection of LSST ExceptionDatas. Each ExceptionData within the 
  * ExceptionStack represents the occurence of an exception event which 
  * couldn't be resolved and was rethrown in anticipation the next level 
  * would succeed. When a handler successfully resolves the event and 
  * normal processing continues, the exception stack goes out of scope and 
  * is deleted.
  
  * Each ExceptionData is composed of attributes providing additional debugging 
  * information to the exception handler catching that specific 
  * error event. Those attributes are specified using the DataProperty class.
  * An ExceptionData is a collection of DataProperty objects.
  
  * When the current exception handler prepares to throw an exception, 
  * it creates an LSST ExceptionData object, adds it to the LSST ExceptionStack 
  * originally provided (or creates a new LSST ExceptionStack), 
  * then initializes the appropriate LSST exception handler with the 
  * ExceptionStack. Finally the current handler throws the newly
  * initialized exception handler.
  *
  * See: DataProperty, ExceptionData, ExceptionStack.
  *
  *   -  NoMaskPlane : Failure to find specified Mask Plane (to be relocated)
  *   -  OutOfPlaneSpace :  Insufficient Plane allocation (to be relocated)
  *
  * \ingroup afw
  *
  * \ author Robert Lupton
  * \ author Roberta Allsman
  */
#include <exception>

#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>

#include <lsst/daf/base.h>
#include <lsst/pex/exceptions.h>


namespace lsst {
namespace afw {
namespace image {

        dnl
        dnl define(name, body) defines an m4 macro
        dnl Define a new subclass $1 of ExceptionData without added functionality; docstring $2
        define(LSST_NEW_EXCEPTION,
`
/** \brief $1 handler thrown or caught on an exceptional event.
  *
  * The handlers "catch" indicates the following situation occurred: 
  * $2
  */
class $1 : public lsst::pex::exceptions::ExceptionStack  {
public:
  /** Construct new $1 object.
    *
    * \param comment  String used to initialize std::runtime_error.what(). 
    * \note         Default ExceptionStack and ExceptionData will be created
    */
    $1(std::string const& comment) throw() :
         lsst::pex::exceptions::ExceptionStack(std::string("$1"),comment) {
         lsst::pex::exceptions::ExceptionData eNode("Node$1");
         this->addExceptionData(eNode);
    };


  /** Construct new $1 object.
    *
    * \param comment  String used to initialize std::runtime_error.what(). 
    * \note         Default ExceptionStack and ExceptionData will be created
    */
    $1(boost::format const& comment) throw() :
         lsst::pex::exceptions::ExceptionStack( std::string("$1"),comment.str()) {
         lsst::pex::exceptions::ExceptionData eNode("Node$1");
         this->addExceptionData(eNode);
    };

  /** Construct new $1 object.
    *
    * \param orig     Reference to ExceptionData to be cloned for use in ExceptionStack.
    * \param comment  String used to initialize std::runtime_error.what(). 
    *                 Default to: "$2".
    */
    $1(lsst::pex::exceptions::ExceptionData &orig, std::string const& comment=std::string("$2")) 
        throw() :
        lsst::pex::exceptions::ExceptionStack( std::string("$1"), comment) {
        this->addExceptionData(orig);
    };

  /** Construct new $1 object.
    *
    * \param orig     Reference to ExceptionData to be cloned for use in ExceptionStack.
    * \param comment  String used to initialize std::runtime_error.what(). 
    *                 Default to: "$2".
    */
    $1(lsst::pex::exceptions::ExceptionData &orig, boost::format const& comment=boost::format("$2")) 
        throw() :
        lsst::pex::exceptions::ExceptionStack(orig, std::string("$1"), comment.str()) {
        this->addExceptionData(orig);
    };

  /** Construct new $1 object.
    *
    * \param orig     Reference to ExceptionStack to be cloned for use.
    * \param comment  String used to initialize std::runtime_error.what(). 
    *                 Default to: "$2".
    */
    $1(lsst::pex::exceptions::ExceptionStack const &orig, std::string const& comment=std::string("$2")) throw() :
         lsst::pex::exceptions::ExceptionStack( orig, std::string("$1"), comment) {
    };

  /** Construct new $1 object.
    *
    * \param orig     Reference to ExceptionStack to be cloned for use.
    * \param comment  String used to initialize std::runtime_error.what(). 
    *                 Default to: "$2".
    */
    $1(lsst::pex::exceptions::ExceptionStack const &orig, boost::format const& comment=boost::format("$2")) throw() :
         lsst::pex::exceptions::ExceptionStack(orig, std::string("$1"), comment.str()){
    };


  /** $1 copy constructor.  This is a clone operation.
    *
    * \param orig  A reference to the $1 object to clone.
    */
    $1(const $1 & orig) throw() :
         lsst::pex::exceptions::ExceptionStack(orig )  {};

  /** operator<< 
    *
    * \param rhs     Reference to ExceptionData to be added to $1 ExceptionStack .
    */
    $1 &operator<< (lsst::pex::exceptions::ExceptionData  rhs) throw() {
        this->getStack()->addProperty(rhs.getExceptionData());
        return *this;
    }

  /** operator<< 
    *
    * \param rhs     Reference to DataProperty to be added to most recent ExceptionData on ExceptionStack.
    */
    $1 &operator<< (lsst::daf::base::DataProperty  const rhs) throw() {
        this->getLast()->addProperty(rhs);
        return *this;
    }

  /** operator<< 
    *
    * \param rhs     Reference to DataProperty to be added to most recent ExceptionData on ExceptionStack.
    */
    $1 &operator<< (lsst::daf::base::DataProperty::PtrType  const rhs) throw() {
        this->getLast()->addProperty(rhs);
        return *this;
    }


  /** Destructor for $1 object
    */
    ~$1() throw() {};

    virtual char const * getPythonModule() const { return "lsst.fw.exceptions"; }
    virtual char const * getPythonClass()  const { return "Lsst$1"; }

}')

// Mapping  package errors 
    LSST_NEW_EXCEPTION(NoMaskPlane,
                       Failure to find specified Mask Plane);

    LSST_NEW_EXCEPTION(OutOfPlaneSpace,
                       Insufficient Plane allocation);

}}} // namespace lsst::afw::image
#endif // LSST_AFW_IMAGE_EXCEPTIONS
