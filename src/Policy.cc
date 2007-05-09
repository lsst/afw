// -*- lsst-c++ -*-
//////////////////////////////////////////////////////////////////////////////
// Policy.cc
// Implementation of class Policy methods
//
// $Author::                                                                 $
// $Rev::                                                                    $
// $Date::                                                                   $
// $Id::                                                                     $
// 
// Contact: Jeff Bartels (jeffbartels@usa.net)
// 
// Created: 03-Apr-2007 5:30:00 PM
//////////////////////////////////////////////////////////////////////////////


#include "lsst/fw/Policy.h"
#include "lsst/fw/Trace.h"

#include <string>
using namespace std;


#define EXEC_TRACE  20
static void execTrace( string s, int level = EXEC_TRACE ){
    lsst::fw::Trace( "fw.Policy", level, s );
}


LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)


Policy::Policy() : Citizen( typeid(this) ){
    execTrace("Enter Policy::Policy()");
    execTrace( boost::str( 
        boost::format ("Exit Policy::Policy() : %s") % this->toString()));
}


Policy::~Policy(){
   execTrace( boost::str( 
       boost::format(
           "Enter Policy::~Policy() : %s") % this->toString()));
   execTrace("Exit Policy::~Policy()");
}


Policy::Policy(const Policy&) : Citizen( typeid(this) ){
    execTrace("Enter Policy::Policy(const Policy&)");
    execTrace( boost::str( 
        boost::format ("Created Policy : %s") % this->toString()),
            EXEC_TRACE + 5 );
    execTrace("Exit Policy::Policy(const Policy&)");
}


Policy::Policy& Policy::operator= (const Policy&){
    execTrace("Policy::operator= (const Policy&)");
    return *this;
}


std::string Policy::toString(){
    return repr();  // In Citizen
}


LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)

