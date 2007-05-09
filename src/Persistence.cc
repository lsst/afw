// -*- lsst-c++ -*-
//////////////////////////////////////////////////////////////////////////////
// Persistence.cc
// Implementation of class Persistence methods
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


#include "lsst/fw/Persistence.h"
#include "lsst/fw/Trace.h"

#include <string>
using namespace std;


#define EXEC_TRACE  20
static void execTrace( string s, int level = EXEC_TRACE ){
    lsst::fw::Trace( "fw.Persistence", level, s );
}


LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)


Persistence::Persistence() : Citizen( typeid(this) ){
    execTrace("Enter Persistence::Persistence()");
    execTrace( boost::str( 
        boost::format( 
            "Exit Persistence::Persistence() : %s") % this->toString()));
}


Persistence::Persistence(const Persistence& from) : Citizen( typeid(this) ){
    execTrace("Enter Persistence::Persistence(Persistence&)");
    execTrace("Exit Persistence::Persistence(Persistence&)");
}


Persistence::Persistence& Persistence::operator= (const Persistence& RHS){
    execTrace("Persistence::operator= (const Persistence&)");
    return *this;
}


Persistence::~Persistence(){
    execTrace( boost::str( 
        boost::format(
            "Enter Persistence::~Persistence() : %s") % this->toString()));
    execTrace("Exit Persistence::~Persistence()");
}


std::string Persistence::toString(){
    return repr();  // In Citizen
}


LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)

