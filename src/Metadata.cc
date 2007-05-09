// -*- lsst-c++ -*-
//////////////////////////////////////////////////////////////////////////////
// Metadata.cc
// Implementation of class Metadata methods
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


#include "lsst/fw/Metadata.h"
#include "lsst/fw/Trace.h"

#include <string>
using namespace std;


#define EXEC_TRACE  20
static void execTrace( string s, int level = EXEC_TRACE ){
    lsst::fw::Trace( "fw.Metadata", level, s );
}


LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)


Metadata::Metadata() : Citizen( typeid(this) ){
    execTrace("Enter Metadata::Metadata()");
    execTrace( boost::str( 
        boost::format( 
            "Exit Metadata::Metadata() : %s") % this->toString()));
}


Metadata::Metadata(const Metadata& from) : Citizen( typeid(this) ){
    execTrace("Enter Metadata::Metadata(Metadata&)");
    execTrace("Exit Metadata::Metadata(Metadata&)");
}


Metadata::Metadata& Metadata::operator= (const Metadata& RHS){
    execTrace("Metadata::operator=(const Metadata&)");
    return *this;
}


Metadata::~Metadata(){
    execTrace( boost::str( 
        boost::format(
            "Enter Metadata::~Metadata() : %s") % this->toString()));
    execTrace("Exit Metadata::~Metadata()");
}


std::string Metadata::toString(){
    return repr();  // In Citizen
}


LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
