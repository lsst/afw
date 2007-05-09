// -*- lsst-c++ -*-
//////////////////////////////////////////////////////////////////////////////
// ReleaseProcess.cc
// Implementation of class ReleaseProcess methods
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


#include "lsst/fw/ReleaseProcess.h"
#include "lsst/fw/Trace.h"

#include <string>
using namespace std;


#define EXEC_TRACE  20
static void execTrace( string s, int level = EXEC_TRACE ){
    lsst::fw::Trace( "fw.ReleaseProcess", level, s );
}

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)


ReleaseProcess::ReleaseProcess() : Citizen( typeid(this) ){
    execTrace("Enter ReleaseProcess::ReleaseProcess()");
    execTrace( boost::str( 
        boost::format( 
            "Exit ReleaseProcess::ReleaseProcess() : %s") % this->toString()));
}


ReleaseProcess::ReleaseProcess(const ReleaseProcess& from) : Citizen( typeid(this) ){
    execTrace("Enter ReleaseProcess::ReleaseProcess(ReleaseProcess&)");
    execTrace("Exit ReleaseProcess::ReleaseProcess(ReleaseProcess&)");
}


ReleaseProcess::ReleaseProcess& ReleaseProcess::operator= (const ReleaseProcess&){
    execTrace("ReleaseProcess::operator= (const ReleaseProcess&)");
    return *this;
}


ReleaseProcess::~ReleaseProcess(){
    execTrace( boost::str( 
        boost::format(
            "Enter ReleaseProcess::~ReleaseProcess() : %s") % this->toString()));
    execTrace("Exit ReleaseProcess::~ReleaseProcess()");
}


std::string ReleaseProcess::toString(){
    return repr();  // In Citizen
}


LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
