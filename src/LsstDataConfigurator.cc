// -*- lsst-c++ -*-
//////////////////////////////////////////////////////////////////////////////
// LsstDataConfigurator.cc
// Implementation of class LsstDataConfigurator methods
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


#include "lsst/fw/LsstDataConfigurator.h"
#include "lsst/fw/Trace.h"
#include "lsst/fw/Utils.h"

#include <string>
using namespace std;


LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)


#define EXEC_TRACE  20
static void execTrace( string s ){
    lsst::fw::Trace( "fw.LsstDataConfigurator", EXEC_TRACE, s );
}


LsstDataConfigurator* LsstDataConfigurator::_singleton = 0;


LsstDataConfigurator& LsstDataConfigurator::the(){
    if( _singleton == 0 ){
        execTrace( "LsstDataConfigurator::the() - creating singleton");
        _singleton = new LsstDataConfigurator();
    }
    return *(_singleton);
}


LsstDataConfigurator::LsstDataConfigurator(){
    execTrace( "Enter LsstDataConfigurator::LsstDataConfigurator()");
    execTrace( "Exit LsstDataConfigurator::LsstDataConfigurator()") ;
}


LsstDataConfigurator::~LsstDataConfigurator(){
    execTrace( "Enter LsstDataConfigurator::~LsstDataConfigurator()");
    execTrace( "Exit LsstDataConfigurator::~LsstDataConfigurator()");
}


LsstDataConfigurator::LsstDataConfigurator(const LsstDataConfigurator&){
}


LsstDataConfigurator::LsstDataConfigurator& LsstDataConfigurator::operator= (const LsstDataConfigurator&){
	return the();
}


void LsstDataConfigurator::configureSupport( LsstData* data, Policy* policy){
    execTrace( 
        boost::str( 
            boost::format( "LsstDataConfigurator::configureSupport(%s)") % 
                policy->toString() ) );
    return;
}


LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
