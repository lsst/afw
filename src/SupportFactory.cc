// -*- lsst-c++ -*-
//////////////////////////////////////////////////////////////////////////////
// SupportFactory.cc
// Implementation of class SupportFactory methods
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


#include "lsst/fw/SupportFactory.h"

#include "lsst/fw/Utils.h"
#include "lsst/fw/Trace.h"

#include <string>
using namespace std;


LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)


SupportFactory* SupportFactory::_singleton = 0;


#define EXEC_TRACE  20
static void execTrace( string s, int level = EXEC_TRACE ){
    lsst::fw::Trace( "fw.SupportFactory", level, s );
}


SupportFactory& SupportFactory::the() {
    if( _singleton == 0 )
    {
        execTrace( "SupportFactory::the() - creating singleton" );
        _singleton = new SupportFactory();
    }
    return *(_singleton);
}


SupportFactory::SupportFactory(){
    execTrace( "Enter SupportFactory::SupportFactory()" );
    execTrace( "Exit SupportFactory::SupportFactory()" );
}


SupportFactory::~SupportFactory(){
    execTrace( "Enter SupportFactory::~SupportFactory()");
    execTrace( "Exit SupportFactory::~SupportFactory()" );
}


SupportFactory::SupportFactory(const SupportFactory&){
}


SupportFactory::SupportFactory& SupportFactory::operator= (const SupportFactory&){
    return the();
}


Metadata::sharedPtrType SupportFactory::createMetadata(){
    the(); // insure singelton is initialized
    execTrace( "Enter SupportFactory::createMetadata()" );
    Metadata::sharedPtrType ret = Metadata::sharedPtrType( new Metadata );
    execTrace( "Exit SupportFactory::createMetadata()" );
    return ret; 
}


Persistence::sharedPtrType SupportFactory::createPersistence(){
    the(); // insure singelton is initialized
    execTrace( "Enter SupportFactory::createPersistence()" );
    Persistence::sharedPtrType ret = Persistence::sharedPtrType( new Persistence );
    execTrace( "Exit SupportFactory::createPersistence()" );
    return ret;
}


Policy::sharedPtrType SupportFactory::createPolicy(){
    the(); // insure singelton is initialized
    execTrace( "Enter SupportFactory::createPolicy()" );
    Policy::sharedPtrType ret = Policy::sharedPtrType( new Policy );
    execTrace( "Exit SupportFactory::createPolicy()" );
    return ret;
}


Provenance::sharedPtrType SupportFactory::createProvenance(){
    the(); // insure singelton is initialized
    execTrace( "Enter SupportFactory::createProvenance()" );
    Provenance::sharedPtrType ret = Provenance::sharedPtrType( new Provenance );
    execTrace( "Exit SupportFactory::createProvenance()" );
    return ret;
}


ReleaseProcess::sharedPtrType SupportFactory::createReleaseProcess(){
    the(); // insure singelton is initialized
    execTrace( "Enter SupportFactory::createReleaseProcess()" );
    ReleaseProcess::sharedPtrType ret = ReleaseProcess::sharedPtrType( new ReleaseProcess );
    execTrace( "Exit SupportFactory::createReleaseProcess()" );
    return ret;
}


Security::sharedPtrType SupportFactory::createSecurity(){
    the(); // insure singelton is initialized
    execTrace( "Enter SupportFactory::createSecurity()" );
    Security::sharedPtrType ret = Security::sharedPtrType( new Security );
    execTrace( "Exit SupportFactory::createSecurity()" );
    return ret;
}


LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)

