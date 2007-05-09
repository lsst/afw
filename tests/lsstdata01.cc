// -*- lsst-c++ -*-
//////////////////////////////////////////////////////////////////////////////
// lsstdata01.cc
//      fw classes sanity check program (see comments below)
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


// 
// lsstdata01.cc demonstrates specialization on LsstBase, 
// object creation and configuration, exercising the factories, 
// the required fw includes, use of trace verbosity to control debug 
// output
//
// This program should be built by scons when run from the overlying
//    directory.
// Run this program to do a sanity check on the fw classes. 
// Examine the output of the program for the following:
// 1) Proper pairing of constructors/destructors, 
// 2) Only one initialization of singleton classes SupportFactory and
//    LsstDataConfigurator
// 3) The last item output should be the message "No memory leaks detected"
// 4) The program should run with no errors. 
// 5) You can specify a trace verbosity level with the command line (default = 100)
// 
// With that check out of the way, you are ready to use the fw classes to 
// develop your LsstBase derivations and application code
//
// SAMPLE OUTPUT (verbosity=100):
//
// Explicitly creating a policy object
//                     SupportFactory::the() - creating singleton
//                     Enter SupportFactory::SupportFactory()
//                     Exit SupportFactory::SupportFactory()
//                     Enter SupportFactory::createPolicy()
//                     Enter Policy::Policy()
//                     Exit Policy::Policy() : 1: 0x985454c lsst::fw::Policy
//                     Exit SupportFactory::createPolicy()
// Done: Created policy object '1: 0x985454c lsst::fw::Policy'
// Creating an LsstData realization
//                     Enter LsstImpl_DC2::LsstImpl_DC2(P6MyLsst)
//                     Enter SupportFactory::createMetadata()
//                     Enter Metadata::Metadata()
//                     Exit Metadata::Metadata() : 3: 0x98545bc lsst::fw::Metadata
//                     Exit SupportFactory::createMetadata()
//                     Enter SupportFactory::createPersistence()
//                     Enter Persistence::Persistence()
//                     Exit Persistence::Persistence() : 4: 0x98548ec lsst::fw::Persistence
//                     Exit SupportFactory::createPersistence()
//                     Enter SupportFactory::createPolicy()
//                     Enter Policy::Policy()
//                     Exit Policy::Policy() : 5: 0x98547bc lsst::fw::Policy
//                     Exit SupportFactory::createPolicy()
//                     Enter SupportFactory::createProvenance()
//                     Enter Provenance::Provenance()
//                     Exit Provenance::Provenance() : 6: 0x9854614 lsst::fw::Provenance
//                     Exit SupportFactory::createProvenance()
//                     Enter SupportFactory::createReleaseProcess()
//                     Enter ReleaseProcess::ReleaseProcess()
//                     Exit ReleaseProcess::ReleaseProcess() : 7: 0x985457c lsst::fw::ReleaseProcess
//                     Exit SupportFactory::createReleaseProcess()
//                     Enter SupportFactory::createSecurity()
//                     Enter Security::Security()
//                     Exit Security::Security() : 8: 0x9854714 lsst::fw::Security
//                     Exit SupportFactory::createSecurity()
//                     Exit LsstImpl_DC2::LsstImpl_DC2() : 2: 0x9854664 MyLsst
//           In MyLsst::MyLsst(An LsstData realization)
//  Done: Created MyLsst object '2: 0x936f754 MyLsst'
//  Configuring the LsstData object
//                     LsstDataConfigurator::configureSupport(1: 0x985454c lsst::fw::Policy)
//  Configured MyLsst object '2: 0x9854664 MyLsst'
//                     Enter LsstImpl_DC2::getChildren(1)
//                     Exit LsstImpl_DC2::getChildren()
//  OK: MyLsst::getChildren returned 0
//  Destroy MyLsst object '2: 0x9854664 MyLsst'
//           MyLsst::~MyLsst()
//                     Enter LsstImpl_DC2::~LsstImpl_DC2() : 2: 0x9854664 MyLsst
//                     Exit LsstImpl_DC2::~LsstImpl_DC2()
//                     Enter Security::~Security() : 8: 0x9854714 lsst::fw::Security
//                     Exit Security::~Security()
//                     Enter ReleaseProcess::~ReleaseProcess() : 7: 0x985457c lsst::fw::ReleaseProcess
//                     Exit ReleaseProcess::~ReleaseProcess()
//                     Enter Provenance::~Provenance() : 6: 0x9854614 lsst::fw::Provenance
//                     Exit Provenance::~Provenance()
//                     Enter Policy::~Policy() : 5: 0x98547bc lsst::fw::Policy
//                     Exit Policy::~Policy()
//                     Enter Persistence::~Persistence() : 4: 0x98548ec lsst::fw::Persistence
//                     Exit Persistence::~Persistence()
//                     Enter Metadata::~Metadata() : 3: 0x98545bc lsst::fw::Metadata
//                     Exit Metadata::~Metadata()
//  Done destroying MyLsst object
//  Policy object '1: 0x985454c lsst::fw::Policy' going out of scope
//                     Enter Policy::~Policy() : 1: 0x985454c lsst::fw::Policy
//                     Exit Policy::~Policy()
//  No leaks detected
//
 

#include "lsst/fw/fw.h"

#include <iostream>
#include <string>
using namespace std;
using namespace lsst::fw;

class MyLsst : public LsstBase
{
public:
    MyLsst(string s);
    virtual ~MyLsst();
};

MyLsst::MyLsst(string s) : LsstBase(typeid(this)) 
{ 
	Trace( "fw.MyLsst", 10, boost::format("In MyLsst::MyLsst(%s)") % s);
}
MyLsst::~MyLsst()
{
	Trace( "fw.MyLsst", 10, "MyLsst::~MyLsst()");
}

int main( int argc, char** argv )
{
    int verbosity = 100;

    if( argc > 1 )
    {
        try{
            int x = atoi(argv[1]);
            verbosity = x;
        }    
        catch(...)
        {
            verbosity = 0;
        }
    }
    
    Trace::setVerbosity("",verbosity);
    {
        //
        // Create a free Policy object to use for LsstData object 
        //    configuration.
        //
        Trace( "lsstdata01",1, "Explicitly creating a policy object");
        Policy::sharedPtrType sp = SupportFactory::createPolicy();
        Trace( "lsstdata01",1, 
            boost::format("Created policy object '%s'") % sp.get()->toString());
         
        //
        // Create a new instance of an LsstData realization
        // Configure the instance via a given Policy object
        //
        Trace( "lsstdata01",1, "Creating an LsstData realization");
        MyLsst* x = new MyLsst( "An LsstData realization" );
        Trace( "lsstdata01",1, 
            boost::format("Done: Created MyLsst object '%s'") % x->toString());

        Trace( "lsstdata01",1, "Configuring the LsstData object");
        LsstDataConfigurator::configureSupport(x,sp.get());
        Trace( "lsstdata01",1, 
            boost::format("Done: Configured MyLsst object '%s'") % x->toString());

        //
        // Exercise the configured LsstData realization via the default
        // methods on the base object
        //
        
        if( x->getChildren() != 0 ) {
            Trace( 
                "lsstdata01", 1,
                    "Error: MyLsst object is a simple object and does not " \
                    "collect children. getChildren should return a null value");
        }
        else {
            Trace( 
                "lsstdata01", 1,
                    "OK: MyLsst::getChildren returned 0");
        }
            
 
        //
        // Now destroy the configured LsstData realization to trigger
        // all of the base class' destructors and generate trace output 
        //          
        Trace( "lsstdata01",1, 
            boost::format("Destroy MyLsst object '%s'") % x->toString());
        delete x;
        Trace( "lsstdata01",1, "Done destroying MyLsst object");

        Trace( "lsstdata01",1, 
            boost::format("Policy object '%s' going out of scope") \
                % sp.get()->toString());
    }
    //
    // Check for memory leaks
    //
    if (Citizen::census(0) == 0) {
        Trace( "lsstdata01",1, "No leaks detected");
    } else {
        Trace( "lsstdata01",1, "ERROR: Memory leaks detected!");
        Citizen::census(cerr);
    }
    return 1;
}

