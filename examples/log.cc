// -*- lsst-c++ -*-
/**
  * \file logExample.cc
  *
  * \brief logExample.c demonstrates simple use of the Log facility.
  */

#include "lsst/fw/DataProperty.h"
#include "lsst/fw/Log.h"
#include <fstream>

using namespace std;
using namespace lsst::fw;
using namespace lsst;

static void workLog() {
    std::cout << "\nVerbosity levels:\n";
    Log::printVerbosity(std::cout);

    // Branching Node:  root
    DataPropertyPtrT root(new DataProperty("root"));
    // Terminal Node:  prop1
    DataPropertyPtrT prop1(new DataProperty("prop1", std::string(
"value1")));
    // Terminal Node:  prop2
    DataPropertyPtrT prop2(new DataProperty("prop2", 2));
    // Terminal Node:  root.prop1
    root->addProperty(prop1);
    // Terminal Node:  root.prop2
    root->addProperty(prop2);

    // Branching Node:  branch1
    DataPropertyPtrT branch(new DataProperty("branch1"));
    // Terminal Node:  prop3
    DataPropertyPtrT prop3(new DataProperty("prop3", 3));
    // Terminal Node:  branch1.prop3
    branch->addProperty(prop3);
    // Terminal Node:  root.branch1.prop3
    root->addProperty(branch);

    // Branching Node:  newroot
    DataPropertyPtrT root1(new DataProperty("newroot"));
    // Terminal Node:  prop4
    DataPropertyPtrT prop4(new DataProperty("prop4", 4));
    root1->addProperty(prop4);
    // Terminal Node:  root.prop4

    // Test use of various UI options for logging
    Log("foo", 1, branch);            
    Log("foo.bar",2) << root << root1;

    Log("foo.bar.goo", 4, DataPropertyPtrT(new DataProperty("inlineKeyword",std::string("inlineKvalue"))));

    Log("foo.bar.goo.hoo", 3, DataProperty("CurKeyword",std::string("CurKvalue"))) << branch;
    Log("foo.tar",5) << DataProperty("NewKeyword1",std::string("NewKvalue1")) <<
        DataProperty("NewKeyword2",std::string("NewKvalue2"));
}

/** \brief Test the Log class.
  *
  * \param '-d' indicates that diagnostic messages and logging records should
  *             be emitted to the same output stream.  Otherwise, the logging
  *             records will be emitted to the file name: "MyLog.log".
  */
int main(int argc, char *argv[]) {

    // One input parameter
    std::string intermingleOption("-d");
    int intermingle = 0;        // do NOT intermingle log and diagnostic output

    // The combined text and Log output stream helps in validation.
    if (( argc > 1) && ( strncmp(argv[1], intermingleOption.c_str(), 
                        intermingleOption.length()) == 0 ) ) {
        intermingle = 1;
    }
    
    static string logName = "MyLog.log";
    static std::ofstream myLog;
    if ( intermingle == 0 ) {
        // Use the following block to direct the Log records to a separate file.
        myLog.open(logName.c_str());
        Log::setDestination(myLog);
    }  else {
        Log::setDestination(std::cout);
    }

    Log::setVerbosity(".", 100);
    workLog();

    Log::setVerbosity(".", 0);
    Log::setVerbosity("foo.bar", 3);
    Log::setVerbosity("foo.bar.goo", 10);
    Log::setVerbosity("foo.tar", 5);
    workLog();

    Log::setVerbosity("foo.tar");
    Log::setVerbosity("foo.bar");
    workLog();
    
    std::cout << "\nReset.";
    Log::reset();
    workLog();

    Log::setVerbosity("", 1);
    Log::setVerbosity("foo.bar.goo.hoo", 10);
    workLog();

    Log::setVerbosity("", 2);
    workLog();

    Log::setVerbosity("");
    Log::setVerbosity("foo.bar.goo.hoo");
    Log::setVerbosity("foo.bar.goo.hoo.joo", 10);
    Log::setVerbosity("foo.bar.goo", 3);
    workLog();
    
    return 0;
}
