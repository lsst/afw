// -*- lsst-c++ -*-
/*
 * This would be a very simple module, except that I didn't want to
 * deal with (char **) in SWIG; so I wrote C wrappers to pass a single
 * (char *) instead.
 */
%define xpa_DOCSTRING
"""
Simple interface to the xpa routines used to communicate with ds9
"""
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.display", docstring=xpa_DOCSTRING) xpa

%ignore XPAGet;
%ignore XPASet;
%ignore XPASetFd;

%rename(get) XPAGet1;
%rename(set) XPASet1;
%rename(setFd1) XPASetFd1;

%{
#include "boost/noncopyable.hpp"
#include "xpa.h"
#include "lsst/pex/exceptions/Runtime.h"

namespace {
    class myXPA : boost::noncopyable {
    public:
        static XPA get(bool reset=false) {
            static myXPA *singleton = NULL;

            if (reset && singleton != NULL) {
                delete singleton;
                singleton = NULL;
            }

            if (singleton == NULL) {
                singleton = new myXPA("w");
            }

            return singleton->_xpa;
        }
    private:
        myXPA(char const *mode) {
            _xpa = XPAOpen((char *)mode);

            if (_xpa == NULL) {
                throw LSST_EXCEPT(lsst::pex::exceptions::IoErrorException, "Unable to open XPA");
            }
        }
        
        ~myXPA() {
            XPAClose(_xpa);
        }

        static XPA _xpa;                // the real XPA connection
    };

    XPA myXPA::_xpa = NULL;
}

/*
 * A binding for XPAGet that talks to only one server, but doesn't have to talk (char **) with SWIG
 */
const char *
XPAGet1(XPA xpa,
	char *xtemplate,
	char *paramlist,
	char *mode)
{
    char *buf = NULL;			/* desired response */
    int len = 0;			/* length of buf; ignored */
    char *error = NULL;			/* returned error if any*/

    if (xpa == NULL) {
        xpa = myXPA::get();
    }

    int n = XPAGet(xpa, xtemplate, paramlist, mode,
		   &buf, &len, NULL, &error, 1);

    if(n == 0) {
	return(NULL);
    }
    if(error != NULL) {
	return(error);
    }

    return(buf);
}

/*****************************************************************************/

const char *
XPASet1(XPA xpa,
	char *xtemplate,
	char *paramlist,
	char *mode,
	char *buf,			// desired extra data
	int len)			// length of buf (or -1)
{
    if(len < 0) {
	len = strlen(buf);		// length of buf
    }
    char *error = NULL;			// returned error if any

    if (xpa == NULL) {
        xpa = myXPA::get();
    }

    int n = XPASet(xpa, xtemplate, paramlist, mode,
		   buf, len, NULL, &error, 1);

    if(n == 0) {
	return(NULL);
    }
    if(error != NULL) {
	return(error);
    }

    return "";
}


/*****************************************************************************/

const char *
XPASetFd1(XPA xpa,
	  char *xtemplate,
	  char *paramlist,
	  char *mode,
	  int fd)			/* file descriptor for xpa to read */
{
    char *error = NULL;			/* returned error if any*/

    if (xpa == NULL) {
        xpa = myXPA::get();
    }

    int n = XPASetFd(xpa, xtemplate, paramlist, mode,
		     fd, NULL, &error, 1);

    if(n == 0) {
	return(NULL);
    }
    if(error != NULL) {
	return(error);
    }

    return NULL;
}
%}

%rename(XPA_in) in;			// avoid conflict with python keyword in xpa.h

%import "prsetup.h"
%import "xpa.h"

%inline %{
    void reset() {
        myXPA::get(true);
    }
%}

%include "exception.i"

%exception {
    $action
    if (result == NULL) {
       SWIG_exception(SWIG_IOError, "XPA returned NULL");
    }
}

const char *XPAGet1(XPA xpa, char *xtemplate, char *paramlist, char *mode);
const char *XPASet1(XPA xpa, char *xtemplate, char *paramlist, char *mode, char *buf, int len);	
const char *XPASetFd1(XPA xpa, char *xtemplate, char *paramlist, char *mode, int fd);
