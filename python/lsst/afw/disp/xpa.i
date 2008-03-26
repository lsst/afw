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
%module(docstring=xpa_DOCSTRING) xpa

%ignore XPAGet;
%ignore XPASet;
%ignore XPASetFd;

%rename(get) XPAGet1;
%rename(set) XPASet1;
%rename(setFd1) XPASetFd1;

%{
#include "xpa.h"

static char *
xmalloc(long n)
{
    char *ptr = (char *)malloc(n);
    assert(ptr != NULL);

    return(ptr);
}

/*
 * A binding for XPAGet that talks to only one server, but doesn't have to talk (char **) with SWIG
 */
char *
XPAGet1(XPA xpa,
	char *xtemplate,
	char *paramlist,
	char *mode)
{
    char *buf = NULL;			/* desired response */
    int len = 0;			/* length of buf; ignored */
    char *error = NULL;			/* returned error if any*/

    int n = XPAGet(xpa, xtemplate, paramlist, mode,
		   &buf, &len, NULL, &error, 1);

    if(n == 0) {
	return(NULL);
    }
    if(error != NULL) {
	char *errStr = xmalloc(strlen(error) + 1);
	strcpy(errStr, error);
	return(errStr);
    }

    return(buf);
}

/*****************************************************************************/

char *
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

    int n = XPASet(xpa, xtemplate, paramlist, mode,
		   buf, len, NULL, &error, 1);

    if(n == 0) {
	return(NULL);
    }
    if(error != NULL) {
	char *errStr = xmalloc(strlen(error) + 1);
	strcpy(errStr, error);
	return(errStr);
    }

    return "";
}


/*****************************************************************************/

char *
XPASetFd1(XPA xpa,
	  char *xtemplate,
	  char *paramlist,
	  char *mode,
	  int fd)			/* file descriptor for xpa to read */
{
    char *error = NULL;			/* returned error if any*/

    int n = XPASetFd(xpa, xtemplate, paramlist, mode,
		     fd, NULL, &error, 1);

    if(n == 0) {
	return(NULL);
    }
    if(error != NULL) {
	char *errStr = xmalloc(strlen(error) + 1);
	strcpy(errStr, error);

	return(errStr);
    }

    return NULL;
}
%}

%rename(XPA_in) in;			// avoid conflict with python keyword in xpa.h

%import "prsetup.h"
%import "xpa.h"

%include "exception.i"

%exception {
    $action
    if (result == NULL) {
       SWIG_exception(SWIG_IOError, "XPA returned NULL");
    }
}

char *XPAGet1(XPA xpa, char *xtemplate, char *paramlist, char *mode);
char *XPASet1(XPA xpa, char *xtemplate, char *paramlist, char *mode, char *buf, int len);	
char *XPASetFd1(XPA xpa, char *xtemplate, char *paramlist, char *mode, int fd);
