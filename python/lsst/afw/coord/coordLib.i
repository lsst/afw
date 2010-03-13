// -*- lsst-c++ -*-
%define coordLib_DOCSTRING
"
Python interface to lsst::afw::coord
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.coord", docstring=coordLib_DOCSTRING) coordLib


%include "lsst/p_lsstSwig.i"

%pythoncode %{
import lsst.utils

def version(HeadURL = r"$HeadURL: svn+ssh://svn.lsstcorp.org/DMS/afw/trunk/python/lsst/afw/coord/coordLib.i $"):
    version_svn = lsst.utils.guessSvnVersion(HeadURL)

    try:
	import eups
    except ImportError:
        return version_svn
    else:
	try:
	    version_eups = eups.setup("afw")
	except AttributeError:
	    return version_svn

    if version_eups == version_svn:
        return version_svn
    else:
	return "%s (setup: %s)" % (version_svn, version_eups)
%}

%lsst_exceptions();

%include "std_pair.i"
%template(pairSS) std::pair<std::string, std::string>;

%import "lsst/daf/base/baseLib.i"
%import "lsst/afw/geom/geomLib.i"
%include "observatory.i"
%include "date.i"
%include "coord.i"
