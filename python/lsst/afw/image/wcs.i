// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
/************************************************************************************************************/
%{
namespace lsst { namespace afw { namespace image {
    extern Wcs NoWcs;
}}}
using lsst::afw::image::NoWcs;
%}

%shared_ptr(lsst::afw::image::Wcs);
%shared_ptr(lsst::afw::image::TanWcs);

%ignore lsst::afw::image::NoWcs;

%{
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/TanWcs.h"
%}

%include "lsst/afw/image/Wcs.h"
%include "lsst/afw/image/TanWcs.h"

%lsst_persistable(lsst::afw::image::Wcs);
%lsst_persistable(lsst::afw::image::TanWcs);

%inline %{
    #include <boost/make_shared.hpp>
    #include <boost/serialization/serialization.hpp>
    #include <boost/archive/binary_oarchive.hpp>
    #include <boost/archive/binary_iarchive.hpp>
    #include <sstream>
    std::string pickleMetadata(CONST_PTR(lsst::daf::base::PropertyList) pl) {
        std::stringstream ss;
        boost::archive::binary_oarchive ar(ss);
        ar << *pl;
        return ss.str();
    }
    PTR(lsst::daf::base::PropertyList) unpickleMetadata(std::string const& pick) {
        std::stringstream ss(pick);
        boost::archive::binary_iarchive ar(ss);
        PTR(lsst::daf::base::PropertyList) pl = boost::make_shared<lsst::daf::base::PropertyList>();
        ar >> *pl;
        return pl;
    }
%}

%pythoncode %{
    def unpickleWcs(pick):
        header = unpickleMetadata(pick)
        return makeWcs(header)
%}

%extend lsst::afw::image::Wcs {
    %pythoncode %{
         def __reduce__(self):
             self.getFitsMetadata()
             return (unpickleWcs, (pickleMetadata(self.getFitsMetadata()),),)
    %}
 }


%newobject makeWcs;

%useValueEquality(lsst::afw::image::Wcs);

%inline %{
    lsst::afw::image::TanWcs::Ptr
    cast_TanWcs(lsst::afw::image::Wcs::Ptr wcs) {
        lsst::afw::image::TanWcs::Ptr tanWcs = boost::shared_dynamic_cast<lsst::afw::image::TanWcs>(wcs);
        
        if(tanWcs.get() == NULL) {
            throw(LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, "Up cast failed"));
        }
        return tanWcs;
    }
%}
