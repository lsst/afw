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
#include "lsst/afw/image/Wcs.h"
%}

%import "lsst/afw/table/io/Persistable.i"

%declareTablePersistable(Wcs, lsst::afw::image::Wcs);

%ignore lsst::afw::image::NoWcs;

%newobject makeWcs;

%include "lsst/afw/image/Wcs.h"

%lsst_persistable(lsst::afw::image::Wcs);

%inline %{
    #include <boost/make_shared.hpp>
    #include <boost/serialization/serialization.hpp>
    #include <boost/archive/binary_oarchive.hpp>
    #include <boost/archive/binary_iarchive.hpp>
    #include "lsst/daf/persistence/PropertySetFormatter.h"
    #include <sstream>
    std::string pickleMetadata(CONST_PTR(lsst::daf::base::PropertySet) header) {
        std::stringstream ss;
        boost::archive::binary_oarchive ar(ss);
        ar << *header;
        return ss.str();
    }
    PTR(lsst::daf::base::PropertySet) unpickleMetadata(std::string const& pick) {
        std::stringstream ss(pick);
        boost::archive::binary_iarchive ar(ss);
        PTR(lsst::daf::base::PropertySet) header = boost::make_shared<lsst::daf::base::PropertySet>();
        ar >> *header;
        return header;
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

%useValueEquality(lsst::afw::image::Wcs);
