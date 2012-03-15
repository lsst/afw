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

%boost_picklable(lsst::afw::image::Wcs);
%boost_picklable(lsst::afw::image::TanWcs);

%newobject makeWcs;

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
