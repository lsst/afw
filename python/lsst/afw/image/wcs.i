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

%import "lsst/afw/table/io/ioLib.i"

%declareTablePersistable(Wcs, lsst::afw::image::Wcs);
%declareTablePersistable(TanWcs, lsst::afw::image::TanWcs);

%ignore lsst::afw::image::NoWcs;

%{
#include <memory>
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/TanWcs.h"
#include "lsst/afw/image/DistortedTanWcs.h"
%}

%shared_ptr(lsst::afw::image::XYTransformFromWcsPair);
%shared_ptr(lsst::afw::image::DistortedTanWcs);

%include "lsst/afw/image/Wcs.h"
%include "lsst/afw/image/TanWcs.h"
%include "lsst/afw/image/DistortedTanWcs.h"

%lsst_persistable(lsst::afw::image::Wcs);
%lsst_persistable(lsst::afw::image::TanWcs);

%pythoncode %{
    def unpickleWcs(pick):
        import pickle
        exposure = pickle.loads(pick)
        return exposure.getWcs()
%}

%extend lsst::afw::image::Wcs {
    %pythoncode %{
         def __reduce__(self):
             import pickle
             exposure = ExposureU(1, 1)
             exposure.setWcs(self)
             return (unpickleWcs, (pickle.dumps(exposure),))
    %}
 }


%newobject makeWcs;

%useValueEquality(lsst::afw::image::Wcs);

// ----------------------------------------------------------
// THIS CASE API IS DEPRECATED IN FAVOR OF %castShared
%inline %{
    lsst::afw::image::TanWcs::Ptr
    cast_TanWcs(lsst::afw::image::Wcs::Ptr wcs) {
        lsst::afw::image::TanWcs::Ptr tanWcs = std::dynamic_pointer_cast<lsst::afw::image::TanWcs>(wcs);

        if(tanWcs.get() == NULL) {
            throw(LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Up cast failed"));
        }
        return tanWcs;
    }
%}
// ----------------------------------------------------------

%castShared(lsst::afw::image::TanWcs, lsst::afw::image::Wcs)
%castShared(lsst::afw::image::DistortedTanWcs, lsst::afw::image::Wcs)
