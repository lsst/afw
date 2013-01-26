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
 
%{
#include "lsst/afw/image/ImagePca.h"
%}

//
// Must go After the %include
//
%define %declareImagePca(TYPE, PIXEL_TYPE...)
%template(vectorImage##TYPE) std::vector<boost::shared_ptr<lsst::afw::image::Image<PIXEL_TYPE> > >;
%template(ImagePca##TYPE) lsst::afw::image::ImagePca<lsst::afw::image::Image<PIXEL_TYPE> >;
%template(ImagePcaM##TYPE) lsst::afw::image::ImagePca<lsst::afw::image::MaskedImage<PIXEL_TYPE> >;
%template(innerProduct) lsst::afw::image::innerProduct<lsst::afw::image::Image<PIXEL_TYPE>,
                                                       lsst::afw::image::Image<PIXEL_TYPE>  >;
%enddef

%include "lsst/afw/image/ImagePca.h"

%declareImagePca(U, boost::uint16_t);
%declareImagePca(L, boost::uint64_t);
%declareImagePca(I, int);
%declareImagePca(F, float);
%declareImagePca(D, double);

