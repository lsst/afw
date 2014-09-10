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
 
#ifndef LSST_AFW_IMAGE_fits_io_mpl_h_INCLUDED
#define LSST_AFW_IMAGE_fits_io_mpl_h_INCLUDED

#include "boost/mpl/for_each.hpp"
#include "boost/mpl/vector.hpp"

#include "lsst/afw/geom.h"
#include "lsst/afw/fits.h"
#include "lsst/afw/image/fits/fits_io.h"
#include "lsst/afw/image/Image.h"

namespace lsst { namespace afw { namespace image {

namespace {
struct found_type : public std::exception { }; // type to throw when we've read our data

template<typename ImageT, typename ExceptionT>
class try_fits_read_array {
public:
    try_fits_read_array(fits::Fits & fitsfile,
                        ndarray::Array<typename ImageT::Pixel,2,2> & array,
                        geom::Point2I & xy0,
                        daf::base::PropertySet & metadata,
                        geom::Box2I const& bbox,
                        ImageOrigin const origin
    ) : _fitsfile(&fitsfile), _array(array), _xy0(xy0), 
        _metadata(metadata), _bbox(bbox), _origin(origin) { }
    
    // read directly into the desired type if the file's the same type
    void operator()(typename ImageT::Pixel) {
        try {
            fits_read_array(*_fitsfile, _array, _xy0, _metadata, _bbox, _origin);
            throw ExceptionT();         // signal that we've succeeded
        } catch(fits::FitsTypeError const&) {
            // ah well.  We'll try another image type
        }
    }

    template <typename OtherPixel> 
        void operator()(OtherPixel) { // read and convert into the desired type
        try {
            ndarray::Array<OtherPixel,2,2> array;
            fits_read_array(*_fitsfile, array, _xy0, _metadata, _bbox, _origin);
            //copy and convert
            _array = ndarray::allocate(array.getShape());
            _array.deep() = array;
            throw ExceptionT();         // signal that we've succeeded
        } catch(fits::FitsTypeError const&) {
            // pass
        }
    }
private:
    fits::Fits * _fitsfile;
    ndarray::Array<typename ImageT::Pixel,2,2> & _array;
    geom::Point2I & _xy0;
    daf::base::PropertySet & _metadata;
    geom::Box2I const& _bbox;
    ImageOrigin _origin;
};

} // anonymous
            
template<typename supported_fits_types, typename ImageT>
void fits_read_image(
    fits::Fits & fitsfile, ImageT& img,
    lsst::daf::base::PropertySet &metadata,
    geom::Box2I const& bbox=geom::Box2I(),
    ImageOrigin const origin=UNDEFINED
) {
    ndarray::Array<typename ImageT::Pixel,2,2> array;
    geom::Point2I xy0;
    try {
        boost::mpl::for_each<supported_fits_types>(
            try_fits_read_array<ImageT, found_type>(
                fitsfile, array, xy0, metadata, bbox, origin
            )
        );
    } catch (found_type &) {
        img = ImageT(array, false, xy0);
        return;
    }
    throw LSST_FITS_EXCEPT(
        fits::FitsError,
        fitsfile,
        "FITS file does not have one of the expected types"
    );
}

template<typename supported_fits_types, typename ImageT>
void fits_read_image(
    fits::Fits & fitsfile, ImageT& img,
    PTR(lsst::daf::base::PropertySet) metadata = PTR(lsst::daf::base::PropertySet)(),
    geom::Box2I const& bbox=geom::Box2I(),
    ImageOrigin const origin=UNDEFINED
) {
    lsst::daf::base::PropertySet metadata_s;
    fits_read_image<supported_fits_types, ImageT>(fitsfile, img, (metadata ? *metadata : metadata_s),
                                                  bbox, origin);
}

}}} // lsst::afw::image
#endif // !LSST_AFW_IMAGE_fits_io_mpl_h_INCLUDED
