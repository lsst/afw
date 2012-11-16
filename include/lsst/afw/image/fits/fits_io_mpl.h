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
 
/**
 * \file
 * \brief  Internal support for reading and writing FITS files
 *
 * Tell doxygen to (usually) ignore this file \cond GIL_IMAGE_INTERNALS
 * \author Robert Lupton (rhl@astro.princeton.edu)
 *         Princeton University
 * \date   September 2008
 */
#if !defined(LSST_FITS_IO_MPL_H)
#define LSST_FITS_IO_MPL_H 1

#include <exception>
#include "boost/mpl/for_each.hpp"
#include "boost/mpl/vector.hpp"

#include "boost/gil/gil_all.hpp"

#include "lsst/afw/geom.h"

#include "lsst/afw/image/lsstGil.h"
#include "fits_io.h"

namespace {
struct found_type : public std::exception { }; // type to throw when we've read our data

template<typename ImageT, typename ExceptionT>
class try_fits_read_image {
public:
    try_fits_read_image(const std::string& file,
                        ndarray::Array<typename ImageT::Pixel,2,2> & array,
                        lsst::afw::geom::Point2I & xy0,
                        lsst::daf::base::PropertySet & metadata,
                        int hdu,
                        lsst::afw::geom::Box2I const& bbox,
                        lsst::afw::image::ImageOrigin const origin
    ) : _file(file), _array(array), _xy0(xy0), 
        _metadata(metadata), _hdu(hdu), _bbox(bbox), _origin(origin) { }
    
    // read directly into the desired type if the file's the same type
    void operator()(typename ImageT::Pixel) {
        try {
            lsst::afw::image::fits_read_image(_file, _array, _xy0, _metadata, _hdu, _bbox, _origin);
            throw ExceptionT();         // signal that we've succeeded
        } catch(lsst::afw::image::FitsWrongTypeException const&) {
            // ah well.  We'll try another image type
        }
    }

    template <typename OtherPixel> 
        void operator()(OtherPixel) { // read and convert into the desired type
        try {
            ndarray::Array<OtherPixel,2,2> array;
            lsst::afw::image::fits_read_image(_file, array, _xy0, _metadata, _hdu, _bbox, _origin);
            //copy and convert
            _array = ndarray::allocate(array.getShape());
            _array.deep() = array;
            throw ExceptionT();         // signal that we've succeeded
        } catch(lsst::afw::image::FitsWrongTypeException const&) {
            // pass
        }
    }
private:
    std::string _file;
    ndarray::Array<typename ImageT::Pixel,2,2> & _array;
    lsst::afw::geom::Point2I & _xy0;
    lsst::daf::base::PropertySet & _metadata;
    int _hdu;
    lsst::afw::geom::Box2I const& _bbox;
    lsst::afw::image::ImageOrigin _origin;
};

}

namespace lsst { namespace afw { namespace image {
            
template<typename supported_fits_types, typename ImageT>
bool fits_read_image(
    std::string const& file, ImageT& img,
    lsst::daf::base::PropertySet &metadata,
    int hdu=0,
    geom::Box2I const& bbox = geom::Box2I(),
    ImageOrigin const origin = LOCAL
)
{
    ndarray::Array<typename ImageT::Pixel,2,2> array;
    geom::Point2I xy0;
    try {
        boost::mpl::for_each<supported_fits_types>(
            try_fits_read_image<ImageT, found_type>(
                file, array, xy0, metadata, hdu, bbox, origin
            )
        );
    } catch (found_type &) {
        img = ImageT(array, false, xy0);
        return true;                    // success
    }

    return false;
}


template<typename supported_fits_types, typename ImageT>
bool fits_read_image(
    std::string const& file, ImageT& img,
    PTR(lsst::daf::base::PropertySet) metadata = PTR(lsst::daf::base::PropertySet)(),
    int hdu=0,
    geom::Box2I const& bbox = geom::Box2I(),
    ImageOrigin const origin = LOCAL
)
{
    lsst::daf::base::PropertySet metadata_s;
    return fits_read_image<supported_fits_types, ImageT>(file, img, (metadata ? *metadata : metadata_s),
                                                         hdu, bbox, origin);
}

}}}                                     // lsst::afw::image
/// \endcond
#endif
