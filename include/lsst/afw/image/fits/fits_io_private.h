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
 
/**
 * \file
 * \brief  Internal support for reading and writing FITS files
 *
 * Tell doxygen to (usually) ignore this file \cond GIL_IMAGE_INTERNALS
 * \author Robert Lupton (rhl@astro.princeton.edu)
 *         Princeton University
 * \date   September 2008
 */
#if !defined(LSST_FITS_IO_PRIVATE_H)
#define LSST_FITS_IO_PRIVATE_H

#include <iostream>
#include <unistd.h>
#include "boost/static_assert.hpp"
#include "boost/format.hpp"

#include "boost/gil/gil_all.hpp"
#include "boost/gil/extension/io/io_error.hpp"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/lsstGil.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/fits.h"

#include "lsst/utils/Utils.h"
#include "lsst/pex/exceptions.h"
#include "lsst/daf/base/PropertySet.h"


/************************************************************************************************************/

#include "lsst/pex/exceptions/Exception.h"

/************************************************************************************************************/

namespace lsst { namespace afw { namespace image {

/**
 * @brief An exception thrown when problems are found connected with FITS files
 */
LSST_EXCEPTION_TYPE(FitsException,
                    lsst::pex::exceptions::Exception, lsst::pex::exceptions::LogicErrorException)
/// An exception thrown when a FITS file is the wrong type
LSST_EXCEPTION_TYPE(FitsWrongTypeException,
                    lsst::pex::exceptions::Exception, lsst::pex::exceptions::InvalidParameterException)

namespace cfitsio {
#if !defined(DOXYGEN)
    extern "C" {
#       include "fitsio2.h"
    }
#endif

    std::string err_msg(fitsfile const *fd, int const status = 0, std::string const &errMsg = "");
    std::string err_msg(std::string const &fileName, int const status = 0, std::string const &errMsg = "");
    inline std::string err_msg(fitsfile const *fd, int const status, boost::format const &fmt) { return err_msg(fd, status, fmt.str()); }

    /************************************************************************************************************/
        
    int ttypeFromBitpix(const int bitpix);

    void move_to_hdu(lsst::afw::image::cfitsio::fitsfile *fd, int hdu, bool relative=false,
                     bool headerOnly=false);

    void appendKey(lsst::afw::image::cfitsio::fitsfile* fd, std::string const &keyWord,
                   std::string const& keyComment, boost::shared_ptr<const lsst::daf::base::PropertySet> metadata);

    int getNumKeys(fitsfile* fd);
    void getKey(fitsfile* fd, int n, std::string & keyWord, std::string & keyValue, std::string & keyComment);

    void getMetadata(fitsfile* fd, lsst::daf::base::PropertySet & metadata, bool strip=true);
}

namespace detail {
//
// Traits types to tell us about supported Fits types
//
template <typename Channel>
struct fits_read_support_private {
    BOOST_STATIC_CONSTANT(bool,is_supported=false);
    BOOST_STATIC_CONSTANT(int , BITPIX=0);
};
//
// A partial specialization to handle const
//
template <typename T>
struct fits_read_support_private<const T> {
    BOOST_STATIC_CONSTANT(bool,is_supported = fits_read_support_private<T>::is_supported);
    BOOST_STATIC_CONSTANT(int, BITPIX = fits_read_support_private<T>::BITPIX);
};
//
// Here are our types
//
template <>
struct fits_read_support_private<unsigned char> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int,BITPIX=BYTE_IMG); // value is from fitsio.h
};
template <>
struct fits_read_support_private<short> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int,BITPIX=SHORT_IMG); // value is from fitsio.h
};
template <>
struct fits_read_support_private<unsigned short> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int,BITPIX=USHORT_IMG); // value is from fitsio.h
};
template <>
struct fits_read_support_private<int> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int , BITPIX=LONG_IMG); // value is from fitsio.h
};
template <>
struct fits_read_support_private<unsigned int> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int , BITPIX=ULONG_IMG); // value is from fitsio.h
};
template <>
struct fits_read_support_private<boost::int64_t> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int , BITPIX=LONGLONG_IMG); // value is from fitsio.h
};
template <>
struct fits_read_support_private<boost::uint64_t> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int , BITPIX=LONGLONG_IMG); // value is from fitsio.h
};
template <>
struct fits_read_support_private<float> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int,BITPIX=FLOAT_IMG); // value is from fitsio.h
};
template <>
struct fits_read_support_private<double> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int , BITPIX=DOUBLE_IMG); // value is from fitsio.h
};


/************************************************************************************************************/

template <typename Channel>
struct fits_write_support_private {
    BOOST_STATIC_CONSTANT(bool,is_supported=false);
};
//
// A partial specialization to handle const
//
template <typename T>
struct fits_write_support_private<const T> {
    BOOST_STATIC_CONSTANT(bool,is_supported = fits_write_support_private<T>::is_supported);
};

template <>
struct fits_write_support_private<unsigned char> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
};
    
/************************************************************************************************************/
// map FITS types to our extended gil ones
template <int bitpix>
struct cfitsio_traits {
    BOOST_STATIC_CONSTANT(bool,is_supported=false);
};
template <>
struct cfitsio_traits<16> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    typedef types_traits<unsigned short>::view_t view_t;
};
template <>
struct cfitsio_traits<32> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    typedef types_traits<int>::view_t view_t;
};
template <>
struct cfitsio_traits<64> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    typedef types_traits<boost::int64_t>::view_t view_t;
};
template <>
struct cfitsio_traits<-32> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    typedef types_traits<float>::view_t view_t;
};
template <>
struct cfitsio_traits<-64> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    typedef types_traits<double>::view_t view_t;
};
//
// Like gil's file_mgr class (from whence cometh this code), but knows about
// cfitsio
//
class fits_file_mgr {
    typedef cfitsio::fitsfile FD;

    FD *_fd_s;                          // storage for _fd; we're not going to delete it in _fd's dtor so this is OK
protected:
    boost::shared_ptr<FD> _fd;
    std::string _filename;                               ///< filename
    std::string _flags;                                  ///< flags used to open file
    
    struct null_deleter { void operator()(void const*) const {} };
    //
    // A functor to pass as the second (cleanup) argument to boost::shared_ptr<>
    // to close our fitsio handle
    //
    struct close_cfitsio {
        void operator()(FD* fd) const {
            if (fd != NULL) {
                int status = 0;
                if (cfitsio::fits_close_file(fd, &status) != 0) {
                    std::cerr << cfitsio::err_msg(fd, status) << std::endl;
                }
            }
        }
    };

    fits_file_mgr(FD* file) : _fd(file, null_deleter()) {}

    fits_file_mgr(const std::string& filename, const std::string& flags) :
        _fd(static_cast<FD *>(NULL)), _filename(filename), _flags(flags) {
        if (flags == "r" || flags == "rb") {
            int status = 0;
            if (fits_open_file(&_fd_s, filename.c_str(), READONLY, &status) != 0) {
                throw LSST_EXCEPT(FitsException, cfitsio::err_msg(filename, status));
            }
        } else if (flags == "w" || flags == "wb" || flags == "pdu") {
            int status = 0;
            (void)unlink(filename.c_str()); // cfitsio doesn't like over-writing files
            if (fits_create_file(&_fd_s, filename.c_str(), &status) != 0) {
                throw LSST_EXCEPT(FitsException, cfitsio::err_msg(filename, status));
            }
        } else if (flags == "a" || flags == "ab") {
            int status = 0;
            if (fits_open_file(&_fd_s, filename.c_str(), READWRITE, &status) != 0) {
                throw LSST_EXCEPT(FitsException, cfitsio::err_msg(filename, status));
            }
            /*
             * Seek to end of the file
             */
            int nHdu = 0;
            if (fits_get_num_hdus(_fd_s, &nHdu, &status) != 0 ||
                fits_movabs_hdu(_fd_s, nHdu, NULL, &status) != 0) {
                (void)cfitsio::fits_close_file(_fd_s, &status);
                throw LSST_EXCEPT(FitsException, cfitsio::err_msg(filename, status));
            }
        } else {
            throw LSST_EXCEPT(FitsException, "Unknown mode " + flags);
        }

        _fd = boost::shared_ptr<FD>(_fd_s, close_cfitsio());
    }

    fits_file_mgr(char **ramFile, size_t *ramFileLen, const std::string& flags) :
        _fd(static_cast<FD *>(NULL)), _flags(flags) {
        if (flags == "r" || flags == "rb") {
            int status = 0;
            if (fits_open_memfile(&_fd_s, "UnusedFilenameParameter", READONLY, (void**)ramFile,
                    ramFileLen, 0, NULL/*Memory allocator unnecessary for READONLY*/, &status) != 0) {
                throw LSST_EXCEPT(FitsException, cfitsio::err_msg("fits_open_memfile", status));
            }
        } else if (flags == "w" || flags == "wb" || flags == "pdu") {
           int status = 0;
            //If ramFile is NULL, we will allocate it here.
            //Otherwise we will assume that ramFileLen is correct for ramFile.
            if (ramFile == NULL)
            {
                *ramFileLen = 2880;    //Initial buffer size (file length)
                *ramFile = new char[*ramFileLen];
            }
            size_t deltaSize = 0;    //0 is a flag that this parameter will be ignored and the default 2880 used instead
            if (fits_create_memfile(&_fd_s, (void**)ramFile,
                                    ramFileLen, deltaSize, &realloc, &status) != 0) {
                throw LSST_EXCEPT(FitsException, cfitsio::err_msg("fits_create_memfile", status));
            }
        } else if (flags == "a" || flags == "ab") {
            int status = 0;
            size_t deltaSize = 0;    //0 is a flag that this parameter will be ignored and the default 2880 used instead
            if (fits_open_memfile(&_fd_s, "UnusedFilenameParameter", READWRITE, (void**)ramFile,
                    ramFileLen, deltaSize, &realloc, &status) != 0) {
                throw LSST_EXCEPT(FitsException, cfitsio::err_msg("fits_open_memfile", status));
            }
            //Seek to end of the file
            int nHdu = 0;
            if (fits_get_num_hdus(_fd_s, &nHdu, &status) != 0 ||
                fits_movabs_hdu(_fd_s, nHdu, NULL, &status) != 0) {
                (void)cfitsio::fits_close_file(_fd_s, &status);
                throw LSST_EXCEPT(FitsException, cfitsio::err_msg("fits_close_file", status));
            }
        } else {
            throw LSST_EXCEPT(FitsException, "Unknown mode " + flags);
        }

        _fd = boost::shared_ptr<FD>(_fd_s, close_cfitsio());
    }
    virtual ~fits_file_mgr() {}
public:
    FD* get() { return _fd.get(); }
};
    
/************************************************************************************************************/
    
class fits_reader : public fits_file_mgr {
    typedef lsst::daf::base::PropertySet PropertySet;
protected:
    int _hdu;                                            //!< desired HDU
    PropertySet & _metadata;                             //!< header metadata
    int _naxis1, _naxis2;                                //!< dimension of image    
    int _ttype;                                          //!< cfitsio's name for data type
    int _bitpix;                                         //!< FITS' BITPIX keyword
    geom::Box2I _bbox;                             //!< Bounding Box of desired part of data
    ImageOrigin _origin;

    void init(bool headerOnly=false) {
        if (_hdu == 0) {
            // User asked for the default 'automatic' header positioning.
            // Make sure they haven't already moved the header cursor by
            // specifying an extension name when opening the file.
            int hduNum = -1;
            fits_get_hdu_num(_fd.get(), &hduNum);
            if (hduNum != 1) {
                _hdu = hduNum;
            }
        }

        move_to_hdu(_fd.get(), _hdu, false, headerOnly);

        /* get image data type */
        int bitpix = 0;     // BITPIX from FITS header
        int status = 0;
        if (fits_get_img_equivtype(_fd.get(), &bitpix, &status) != 0) {
            throw LSST_EXCEPT(
                FitsException, 
                cfitsio::err_msg(_fd.get(), status)
            );
        }
        /*
         * Lookip cfitsio data type
         */
        _ttype = cfitsio::ttypeFromBitpix(bitpix);
    
        /* get image number of dimensions */
        int nAxis = 0;  // number of axes in file
        if (fits_get_img_dim(_fd.get(), &nAxis, &status) != 0) {
            throw LSST_EXCEPT(
                FitsException,
                cfitsio::err_msg(
                    _fd.get(), 
                    status, 
                    boost::format("Getting NAXIS from %s") % _filename
                )
            );
        }

        /* validate the number of axes */
        long nAxes[3];  // dimensions of image in file

        if (nAxis == 0) {
            nAxes[0] = nAxes[1] = 0;
        } else {
            if (nAxis < 2 || nAxis > 3) {
                throw LSST_EXCEPT(FitsException,
                                  cfitsio::err_msg(_fd.get(), 0,
                                                   boost::format("Dimensions of '%s' is not supported (NAXIS=%i)") %
                                                   _filename % nAxis));
            }
        
            if (fits_get_img_size(_fd.get(), nAxis, nAxes, &status) != 0) {
                throw LSST_EXCEPT(FitsException,
                                  cfitsio::err_msg(_fd.get(), status,
                                                   boost::format("Failed to find number of rows in %s") % _filename));
            }
            /* if really a 2D image, assume 3rd dimension is 1 */
            if (nAxis == 2) {
                nAxes[2] = 1;
            }
            if (nAxes[2] != 1) {
                throw LSST_EXCEPT(
                    FitsException,
                    cfitsio::err_msg(
                        _fd.get(), 0,
                        boost::format("3rd dimension %d of %s is not 1") % 
                        nAxes[2] % _filename
                    )
                );
            }
        }

        _naxis1 = nAxes[0];
        _naxis2 = nAxes[1];
        

        _bitpix = bitpix;
        //
        // Don't read the rest of the metadata here -- we don't yet know if the view is the right type
        //
    }
    
public:
    fits_reader(cfitsio::fitsfile *file,
                lsst::daf::base::PropertySet & metadata,
                int hdu=0, geom::Box2I const& bbox=geom::Box2I(), 
                ImageOrigin const origin = LOCAL
    ) : fits_file_mgr(file), _hdu(hdu), _metadata(metadata), _bbox(bbox), _origin(origin) {         
        init(); 
    }

    fits_reader(const std::string& filename,
                lsst::daf::base::PropertySet & metadata,
                int hdu=0, geom::Box2I const& bbox=geom::Box2I(),
                ImageOrigin const origin = LOCAL
    ) : fits_file_mgr(filename, "rb"), _hdu(hdu), _metadata(metadata), _bbox(bbox), _origin(origin) { 
        init(); 
    }

    fits_reader(char **ramFile, size_t *ramFileLen,
                lsst::daf::base::PropertySet & metadata,
                int hdu=0, geom::Box2I const& bbox=geom::Box2I(),
                ImageOrigin const origin = LOCAL
    ) : fits_file_mgr(ramFile, ramFileLen, "rb"), _hdu(hdu), _metadata(metadata), _bbox(bbox), _origin(origin) { 
        init(); 
    }

    fits_reader(const std::string& filename,
                lsst::daf::base::PropertySet & metadata,
                int hdu, bool headerOnly
               ) : fits_file_mgr(filename, "rb"), _hdu(hdu), _metadata(metadata),
                   _bbox(geom::Box2I()), _origin(LOCAL) { 
        init(headerOnly); 
    }

    ~fits_reader() { }
    
    template <typename PixelT>
    geom::Point2I apply(ndarray::Array<PixelT,2,2> const & array) {
        const int BITPIX = detail::fits_read_support_private<PixelT>::BITPIX;

        if (BITPIX != _bitpix) {            
            throw LSST_EXCEPT(
                FitsWrongTypeException, 
                (boost::format("Incorrect value of BITPIX; saw %d expected %d") % _bitpix % BITPIX).str()
            );
        }

        /*
         * Read metadata
         */
        cfitsio::getMetadata(_fd.get(), _metadata);

        // Origin of part of image to read
        geom::Point2I xy0(0,0);

        geom::Extent2I xyOffset(getImageXY0FromMetadata(wcsNameForXY0, &_metadata));
        if (!_bbox.isEmpty()) {
            if(_origin == PARENT) {
                _bbox.shift(-xyOffset);
            }
            
            xy0 = _bbox.getMin();

            if (_bbox.getMinX() < 0 || _bbox.getMinY() < 0 ||
                _bbox.getWidth() > _naxis1 || _bbox.getHeight() > _naxis2
            ) {
                throw LSST_EXCEPT(
                    lsst::pex::exceptions::LengthErrorException,
                    (boost::format("BBox (%d,%d) %dx%d doesn't fit in image %dx%d") %
                    _bbox.getMinX() % _bbox.getMinY() % _bbox.getWidth() % _bbox.getHeight() %
                    _naxis1 % _naxis2).str()
                ); 
            } 
        }
        geom::Extent2I dimensions = getDimensions();
        if (array.template getSize<1>() != dimensions.getX() 
            || array.template getSize<0>() != dimensions.getY()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthErrorException,
                (boost::format("Image dimensions (%d,%d) do not match requested read dimensions %dx%d") %
                 array.template getSize<1>() % array.template getSize<0>() %
                 dimensions.getX() % dimensions.getY()).str()
            );
        }
        // 'bottom left corner' of the subsection (1-indexed)
        long blc[2] = {xy0.getX() + 1, xy0.getY() + 1};
        // 'top right corner' of the subsection
        long trc[2] = {xy0.getX() + dimensions.getX(), xy0.getY() + dimensions.getY()}; 
        // increment to be applied in each dimension (of file)
        long inc[2] = {1, 1};                       

        int status = 0;                 // cfitsio function return status

        if (fits_read_subset(_fd.get(), _ttype, blc, trc, inc, NULL, array.getData(), NULL, &status) != 0) {
            throw LSST_EXCEPT(FitsException, cfitsio::err_msg(_fd.get(), status));
        }

        return xy0 + xyOffset;
    }
   
    template <typename PixelT>
    void read_image(ndarray::Array<PixelT,2,2> & array, geom::Point2I & xy0) {
        array = ndarray::allocate(getDimensions().getY(), getDimensions().getX());
        xy0 = apply(array);        
    }

    geom::Extent2I getDimensions() const {
        if (_bbox.isEmpty()) {
            return geom::Extent2I(_naxis1, _naxis2);
        } else {
            return _bbox.getDimensions();
        }    
    }
};

} // namespace detail

}}}                             // namespace lsst::afw::image
/// \endcond
#endif
