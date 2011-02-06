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
#include "lsst/afw/image/lsstGil.h"
#include "lsst/afw/image/Utils.h"

#include "lsst/utils/Utils.h"
#include "lsst/pex/exceptions.h"
#include "lsst/daf/base/PropertySet.h"


/************************************************************************************************************/

#include "lsst/pex/exceptions/Exception.h"

namespace pexExcept = lsst::pex::exceptions;

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

    void move_to_hdu(lsst::afw::image::cfitsio::fitsfile *fd, int hdu, bool relative=false);

    void appendKey(lsst::afw::image::cfitsio::fitsfile* fd, std::string const &keyWord,
                   std::string const& keyComment, boost::shared_ptr<const lsst::daf::base::PropertySet> metadata);
    int getNumKeys(fitsfile* fd);
    void getKey(fitsfile* fd, int n, std::string & keyWord, std::string & keyValue, std::string & keyComment);

    void getMetadata(fitsfile* fd, lsst::daf::base::PropertySet::Ptr  metadata, bool strip=true);
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
    virtual ~fits_file_mgr() {}
public:
    FD* get() { return _fd.get(); }
};
    
/************************************************************************************************************/
    
class fits_reader : public fits_file_mgr {
    typedef lsst::daf::base::PropertySet PropertySet;
protected:
    int _hdu;                                            //!< desired HDU
    PropertySet::Ptr _metadata;                          //!< header metadata
    int _naxis1, _naxis2;                                //!< dimension of image
    int _ttype;                                          //!< cfitsio's name for data type
    int _bitpix;                                         //!< FITS' BITPIX keyword
    BBox const& _bbox;                                   //!< Bounding Box of desired part of data

    void init() {
        move_to_hdu(_fd.get(), _hdu, false);

        /* get image data type */
        int bitpix = 0;     // BITPIX from FITS header
        int status = 0;
        if (fits_get_img_equivtype(_fd.get(), &bitpix, &status) != 0) {
            throw LSST_EXCEPT(FitsException, cfitsio::err_msg(_fd.get(), status));
        }
        /*
         * Lookip cfitsio data type
         */
        _ttype = cfitsio::ttypeFromBitpix(bitpix);
    
        /* get image number of dimensions */
        int nAxis = 0;  // number of axes in file
        if (fits_get_img_dim(_fd.get(), &nAxis, &status) != 0) {
            throw LSST_EXCEPT(FitsException,
                              cfitsio::err_msg(_fd.get(), status, boost::format("Getting NAXIS from %s") % _filename));
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
                throw LSST_EXCEPT(FitsException,
                                  cfitsio::err_msg(_fd.get(), 0,
                                                   boost::format("3rd dimension %d of %s is not 1") % nAxes[2] %
                                                   _filename));
            }
        }

        _naxis1 = nAxes[0];
        _naxis2 = nAxes[1];
        
        
        if (_bbox) {
            if (_bbox.getX0() < 0 || _bbox.getY0() < 0 ||
                _bbox.getX1() >= _naxis1 || _bbox.getY1() >= _naxis2
            ) {
                throw LSST_EXCEPT(
                    lsst::pex::exceptions::LengthErrorException,
                    (boost::format("BBox (%d,%d) %dx%d doesn't fit in image %dx%d") %
                    _bbox.getX0() % _bbox.getY0() % _bbox.getWidth() % _bbox.getHeight() %
                    _naxis1 % _naxis2).str()
                ); 
            }
        }
        _bitpix = bitpix;
        //
        // Don't read the rest of the metadata here -- we don't yet know if the view is the right type
        //
    }
    
public:
    fits_reader(cfitsio::fitsfile *file,
                lsst::daf::base::PropertySet::Ptr  metadata,
                int hdu=0, BBox const& bbox=BBox()
    ) : fits_file_mgr(file), _hdu(hdu), _metadata(metadata), _bbox(bbox) { 
        init(); 
    }
    fits_reader(const std::string& filename,
                lsst::daf::base::PropertySet::Ptr  metadata,
                int hdu=0, BBox const& bbox=BBox()
    ) : fits_file_mgr(filename, "rb"), _hdu(hdu), _metadata(metadata), _bbox(bbox) { 
        init(); 
    }

    ~fits_reader() { }
    
    template <typename ImageT>
    void apply(ImageT & image) {
        const int BITPIX = detail::fits_read_support_private<typename ImageT::Pixel>::BITPIX;

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
        int x0 = 0, y0 = 0;                     if (_bbox) {
            x0 = _bbox.getX0(); y0 = _bbox.getY0();
        }
        std::pair<int, int> dimensions = get_Dimensions();
        if (image.getWidth() != dimensions.first || image.getHeight() != dimensions.second) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthErrorException,
                (boost::format("Image dimensions (%d,%d) do not match requested read dimensions %dx%d") %
                image.getWidth() % image.getHeight() % dimensions.first % dimensions.second).str()
            );
        }

        long blc[2] = {x0 + 1, y0 + 1}; // 'bottom left corner' of the subsection
        long trc[2] = {x0 + dimensions.first, y0 + dimensions.second};  // 'top right corner' of the subsection
        long inc[2] = {1, 1};                       // increment to be applied in each dimension (of file)

        int status = 0;                 // cfitsio function return status

        if (fits_read_subset(_fd.get(), _ttype, blc, trc, inc, NULL, &(*image.begin()), NULL, &status) != 0) {
            throw LSST_EXCEPT(FitsException, cfitsio::err_msg(_fd.get(), status));
        }
    }
   
    template <typename ImageT>
    void read_image(ImageT & image) {
        ImageT tmp(get_Dimensions());
        apply(tmp);
        image=tmp;
    }

    std::pair<int,int> get_Dimensions() const {
        if (_bbox) {
            return _bbox.getDimensions();
        } else {
            return std::pair<int, int>(_naxis1, _naxis2);
        }
    }
};
    
class fits_writer : public fits_file_mgr {
    void init() {
        ;
    }
public:
    fits_writer(cfitsio::fitsfile *file) :     fits_file_mgr(file)           { init(); }
    fits_writer(std::string const& filename, std::string const&mode) : fits_file_mgr(filename, mode) { init(); }
    ~fits_writer() { }
    
    template <typename ImageT>
    void apply(
        ImageT const & image,
        boost::shared_ptr<const lsst::daf::base::PropertySet> metadata
    ) {
        const int nAxis = 2;
        long nAxes[nAxis];
        nAxes[0] = image.getWidth();
        nAxes[1] = image.getHeight();
        long imageSize = nAxes[0]*nAxes[1];

        const int BITPIX = detail::fits_read_support_private<typename ImageT::Pixel>::BITPIX;

        int status = 0;
        if (_flags == "pdu") {
            if (fits_create_img(_fd.get(), 8, 0, nAxes, &status) != 0) {
                throw LSST_EXCEPT(FitsException, cfitsio::err_msg(_fd.get(), status));
            }
        } else {
            if (fits_create_img(_fd.get(), BITPIX, nAxis, nAxes, &status) != 0) {
                throw LSST_EXCEPT(FitsException, cfitsio::err_msg(_fd.get(), status));
            }
        }
        /*
         * Write metadata to header.  
         * Ugliness is required to avoid multiple SIMPLE, etc keywords in Fits file,
         * since cfitsio will put in its own in any case.
         */
#if 1
        if (metadata != NULL) {
            typedef std::vector<std::string> NameList;
            NameList paramNames;

            boost::shared_ptr<lsst::daf::base::PropertyList const> pl =
                boost::dynamic_pointer_cast<lsst::daf::base::PropertyList const,
                lsst::daf::base::PropertySet const>(metadata);
            if (pl) {
                paramNames = pl->getOrderedNames();
            } else {
                paramNames = metadata->paramNames(false);
            }
            for (NameList::const_iterator i = paramNames.begin(), e = paramNames.end(); i != e; ++i) {
                if (*i != "SIMPLE" && *i != "BITPIX" &&
                    *i != "NAXIS" && *i != "NAXIS1" && *i != "NAXIS2" && *i != "EXTEND") {
                    cfitsio::appendKey(_fd.get(), *i, "", metadata);
                }
            }
        }
#endif
        if (_flags == "pdu") {            // no data to write
            return;
        }
        
        /*
         * Write the data itself.
         */
        int const ttype = cfitsio::ttypeFromBitpix(BITPIX);
        status = 0;                     // cfitsio function return status

        if (fits_write_img(_fd.get(), ttype, 1, imageSize, &(*image.begin()), &status) != 0) {
            throw LSST_EXCEPT(FitsException, cfitsio::err_msg(_fd.get(), status));
        }
    }
};

} // namespace detail

}}}                             // namespace lsst::afw::image
/// \endcond
#endif
