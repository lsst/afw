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
struct fits_read_support_private<boost::gil::gray8_view_t> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int,BITPIX=BYTE_IMG); // value is from fitsio.h
};
template <>
struct fits_read_support_private<boost::gil::gray16s_view_t> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int,BITPIX=SHORT_IMG); // value is from fitsio.h
};
template <>
struct fits_read_support_private<boost::gil::gray16_view_t> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int,BITPIX=USHORT_IMG); // value is from fitsio.h
};
template <>
struct fits_read_support_private<boost::gil::gray32s_view_t> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int , BITPIX=LONG_IMG); // value is from fitsio.h
};
template <>
struct fits_read_support_private<boost::gil::gray32_view_t> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int , BITPIX=ULONG_IMG); // value is from fitsio.h
};
template <>
struct fits_read_support_private<boost::gil::gray32f_noscale_view_t> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    BOOST_STATIC_CONSTANT(int,BITPIX=FLOAT_IMG); // value is from fitsio.h
};
template <>
struct fits_read_support_private<boost::gil::gray64f_noscale_view_t> {
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
struct fits_write_support_private<boost::gil::gray8_view_t> {
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
	
    fits_file_mgr(char **ramFile, size_t *ramFileLen, const std::string& flags) :
		_fd(static_cast<FD *>(NULL)), _flags(flags) {
		if (flags == "r" || flags == "rb") {
			int status = 0;
			if (fits_open_memfile(&_fd_s, "UnusedFilenameParameter", READONLY, (void**)ramFile,
								  ramFileLen, 0,
								  NULL/*Memory allocator unnecessary for READONLY*/, &status) != 0) {
				throw LSST_EXCEPT(FitsException, cfitsio::err_msg("fits_open_memfile", status));
			}
		} else if (flags == "w" || flags == "wb" || flags == "pdu") {
			int status = 0;
			//If ramFile is NULL, we will allocate it here.
			//Otherwise we will assume that ramFileLen is correct for ramFile.
			if (ramFile == NULL)
			{
				*ramFileLen = 2880;	//Initial buffer size (file length)
				*ramFile = new char[*ramFileLen];
			}
			size_t deltaSize = 0;	//0 is a flag that this parameter will be ignored and the default 2880 used instead
			if (fits_create_memfile(&_fd_s, (void**)ramFile,
									ramFileLen, deltaSize, &realloc, &status) != 0) {
				throw LSST_EXCEPT(FitsException, cfitsio::err_msg("fits_create_memfile", status));
			}
        } else if (flags == "a" || flags == "ab") {
            int status = 0;
			size_t deltaSize = 0;	//0 is a flag that this parameter will be ignored and the default 2880 used instead
            if (fits_open_memfile(&_fd_s, "UnusedFilenameParameter", READWRITE, (void**)ramFile,
									  ramFileLen, deltaSize,
									  &realloc, &status) != 0) {
				throw LSST_EXCEPT(FitsException, cfitsio::err_msg("fits_open_memfile", status));
            }
            /*
             * Seek to end of the file
             */
            int nHdu = 0;
            if (fits_get_num_hdus(_fd_s, &nHdu, &status) != 0 ||
                fits_movabs_hdu(_fd_s, nHdu, NULL, &status) != 0) {
                (void)cfitsio::fits_close_file(_fd_s, &status);
				throw LSST_EXCEPT(FitsException, cfitsio::err_msg("fits_close_file", status));
            }
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
         * Lookup cfitsio data type
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
        _bitpix = bitpix;
        //
        // Don't read the rest of the metadata here -- we don't yet know if the view is the right type
        //
    }
    
public:
    fits_reader(cfitsio::fitsfile *file,
                lsst::daf::base::PropertySet::Ptr  metadata,
                int hdu=0, BBox const& bbox=BBox()) :
        fits_file_mgr(file), _hdu(hdu), _metadata(metadata), _bbox(bbox) { init(); }
    fits_reader(const std::string& filename,
                lsst::daf::base::PropertySet::Ptr  metadata,
                int hdu=0, BBox const& bbox=BBox()) :
		fits_file_mgr(filename, "rb"), _hdu(hdu), _metadata(metadata), _bbox(bbox) { init(); }
    fits_reader(char **ramFile, size_t *ramFileLen,
                lsst::daf::base::PropertySet::Ptr  metadata,
                int hdu=0, BBox const& bbox=BBox()) :
		fits_file_mgr(ramFile, ramFileLen, "rb"), _hdu(hdu), _metadata(metadata), _bbox(bbox) { init(); }

    ~fits_reader() { }

    template <typename View>
    void apply(View& view) {
        const int BITPIX = detail::fits_read_support_private<View>::BITPIX;
        if (BITPIX != _bitpix) {
            const std::string msg = (boost::format("Incorrect value of BITPIX; saw %d expected %d") % _bitpix % BITPIX).str();
            throw LSST_EXCEPT(FitsWrongTypeException, msg);
        }

        /*
         * Read metadata
         */
        cfitsio::getMetadata(_fd.get(), _metadata);

        int x0 = 0, y0 = 0;             // Origin of part of image to read
        if (_bbox) {
            x0 = _bbox.getX0(); y0 = _bbox.getY0();

            if (x0 + view.width() > _naxis1 || y0 + view.height() > _naxis2) {
                throw LSST_EXCEPT(pexExcept::LengthErrorException,
                                  (boost::format("BBox (%d,%d) -- (%d,%d) doesn't fit in image of size %dx%d")
                                   % x0 % y0 % _bbox.getX1() % _bbox.getY1() % _naxis1 % _naxis2).str());
                
            }
        }
        /*
         * Cfitsio 3.006 is able to read some, but not all, data types from top to bottom;  floats are OK,
         * but unsigned short isn't.
         *
         * When cfitsio cooperates it saves us from having to flip the rows ourselves
         */
        long blc[2] = {x0, y0 + view.height() - 1}; // 'bottom left corner' of the subsection
        long trc[2] = {x0 + view.width() - 1, y0};  // 'top right corner' of the subsection
        long inc[2] = {1, 1};                       // increment to be applied in each dimension (of file)

        blc[0]++; blc[1]++;             // 1-indexed.
        trc[0]++; trc[1]++;             //            Grrrrrrrr

        int status = 0;                 // cfitsio function return status
#if 0                                   // this generates slower code (more seeks) than the read-and-swap
        if (fits_read_subset(_fd.get(), _ttype, blc, trc, inc, NULL, view.row_begin(0), NULL, &status) == 0) {
            return;                     // The simple case; the read succeeded
        }
        
        if (status != BAD_PIX_NUM) {
            throw LSST_EXCEPT(FitsException, cfitsio::err_msg(_fd.get(), status));
        }
        /*
         * cfitsio returned a BAD_PIX_NUM errror, which (usually?) means that this type can't be read
         * in the desired order;  so we'll do it ourselves --- i.e. do the read and flip the rows
         */
#endif
        std::swap(blc[1], trc[1]);

        status = 0;
        if (fits_read_subset(_fd.get(), _ttype, blc, trc, inc, NULL, view.row_begin(0), NULL, &status) != 0) {
            throw LSST_EXCEPT(FitsException, cfitsio::err_msg(_fd.get(), status));
        }
        // Here's the row flip
        std::vector<typename View::value_type> tmp(view.width());
        for (int y = 0; y != view.height()/2; ++y) {
            int const yp = view.height() - y - 1;
            std::copy(view.row_begin(y),  view.row_end(y),  tmp.begin());
            std::copy(view.row_begin(yp), view.row_end(yp), view.row_begin(y));
            std::copy(tmp.begin(),        tmp.end(),        view.row_begin(yp));
        }
    }
    
    template <typename Image>
    void read_image(Image& im) {
        im.recreate(get_Dimensions());
        apply(view(im));
    }

    boost::gil::point2<std::ptrdiff_t> get_Dimensions() const {
        if (_bbox) {
            return boost::gil::point2<std::ptrdiff_t>(_bbox.getWidth(), _bbox.getHeight());
        } else {
            return boost::gil::point2<std::ptrdiff_t>(_naxis1, _naxis2);
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
    fits_writer(char **ramFile, size_t *ramFileLen, std::string const&mode) :
		fits_file_mgr(ramFile, ramFileLen, mode) { init(); }
    ~fits_writer() { }
    
    template <typename View>
    void apply(const View& view,
               boost::shared_ptr<const lsst::daf::base::PropertySet> metadata
              ) {
        const int nAxis = 2;
        long nAxes[nAxis];
        nAxes[0] = view.width();
        nAxes[1] = view.height();

        const int BITPIX = detail::fits_read_support_private<View>::BITPIX;

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
         * Write the data itself.  Our underlying boost::gil image has the lowest-address row at the top so we
         * have to flip rows to write it correctly even if the image is contiguous (which it may not be if
         * it's a subimage)
         *
         * An alternative is write it row-by-row
         */
        int const ttype = cfitsio::ttypeFromBitpix(BITPIX);
        status = 0;                     // cfitsio function return status
#if 1                                   // Write in one go via a copy
        std::vector<typename View::value_type> tmp(view.size());
        typename std::vector<typename View::value_type>::iterator tptr = tmp.begin();
        for (int y = 0; y != view.height(); ++y, tptr += view.width()) {
            std::copy(view.row_begin(y), view.row_end(y), tptr);
        }
		
        if (fits_write_img(_fd.get(), ttype, 1, tmp.size(), &tmp[0], &status) != 0) {
            throw LSST_EXCEPT(FitsException, cfitsio::err_msg(_fd.get(), status));
        }
#else
        /*
         * Write row-by-row; less efficient as cfitsio isn't very smart, but economical on memory
         */
        for (int y = 0; y != view.height(); ++y) {
            if (fits_write_img(_fd.get(), ttype, 1 + y*view.width(), view.width(),
                               view.row_begin(y), &status) != 0) {
                throw LSST_EXCEPT(FitsException,
                                  cfitsio::err_msg(_fd.get(), status, boost::format("Writing row %d") % y));
            }
        }
#endif
    }
};

} // namespace detail

}}}                             // namespace lsst::afw::image
/// \endcond
#endif
