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

#include "lsst/utils/Utils.h"
#include "lsst/pex/exceptions.h"


namespace lsst { namespace afw { namespace image {

namespace cfitsio {
#if !defined(DOXYGEN)
    extern "C" {
#       include "fitsio.h"
    }
#endif

    void _throw_cfitsio_error(const char *file, const int line,   //!< line in file (from __FILE__, __LINE__)
                              lsst::afw::image::cfitsio::fitsfile *fd, //!< (possibly invalid) file descriptor
                              const int status = 0, //!< cfitsio error status (0 => no error)
                              const std::string errStr = ""//!< optional extra information
                             );
    void _throw_cfitsio_error(const char *file, const int line, //!< line in file (from __FILE__/__LINE__)
                              lsst::afw::image::cfitsio::fitsfile *fd, //!< (possibly invalid) file descriptor
                              const std::string errStr = "" //!< optional extra information
                             );
    //
    // A utility routine to throw an error. Note that the macro form includes
    // the line number, which is helpful as the cfitsio errors don't tell me
    // where the error was detected
    //
#   define throw_cfitsio_error(...) cfitsio::_throw_cfitsio_error(__FILE__, __LINE__, __VA_ARGS__)

    /************************************************************************************************************/
        
    int ttypeFromBitpix(const int bitpix);

    void move_to_hdu(lsst::afw::image::cfitsio::fitsfile *fd, int hdu, bool relative = false);

    void appendKey(lsst::afw::image::cfitsio::fitsfile* fd,
                   const std::string & keyWord, const boost::any & keyValue, const std::string & keyComment);
    int getNumKeys(fitsfile* fd);
    void getKey(fitsfile* fd, int n, std::string & keyWord, std::string & keyValue, std::string & keyComment);

    void getMetadata(fitsfile* fd, lsst::daf::base::DataProperty::PtrType metadata);    
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
    typedef lsst::afw::image::detail::types_traits<unsigned short>::view_t view_t;
};
template <>
struct cfitsio_traits<32> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    typedef lsst::afw::image::detail::types_traits<int>::view_t view_t;
};
template <>
struct cfitsio_traits<-32> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    typedef lsst::afw::image::detail::types_traits<float>::view_t view_t;
};
template <>
struct cfitsio_traits<-64> {
    BOOST_STATIC_CONSTANT(bool,is_supported=true);
    typedef lsst::afw::image::detail::types_traits<double>::view_t view_t;
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
    std::string _filename;                               //!< filename
    
    struct null_deleter { void operator()(void const*) const {} };
    //
    // A functor to pass as the second (cleanup) argument to boost::shared_ptr<>
    // to close our fitsio handle
    //
    struct close_cfitsio {
        void operator()(FD* fd) const {
            if (fd != NULL) {
                int status = 0;
                if (lsst::afw::image::cfitsio::fits_close_file(fd, &status) != 0) {
                    char fitsErr[FLEN_ERRMSG];
                    (void)lsst::afw::image::cfitsio::fits_get_errstatus(status, fitsErr);

                    fprintf(stderr, "Problems closing %s: %s (%d)\n", "filename", fitsErr, status);
                }
            }
        }
    };

    fits_file_mgr(FD* file) : _fd(file, null_deleter()) {}

    fits_file_mgr(const std::string& filename, const std::string& flags) :
        _fd(static_cast<FD *>(NULL)), _filename(filename) {
        if (flags == "r" || flags == "rb") {
            int status = 0;
            if (fits_open_file(&_fd_s, filename.c_str(), READONLY, &status) != 0) {
                throw_cfitsio_error(0, status);
            }
        } else if (flags == "w" || flags == "wb") {
            int status = 0;
            (void)unlink(filename.c_str()); // cfitsio doesn't like over-writing files
            if (fits_create_file(&_fd_s, filename.c_str(), &status) != 0) {
                throw_cfitsio_error(0, status);
            }
        } else {
            abort();
        }

        _fd = boost::shared_ptr<FD>(_fd_s, close_cfitsio());
    }
    
public:
    FD* get() { return _fd.get(); }
};
    
/************************************************************************************************************/
    
class fits_reader : public fits_file_mgr {
    typedef lsst::daf::base::DataProperty DataProperty;
protected:
    int _hdu;                                            //!< desired HDU
    DataProperty::PtrType _metadata;                     //!< header metadata
    int _naxis1, _naxis2;                                //!< dimension of image
    int _ttype;                                          //!< cfitsio's name for data type
    int _bitpix;                                         //!< FITS' BITPIX keyword

    void init() {
        move_to_hdu(_fd.get(), _hdu);

        /* get image data type */
        int bitpix = 0;     // BITPIX from FITS header
        int status = 0;
        if (fits_get_img_equivtype(_fd.get(), &bitpix, &status) != 0) {
            throw_cfitsio_error(_fd.get(), status);
        }
        /*
         * Lookip cfitsio data type
         */
        _ttype = cfitsio::ttypeFromBitpix(bitpix);
    
        /* get image number of dimensions */
        int nAxis = 0;  // number of axes in file
        if (fits_get_img_dim(_fd.get(), &nAxis, &status) != 0) {
            throw_cfitsio_error(_fd.get(), status, (boost::format("Getting NAXIS from %s") % _filename).str());
        }

        /* validate the number of axes */
        if (nAxis != 0 && (nAxis < 2 || nAxis > 3)) {
            throw_cfitsio_error(_fd.get(), (boost::format("Dimensions of '%s' is not supported (NAXIS=%i)") %
                                            _filename % nAxis).str());
        }
        
        long nAxes[3];  // dimensions of image in file
        if (fits_get_img_size(_fd.get(), nAxis, nAxes, &status) != 0) {
            throw_cfitsio_error(_fd.get(), status, (boost::format("Failed to find number of rows in %s") % _filename).str());
        }
        /* if really a 2D image, assume 3rd dimension is 1 */
        if (nAxis == 2) {
            nAxes[2] = 1;
        }
        if (nAxes[2] != 1) {
            throw_cfitsio_error(_fd.get(),
                                (boost::format("3rd dimension %d of %s is not 1") % nAxes[2] % _filename).str());
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
#if 1                                   // Old name for boost::shared_ptrs
                lsst::daf::base::DataProperty::PtrType metadata, // = typename lsst::daf::base::DataProperty::PtrType(static_cast<lsst::daf::base::DataProperty *>(0)),
#else
                lsst::daf::base::DataProperty::ConstPtr metadata, // = typename lsst::daf::base::DataProperty::ConstPtr(static_cast<lsst::daf::base::DataProperty *>(0)),
#endif
                int hdu=0) :
        fits_file_mgr(file), _hdu(hdu), _metadata(metadata) { init(); }
    fits_reader(const std::string& filename,
#if 1                                   // Old name for boost::shared_ptrs
                lsst::daf::base::DataProperty::PtrType metadata,
#else
                lsst::daf::base::DataProperty::ConstPtr metadata,
#endif
                int hdu=0) :
        fits_file_mgr(filename, "rb"), _hdu(hdu), _metadata(metadata) { init(); }

    ~fits_reader() { }

    template <typename View>
    void apply(View& view) {
        if (_hdu != 0) {
            throw lsst::pex::exceptions::FitsError(boost::format("Non-default HDUs are not yet supported: %d") % _hdu);
        }
    
        const int BITPIX = detail::fits_read_support_private<View>::BITPIX;
        if (BITPIX != _bitpix) {
            const std::string msg = str(boost::format("Incorrect value of BITPIX; saw %d expected %d") % _bitpix % BITPIX);
#if 1
            throw lsst::pex::exceptions::FitsError(msg);
#else
            std::cerr << msg << std::endl;
#endif
        }

        /*
         * Read metadata
         */
        cfitsio::getMetadata(_fd.get(), _metadata);

        for (int y = 0; y != view.height(); ++y) {
            long fpixel[2];                     // tell cfitsio which pixels to read
            fpixel[0] = 1;                      // 1 indexed.
            fpixel[1] = 1 + y;                  //            grrrrrr
            int anynull = 0;
            int status = 0;                     // cfitsio function return status

            if (fits_read_pix(_fd.get(), _ttype, fpixel, view.width(), NULL,
                              view.row_begin(view.height() - y - 1), &anynull, &status) != 0) {
                throw_cfitsio_error(_fd.get(), status, str(boost::format("Reading row %d") % y));
            }
        }
    }
    
    template <typename Image>
    void read_image(Image& im) {
        im.recreate(get_getDimensions());
        apply(view(im));
    }

    boost::gil::point2<std::ptrdiff_t> get_getDimensions() const {
        return boost::gil::point2<std::ptrdiff_t>(_naxis1, _naxis2);
    }
};
    
class fits_writer : public fits_file_mgr {
    void init() {
        ;
    }
public:
    fits_writer(cfitsio::fitsfile *file) :     fits_file_mgr(file)           { init(); }
    fits_writer(std::string const& filename) : fits_file_mgr(filename, "wb") { init(); }
    ~fits_writer() { }
    
    template <typename View>
    void apply(const View& view,
#if 1                                   // Old name for boost::shared_ptrs
               lsst::daf::base::DataProperty::PtrType metadata
#else
               lsst::daf::base::DataProperty::ConstPtr metadata
#endif
              ) {
        const int nAxis = 2;
        long nAxes[nAxis];
        nAxes[0] = view.width();
        nAxes[1] = view.height();

        const int BITPIX = detail::fits_read_support_private<View>::BITPIX;

        int status = 0;
        if (fits_create_img(_fd.get(), BITPIX, nAxis, nAxes, &status) != 0) {
            throw_cfitsio_error(_fd.get(), status);
        }
        /*
         * Write metadata to header.  
         * Ugliness is required to avoid multiple SIMPLE, etc keywords in Fits file,
         * since cfitsio will put in its own in any case.
         */
#if 1
        if (metadata != NULL) {
            using lsst::daf::base::DataProperty;
            
            DataProperty::iteratorRangeType range = metadata->getChildren();
            DataProperty::ContainerIteratorType iter;
            for (iter = range.first; iter != range.second; ++iter) {
                DataProperty::PtrType dpItemPtr = *iter;
                std::string keyName = dpItemPtr->getName();
                if (keyName != "SIMPLE" && keyName != "BITPIX" && 
                    keyName != "NAXIS" && keyName != "NAXIS1" && keyName != "NAXIS2" &&
                    keyName != "EXTEND") {
                    cfitsio::appendKey(_fd.get(), keyName, dpItemPtr->getValue(), "");
                }
            }
        }
#endif
        /*
         * Write the data itself
         */
        const int ttype = cfitsio::ttypeFromBitpix(BITPIX);
        for (int y = 0; y != view.height(); ++y) {
            int status = 0;                     // cfitsio function return status
            if (fits_write_img(_fd.get(), ttype, 1 + y*view.width(), view.width(), view.row_begin(y), &status) != 0) {
                throw_cfitsio_error(_fd.get(), status, str(boost::format("Writing row %d") % y));
            }
        }
    }
};

} // namespace detail

}}}                             // namespace lsst::afw::image
/// \endcond
#endif
