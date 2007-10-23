//! \file
//! \brief Provides support for the FITS file format.

#include <vector>
#include <boost/scoped_array.hpp>
#include <boost/format.hpp>
#include <vw/Core/Exception.h>
#include <vw/Image/ImageMath.h>
#include "lsst/mwi/exceptions.h"
#include "lsst/fw/DiskImageResourceFITS.h"

// these two necessary only for appendKey()
#include <string.h>
char *strncpy(char *dest, const char *src, size_t n);

extern "C" {
#include "fitsio.h"
}

using namespace lsst::fw;

using lsst::mwi::exceptions::FitsError;
using lsst::mwi::data::DataProperty;

//
// A utility routine to throw an error. Note that the macro form includes
// the line number, which is helpful as the cfitsio errors don't tell me
// where the error was detected
//
#define throw_cfitsio_error(...) _throw_cfitsio_error(__LINE__, __VA_ARGS__)

namespace {
    void _throw_cfitsio_error(const int line,   //!< line in file (from __LINE__)
                              fitsfile *fd,     //!< (possibly invalid) file descriptor
                              const int status, //!< cfitsio error status (0 => no error)
                              const std::string errStr = "" //!< optional extra information
                             ) {
        if (status == 0) {
            if (errStr == "") {
                return;
            }
            throw vw::IOErr() << "DiskImageResourceFITS:L" << line << ": " << errStr;
        } else {
            char fitsErr[FLEN_ERRMSG];
            (void)fits_get_errstatus(status, fitsErr);
            boost::format msg = boost::format("DiskImageResourceFITS: cfitsio error: %d: %s%s%s")
                % line
                % fitsErr
                % (errStr == "" ? "" : " : ")
                % (errStr == "" ? std::string("") : errStr);
            
            switch (status) {
              case FILE_NOT_OPENED:
                throw vw::IOErr() << msg.str();
                break;
              default:
                throw FitsError(msg) << DataProperty("status", status);
                break;
            }
        }
    }

    void _throw_cfitsio_error(const int line, //!< line in file (from __LINE__)
                              fitsfile *fd, //!< (possibly invalid) file descriptor
                              const std::string errStr = "" //!< optional extra information
                             ) {
        _throw_cfitsio_error(line, fd, 0, errStr);
    }

    /******************************************************************************/
    // Multiply a vw::ImageBuffer by a constant
    // Here's the templated function; the driver that looks up
    // the type comes next
    //
    template<typename PIXTYPE>
    void _multiplyImageBuffer(vw::ImageBuffer const& buff, // the buffer in question
                              double value // the value to multiply by
                             ) {
        PIXTYPE *data = static_cast<PIXTYPE *>(buff.data);
        
        for (unsigned int i = 0; i < buff.format.rows*buff.format.cols; i++) {
            *data = static_cast<PIXTYPE>(*data * value);
            data++;
        }
    }

    void multiplyImageBuffer(vw::ImageBuffer const& buff, // the buffer in question
                             double value // the value to multiply by
                            ) {
        
        switch (buff.format.channel_type) {
          case vw::VW_CHANNEL_FLOAT32:
            _multiplyImageBuffer<float>(buff, value);
            break;
          case vw::VW_CHANNEL_FLOAT64:
            _multiplyImageBuffer<double>(buff, value);
            break;
          default:
            throw vw::IOErr() << "MultiplyImageBuffer: unknown type. " << buff.format.channel_type;
        }
    }

    /******************************************************************************/
    //! \brief Move to the specified HDU
    void move_to_hdu(fitsfile *fd,          //!< cfitsio file descriptor
                     int hdu,               //!< desired HDU
                     bool relative = false //!< Is move relative to current HDU?
                    ) {
        int status = 0;			// cfitsio status
        
        if (relative) {
            if (fits_movrel_hdu(fd, hdu, NULL, &status) != 0) {
                throw_cfitsio_error(fd, status, 
                                    str(boost::format("Attempted to select relative HDU %d") % hdu));
            }
        } else {
            if (hdu == 0) { // PDU; go there
                hdu = 1;
            } else {
                if (fits_movabs_hdu(fd, hdu, NULL, &status) != 0) {
                    throw_cfitsio_error(fd, status, 
                                        str(boost::format("Attempted to select absolute HDU %d") % hdu));
                }
            }
        }
    }
}

/******************************************************************************/
//
// Register our suffices so that vw::read_image() will work.
//
bool DiskImageResourceFITS::_typeIsRegistered = false; // We haven't registered our file suffixes with VW
int DiskImageResourceFITS::_defaultHdu = 0;

static DiskImageResourceFITS::DiskImageResourceFITS registerMe; // actually do the registering

/******************************************************************************/
// Constructors

//! Don't open a file, just register our file suffixes for vw::read_image()
DiskImageResourceFITS::DiskImageResourceFITS() : vw::DiskImageResource("") {
    if (!_typeIsRegistered) {
        vw::DiskImageResource::register_file_type(".fit",
                                                  &DiskImageResourceFITS::construct_open,
                                                  &DiskImageResourceFITS::construct_create);
        vw::DiskImageResource::register_file_type(".fits",
                                                  &DiskImageResourceFITS::construct_open,
                                                  &DiskImageResourceFITS::construct_create);

        _typeIsRegistered = true;
    }
    _fd = 0;
}

DiskImageResourceFITS::DiskImageResourceFITS(std::string const& filename //!< file to open
                                            ) : vw::DiskImageResource(filename) {
    _fd = 0;
    open(filename);
}

DiskImageResourceFITS::DiskImageResourceFITS(std::string const& filename, 
                                             vw::ImageFormat const& format
                                            ) : vw::DiskImageResource(filename) {
    _fd = 0;
    create(filename, format);
}

// Destructor
//! Close the FITS file when the object is destroyed
DiskImageResourceFITS::~DiskImageResourceFITS() {
    this->flush();
}
    
//! Flush the buffered data to disk
void DiskImageResourceFITS::flush() {
    if (_fd != 0) {
        int status = 0;
        fitsfile *fd = static_cast<fitsfile *>(_fd);
            
        if (fits_close_file(fd, &status) != 0) {
            throw_cfitsio_error(fd, status);
        }
        _fd = 0;
    }
}

//! Bind the resource to a file for reading.  Confirm that we can open
//! the file and that it has a sane pixel format.  
void DiskImageResourceFITS::open(std::string const& filename //!< Desired filename
                                ) {
    if (_fd) {
        throw vw::IOErr() << "DiskImageResourceFITS: A file is already open.";
    }

    int status = 0;
    if (fits_open_file((fitsfile **)&_fd, filename.c_str(), READONLY, &status) != 0) {
        throw_cfitsio_error(0, status);
    }
    _filename = filename;
    _hdu = _defaultHdu;

    fitsfile *fd = static_cast<fitsfile *>(_fd); // for convenience

    move_to_hdu(fd, _hdu);

    /* get image data type */
    int bitpix = 0;			// BITPIX from FITS header
    if (fits_get_img_equivtype(fd, &bitpix, &status) != 0) {
        throw_cfitsio_error(fd, status);
    }
    /*
     * Find out the image type and convert to cfitsio name/vw channel type
     */
    switch (bitpix) {
      case BYTE_IMG:
	_ttype = TBYTE;
        _channelType = vw::VW_CHANNEL_INT8;
	break;
      case SHORT_IMG:                   // int16
	_ttype = TSHORT;
        _channelType = vw::VW_CHANNEL_INT16;
	break;
      case USHORT_IMG:                  // uint16
	_ttype = TUSHORT;               // n.b. cfitsio does magic things with bzero/bscale to make Uint16
        _channelType = vw::VW_CHANNEL_UINT16;
	break;
      case LONG_IMG:                    // int32
	_ttype = TINT;
        _channelType = vw::VW_CHANNEL_INT32;
	break;
      case FLOAT_IMG:                   // float
	_ttype = TFLOAT;
        _channelType = vw::VW_CHANNEL_FLOAT32;
	break;
      case DOUBLE_IMG:                  // double
	_ttype = TDOUBLE;
        _channelType = vw::VW_CHANNEL_FLOAT64;
	break;
      default:
        throw_cfitsio_error(fd, (boost::format("Unsupported value BITPIX==%d in file \"%s\"") %
                             bitpix % _filename).str());
    }
    
    /* get image number of dimensions */
    int nAxis = 0;			// number of axes in file
    if (fits_get_img_dim(fd, &nAxis, &status) != 0) {
        throw_cfitsio_error(fd, status,
                            str(boost::format("Getting NAXIS from %s") % filename));
    }

    /* validate the number of axes */
    if (nAxis != 0 && (nAxis < 2 || nAxis > 3)) {
        throw_cfitsio_error(fd, (boost::format("Dimensions of '%s' is not supported (NAXIS=%i)") %
                             _filename.c_str() % nAxis).str());
    }
	
    long nAxes[3];		// dimensions of image in file
    if (fits_get_img_size(fd, nAxis, nAxes, &status) != 0) {
        throw_cfitsio_error(fd, status,
                            str(boost::format("Failed to find number of rows in %s") % filename));
    }
    /* if really a 2D image, assume 3rd dimension is 1 */
    if (nAxis == 2) {
        nAxes[2] = 1;
    }
    if (nAxes[2] != 1) {
        throw_cfitsio_error(fd, str(boost::format("3rd dimension %d of %s is not 1") % nAxes[2] % filename));
    }
    
    m_format.rows = nAxes[1];
    m_format.cols = nAxes[0];

    m_format.channel_type = _channelType;
    m_format.pixel_format = vw::VW_PIXEL_SCALAR;
    m_format.planes = 1;
}

//! Bind the resource to a file for writing.
void DiskImageResourceFITS::create(std::string const& filename, //!< file to write
                                   vw::ImageFormat const& format, //!< format. What is this??
                                   DataProperty *metaData //!< metadata to write to header; or NULL
                                  ) {
                                   
    if (format.planes != 1)
        throw vw::NoImplErr() << "We don't support multi-plane images";
    if (format.pixel_format != vw::VW_PIXEL_SCALAR )
        throw vw::NoImplErr() << "We don't support compound pixel types.";
    if (_fd != 0)
        throw vw::IOErr() << "DiskImageResourceFITS: A file is already open.";

    /*
     * Find out channel type and convert to cfitsio name
     */
    _channelType = format.channel_type;

    switch (_channelType) {
      case vw::VW_CHANNEL_UINT8:
	_ttype = TBYTE;
        _bitpix = 8;
	break;
      case vw::VW_CHANNEL_UINT16:
	_ttype = TUSHORT;		 // n.b. cfitsio does magic things with bzero/bscale to make Uint16
        _bitpix = 16;
	break;
      case vw::VW_CHANNEL_INT32:
	_ttype = TINT;
        _bitpix = 32;
	break;
      case vw::VW_CHANNEL_UINT32:
	_ttype = TUINT;
        _bitpix = 32;
	break;
      case vw::VW_CHANNEL_FLOAT32:
	_ttype = TFLOAT;
        _bitpix = -32;
	break;
      case vw::VW_CHANNEL_FLOAT64:
	_ttype = TDOUBLE;
        _bitpix = -64;
	break;
      default:
        throw_cfitsio_error(0, (boost::format("Unsupported channel type==%d in file \"%s\"") %
                                _channelType % _filename).str());
    }
    // Open the file on disk
    (void)unlink(filename.c_str());   // cfitsio doesn't like over-writing files
    
    int status = 0;
    if (fits_create_file((fitsfile **)&_fd, filename.c_str(), &status) != 0) {
        throw_cfitsio_error(static_cast<fitsfile *>(_fd), status);
    }
    
    _filename = filename;
    m_format = format;
    _metaData = metaData;
}

//! Read the disk image into the given buffer.
void DiskImageResourceFITS::read(vw::ImageBuffer const& dest, //!< Where to put the image
                                 vw::BBox2i const& bbox //!< Desired bounding box
                                ) const {
    int status = 0;			// cfitsio function return status

    VW_ASSERT(dest.format.cols == cols() && dest.format.rows == rows(),
              vw::IOErr() << "Buffer has wrong dimensions in FITS read.");

    if (_hdu != 0) {
        throw vw::IOErr() << str(boost::format("Non-default HDUs are not yet supported: %d") % _hdu);
    }

    fitsfile *fd = static_cast<fitsfile *>(_fd); // cfitsio file descriptor
    // Allocate the input buffer, and prepare to read
    const int npix = cols()*rows();     // number of pixels in image
    const int sizeof_pixel = vw::channel_size(_channelType);
    boost::scoped_array<char> buf(new char[npix*sizeof_pixel]); // input buffer
    
    long fpixel[2];			// tell cfitsio which pixels to read
    fpixel[0] = 1;                      // 1 indexed.
    fpixel[1] = 1;                      //            grrrrrr
    int anynull = 0;
    if (fits_read_pix(fd, _ttype, fpixel, npix, NULL, buf.get(), &anynull, &status) != 0) {
        throw_cfitsio_error(fd, status);
    }
    /*
     * Read the fits header.  Details omitted --- see actUtils/src/fitsio.c
     */

    // Set up a generic image buffer around the raw fits data
    vw::ImageBuffer src;
    src.data = buf.get();
    src.format = m_format;
    
    src.cstride = sizeof_pixel;
    src.rstride = src.cstride*cols();
    src.pstride = src.rstride*rows();
    
    convert(dest, src);
    /*
     * Undo delitarious effects of some conversions
     */
    switch (_ttype) {
      case TSHORT:
        multiplyImageBuffer(dest, 65535); // undo the effects of converting short to float
        break;
    }
}

//! Write the given buffer into the disk image.
void DiskImageResourceFITS::write(vw::ImageBuffer const& src, //!< the buffer to write
                                  vw::BBox2i const& bbox) { //!< Desired bounding box
    VW_ASSERT(src.format.cols == cols() && src.format.rows == rows(),
              vw::IOErr() << "Buffer has wrong dimensions in FITS write." );

    fitsfile *fd = static_cast<fitsfile *>(_fd); // cfitsio file descriptor
    
    // Set up the generic image buffer and convert the data into this buffer
    const int npix = cols()*rows();     // number of pixels in image
    const int sizeof_pixel = vw::channel_size(_channelType);

    vw::ImageBuffer dest;
    boost::scoped_array<char> buf(new char[npix*sizeof_pixel]);
    dest.data = buf.get();
    dest.format = m_format;

    dest.cstride = sizeof_pixel;
    dest.rstride = dest.cstride*cols();
    dest.pstride = dest.rstride*rows();
    convert(dest, src);
    /*
     * Undo delitarious effects of some conversions
     */
    switch (_ttype) {
      case TSHORT:
        multiplyImageBuffer(dest, 65535); // undo the effects of converting short to float
        break;
    }
    //
    // OK, so we have the data in the format that they requested.  Now all
    // that we have to do is to write it to a file
    //
    long nAxes[3];			// dimensions of image in file
    int nAxis = 0;			// Image dimension
    nAxes[nAxis++] = cols();
    nAxes[nAxis++] = rows();

    /*  create the mandatory image keywords */
    int status = 0;			// cfitsio status
    if (fits_create_img(fd, _bitpix, nAxis, nAxes, &status) != 0) {
        throw_cfitsio_error(fd, status);
    }
    /*
     * Write metadata to header.  
     * Ugliness is required to avoid multiple SIMPLE, etc keywords in Fits file,
     * since cfitsio will put in its own in any case.
     */
    if (_metaData != NULL) {
        DataProperty::iteratorRangeType range = _metaData->getChildren();
        DataProperty::ContainerIteratorType iter;
        for ( iter = range.first; iter != range.second; iter++) {
            DataProperty::PtrType dpItemPtr = *iter;
            std::string keyName = dpItemPtr->getName();
            if (keyName != "SIMPLE" && keyName != "BITPIX" && 
                keyName != "NAXIS" && keyName != "NAXIS1" && keyName != "NAXIS2" &&
                keyName != "EXTEND") {
		        appendKey(keyName, dpItemPtr->getValue(), "");
	        }
        }
    }
    /*
     * Write the data itself
     */
    if (fits_write_img(fd, _ttype, 1, npix, buf.get(), &status) != 0) {
        throw_cfitsio_error(fd, status);
    }
}

//! A FileIO hook to open a file for reading
vw::DiskImageResource* DiskImageResourceFITS::construct_open(std::string const& filename
                                                            ) {
    return new DiskImageResourceFITS(filename);
}

//! A FileIO hook to open a file for writing
vw::DiskImageResource* DiskImageResourceFITS::construct_create(std::string const& filename,
                                                               vw::ImageFormat const& format) {
    return new DiskImageResourceFITS(filename, format);
}

//! Set the default HDU
void DiskImageResourceFITS::setDefaultHdu(const int hdu //!< desired hdu
                                         ) {
    _defaultHdu = hdu;
}

//! Get the number of keywords in the header
int DiskImageResourceFITS::getNumKeys()
{
     int keynum = 0;
     int numKeys = 0;
     int status = 0;
     fitsfile *fd = static_cast<fitsfile *>(_fd); // cfitsio file descriptor
 
     if (fits_get_hdrpos(fd, &numKeys, &keynum, &status) != 0) {
	  throw_cfitsio_error(fd, status);
     }

     return numKeys;
}

bool DiskImageResourceFITS::getKey(int n, std::string & keyWord, std::string & keyValue, std::string & keyComment)
{
     // NOTE:  the sizes of arrays are tied to FITS standard
     char keyWordChars[80];
     char keyValueChars[80];
     char keyCommentChars[80];

     int status = 0;
     fitsfile *fd = static_cast<fitsfile *>(_fd); // cfitsio file descriptor

     int cfitsioError = fits_read_keyn(fd, n, keyWordChars, keyValueChars, keyCommentChars, &status);
     if (!cfitsioError) {
	  keyWord = keyWordChars;
	  keyValue = keyValueChars;
	  keyComment = keyCommentChars;
	  return true;
     }
     return false;
}

// append a record to the FITS header.   Note the specialization to string values

bool DiskImageResourceFITS::appendKey(const std::string & keyWord, const boost::any & keyValue, const std::string & keyComment)
{

     // NOTE:  the sizes of arrays are tied to FITS standard
     // These shenanigans are required only because fits_write_key does not take const args...
 
     char keyWordChars[80];
     char keyValueChars[80];
     char keyCommentChars[80];

     strncpy(keyWordChars, keyWord.c_str(), 80);
     strncpy(keyCommentChars, keyComment.c_str(), 80);

     int status = 0;
     int cfitsioError = -1;

     fitsfile *fd = static_cast<fitsfile *>(_fd); // cfitsio file descriptor

    if (keyValue.type() == typeid(int)) {
	 int tmp = boost::any_cast<const int>(keyValue);
	 cfitsioError = fits_write_key(fd, TINT, keyWordChars, &tmp, keyCommentChars, &status);

    } else if (keyValue.type() == typeid(double)) {
        double tmp = boost::any_cast<const double>(keyValue);
	cfitsioError = fits_write_key(fd, TDOUBLE, keyWordChars, &tmp, keyCommentChars, &status);

    } else if (keyValue.type() == typeid(std::string)) {
        std::string tmp = boost::any_cast<const std::string>(keyValue);
	strncpy(keyValueChars, tmp.c_str(), 80);
	cfitsioError = fits_write_key(fd, TSTRING, keyWordChars, keyValueChars, keyCommentChars, &status);
    }

     if (!cfitsioError) {
	  return true;
     }
     return false;
}
