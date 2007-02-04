// -*- LSST-C++ -*-
//! \file
//! \brief Provides support for file formats via libJPEG.

#if !defined(LSST_DISK_IMAGE_RESOUCE_FITS_H)
#define LSST_DISK_IMAGE_RESOUCE_FITS_H 1

#include <string>

#include <vw/Image/PixelTypes.h>
#include <vw/Image/GenericImageBuffer.h>
#include <vw/FileIO/DiskImageResource.h>

namespace lsst { namespace fits {
    class DiskImageResourceFITS : public vw::DiskImageResource {
    public:
        explicit DiskImageResourceFITS();
        DiskImageResourceFITS(std::string const& filename);
        DiskImageResourceFITS(std::string const& filename, 
                              vw::GenericImageFormat const& format);
        
        virtual ~DiskImageResourceFITS();
		
        virtual void read_generic(vw::GenericImageBuffer const& dest) const;
        virtual void write_generic(vw::GenericImageBuffer const& dest);
        virtual void flush();

        void open(std::string const& filename);

        void create(std::string const& filename,
                    vw::GenericImageFormat const& format);

        static DiskImageResource* construct_open(std::string const& filename);

        static DiskImageResource* construct_create(std::string const& filename,
                                                   vw::GenericImageFormat const& format);
        // Accessors
        const int getHdu() const { return _hdu; }
        void setHdu(const int hdu) { _hdu = hdu; }
    private:
        static bool _typeIsRegistered;   //!< Have we registered our file suffixes with VW?
        static int _defaultHdu;          //!< Default HDU to use when opening files
        static void setDefaultHdu(const int hdu);

        std::string _filename;          //!< filename
        void *_fd;                      //<! really a fitsfile, but we aren't including cfitsio.h
        int _hdu;                       //<! desired HDU
        int _ttype;                     //<! cfitsio's name for data type
        int _bitpix;                    //<! FITS' BITPIX keyword
        vw::ChannelTypeEnum _channelType; //<! VW's name for data type
    };

    //! Free function to read a FITS image on disk into a vw::ImageView<T> object.
    //
    //! We need this routine so as to:
    //!   1. be able to read FITS files with arbitrary names
    //!   2. be able to control which HDU we read
    template <typename PixelT>
    void read(vw::ImageView<PixelT>& image, //!< Desired image
              const std::string &filename,  //!< File to read
              const int hdu = 0             //!< Desired HDU; 0 => PDU
             ) {
        // Open the file for reading.  We call DiskImageResourceFITS::construct_open
        // directly to bypass the need to register our file suffixes (although we
        // do that too)
        DiskImageResourceFITS *r =
            dynamic_cast<DiskImageResourceFITS *>(DiskImageResourceFITS::construct_open(filename));
        r->setHdu(false ? hdu : 10);        // XXX
        r->read(image);
    
        delete r;
    }
    
}}
#endif
