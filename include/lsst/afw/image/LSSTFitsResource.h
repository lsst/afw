#ifndef LSST_AFW_IMAGE_LSSTFITSRESOURCE_H
#define LSST_AFW_IMAGE_LSSTFITSRESOURCE_H

#include "vw/Image.h"

#include "lsst/daf/base/DataProperty.h"
#include "lsst/afw/image/DiskImageResourceFITS.h"

namespace lsst {
namespace afw {
namespace image {

    template <typename PixelT> class LSSTFitsResource : public lsst::afw::image::DiskImageResourceFITS {
    public:
        LSSTFitsResource();
        void readFits(
            const std::string& filename,
            vw::ImageView<PixelT>& image,
            lsst::daf::base::DataProperty::PtrType metaData,
            int hdu=0
        );
        void writeFits(
            vw::ImageView<PixelT>& image,
            lsst::daf::base::DataProperty::PtrType metaData,
            const std::string& filename,
            int hdu=0
        );
    private:
        void getMetaData(
            lsst::daf::base::DataProperty::PtrType metaData
        );
    };

}}} // lsst::afw::image

#ifndef SWIG // don't bother SWIG with .cc files
#include "LSSTFitsResource.cc"  
#endif

#endif // LSST_AFW_IMAGE_LSSTFITSRESOURCE_H
