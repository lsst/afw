#ifndef LSST_AFW_IMAGE_LSSTFITSRESOURCE_H
#define LSST_AFW_IMAGE_LSSTFITSRESOURCE_H

#include <vw/Image.h>

#include <lsst/daf/data/DataProperty.h>
#include <lsst/pex/utils/Utils.h>
#include <lsst/pex/exceptions.h>
#include <lsst/afw/image/DiskImageResourceFITS.h>

using namespace vw;
using namespace lsst::daf::data;
using namespace lsst::pex::utils;

namespace lsst {
namespace afw {
namespace image {

    template <typename PixelT> class LSSTFitsResource : public lsst::afw::DiskImageResourceFITS {
    public:
        LSSTFitsResource();
        void readFits( const std::string& filename, ImageView<PixelT>& image, DataProperty::PtrType metaData, int hdu=0);
        void writeFits(ImageView<PixelT>& image, DataProperty::PtrType metaData, const std::string& filename, int hdu=0);
    private:
        void getMetaData(DataProperty::PtrType metaData);
    };

}}} // lsst::afw::image

#ifndef SWIG // don't bother SWIG with .cc files
#include "LSSTFitsResource.cc"  
#endif

#endif // LSST_AFW_IMAGE_LSSTFITSRESOURCE_H
