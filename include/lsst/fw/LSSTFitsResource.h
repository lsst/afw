#ifndef LSST_LSSTFITSRESOURCE_H
#define LSST_LSSTFITSRESOURCE_H

#include <vw/Image.h>
#include "lsst/fw/DataProperty.h"
#include "lsst/fw/DiskImageResourceFITS.h"

using namespace vw;

namespace lsst {

    namespace fw {
        template <typename PixelT> class LSSTFitsResource : public lsst::fw::DiskImageResourceFITS {
        public:
            LSSTFitsResource();
            void readFits(const std::string& filename, ImageView<PixelT>& image, DataProperty::DataPropertyPtrT metaData, int hdu=0);
            void writeFits(ImageView<PixelT>& image, DataProperty::DataPropertyPtrT metaData, const std::string& filename, int hdu=0);
        private:
            void getMetaData(DataProperty::DataPropertyPtrT metaData);
        };
        
#include "LSSTFitsResource.cc"  

    } // namespace fw

} // namespace lsst

#endif // LSST_LSSTFITSRESOURCE_H
