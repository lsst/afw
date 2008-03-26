#ifndef LSST_LSSTFITSRESOURCE_H
#define LSST_LSSTFITSRESOURCE_H

#include <vw/Image.h>
#include "lsst/mwi/data/DataProperty.h"
#include "lsst/fw/DiskImageResourceFITS.h"
#include "lsst/mwi/utils/Utils.h"
#include "lsst/mwi/exceptions.h"


using namespace vw;
using namespace lsst::mwi::data;
using namespace lsst::mwi::utils;


namespace lsst {

    namespace fw {

        template <typename PixelT> class LSSTFitsResource : public lsst::fw::DiskImageResourceFITS {
        public:
            LSSTFitsResource();
            void readFits( const std::string& filename, ImageView<PixelT>& image, DataProperty::PtrType metaData, int hdu=0);
            void writeFits(ImageView<PixelT>& image, DataProperty::PtrType metaData, const std::string& filename, int hdu=0);
        private:
            void getMetaData(DataProperty::PtrType metaData);
        };
        
#include "LSSTFitsResource.cc"  

    } // namespace fw

} // namespace lsst

#endif // LSST_LSSTFITSRESOURCE_H
