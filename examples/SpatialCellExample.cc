/*
 * This C++ example does the same thing as SpatialCellExample.py.  The latter of the python version
 * is that you can set display == True and see what's going on
 */
#include <string>
#include "lsst/utils/Utils.h"
#include "lsst/pex/exceptions.h"
#include "lsst/daf/base.h"
#include "lsst/afw/detection.h"
#include "lsst/afw/image.h"
#include "lsst/afw/math.h"

#include "testSpatialCell.h"

namespace afwDetection = lsst::afw::detection;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

typedef float PixelT;

/*
 * Read an image and background subtract it
 */
std::pair<afwImage::MaskedImage<PixelT>::Ptr, afwDetection::FootprintSet<PixelT>::Ptr>
readImage() {
    afwImage::MaskedImage<PixelT>::Ptr mi;

    try {
        std::string dataDir = lsst::utils::eups::productDir("afwdata");

        std::string filename = dataDir + "/CFHT/D4/cal-53535-i-797722_1";
        
        afwImage::BBox bbox = afwImage::BBox(afwImage::PointI(270, 2530), 512, 512);
        
        lsst::daf::base::PropertySet::Ptr md;
        mi.reset(new afwImage::MaskedImage<PixelT>(filename, 0, md, bbox));
        mi->setXY0(afwImage::PointI(0, 0));
    } catch (lsst::pex::exceptions::NotFoundException &e) {
        std::cerr << e << std::endl;
        exit(1);
    }
    /*
     * Subtract the background.  We can't fix those pesky cosmic rays, as that's in a dependent product
     * (meas/algorithms)
     */
    afwMath::BackgroundControl bctrl(afwMath::Interpolate::NATURAL_SPLINE);
    bctrl.setNxSample(mi->getWidth()/256 + 1);
    bctrl.setNySample(mi->getHeight()/256 + 1);
    bctrl.sctrl.setNumSigmaClip(3.0);
    bctrl.sctrl.setNumIter(2);

    afwImage::Image<PixelT>::Ptr im = mi->getImage();
    try {
        *mi->getImage() -= *afwMath::makeBackground(*im, bctrl).getImage<PixelT>();
    } catch(std::exception &e) {
        bctrl.setInterpStyle(afwMath::Interpolate::CONSTANT);
        *mi->getImage() -= *afwMath::makeBackground(*im, bctrl).getImage<PixelT>();
    }
    /*
     * Find sources
     */
    afwDetection::Threshold threshold(5, afwDetection::Threshold::STDEV);
    int npixMin = 5;                    // we didn't smooth
    afwDetection::FootprintSet<PixelT>::Ptr fs =
        afwDetection::makeFootprintSet(*mi, threshold, "DETECTED", npixMin);
    int const grow = 1;
    bool const isotropic = false;
    afwDetection::FootprintSet<PixelT>::Ptr grownFs = afwDetection::makeFootprintSet(*fs, grow, isotropic);
    grownFs->setMask(mi->getMask(), "DETECTED");

    return std::make_pair(mi, grownFs);
}

/************************************************************************************************************/
/*
 * A demonstration of the use of a SpatialCellSet
 */
void SpatialCellSetDemo() {
    afwImage::MaskedImage<PixelT>::Ptr im;
    afwDetection::FootprintSet<PixelT>::Ptr fs;
    boost::tie(im, fs) = readImage();
    /*
     * Create an (empty) SpatialCellSet
     */
    afwMath::SpatialCellSet cellSet(afwImage::BBox(afwImage::PointI(0, 0), im->getWidth(), im->getHeight()),
                                    260, 200);
    /*
     * Populate the cellSet using the detected object in the FootprintSet
     */
    for (afwDetection::FootprintSet<PixelT>::FootprintList::iterator ptr = fs->getFootprints().begin(),
             end = fs->getFootprints().end(); ptr != end; ++ptr) {
        afwImage::BBox const bbox = (*ptr)->getBBox();
        float const xc = (bbox.getX0() + bbox.getX1())/2.0;
        float const yc = (bbox.getY0() + bbox.getY1())/2.0;
        ExampleCandidate::Ptr tc(new ExampleCandidate(xc, yc, im->getImage(), bbox));
        cellSet.insertCandidate(tc);
    }
    /*
     * OK, the SpatialCellList is populated.  Let's do something with it
     */
    ExampleCandidateVisitor visitor;

    cellSet.visitCandidates(&visitor);
    std::cout << boost::format("There are %d candidates\n") % visitor.getN();
    /*
     * Now label too-small object as BAD
     */
    for (unsigned int i = 0; i != cellSet.getCellList().size(); ++i) {
        afwMath::SpatialCell::Ptr cell = cellSet.getCellList()[i];

        int j = 0;
        for (afwMath::SpatialCell::iterator candidate = cell->begin(), candidateEnd = cell->end();
             candidate != candidateEnd; ++candidate, ++j) {
            int w, h;
            boost::tie(w, h) =
                dynamic_cast<ExampleCandidate *>((*candidate).get())->getBBox().getDimensions();
            
#if 0
            std::cout << boost::format("%d %5.2f %5.2f %d\n")
                % i % (*candidate)->getXCenter() % (*candidate)->getYCenter() % (w*h);
#endif
            if (w*h < 75) {
                (*candidate)->setStatus(afwMath::SpatialCellCandidate::BAD);
            }
        }
    }
    /*
     * Now count the good and bad candidates
     */        
    for (unsigned int i = 0; i != cellSet.getCellList().size(); ++i) {
        afwMath::SpatialCell::Ptr cell = cellSet.getCellList()[i];
        cell->visitCandidates(&visitor);

        cell->setIgnoreBad(false);       // include BAD in cell.size()
        std::cout << boost::format("%s nobj=%d N_good=%d NPix_good=%d\n") %
            cell->getLabel() % cell->size() % visitor.getN() % visitor.getNPix();
    }
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
/*
 * Run the example
 */
int main() {
    std::pair<afwImage::MaskedImage<PixelT>::Ptr, afwDetection::FootprintSet<PixelT>::Ptr> data = readImage();

    SpatialCellSetDemo();
}
