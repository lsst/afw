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
    int const grow = 2;
    bool const isotropic = false;
    afwDetection::FootprintSet<PixelT>::Ptr grownFs = afwDetection::makeFootprintSet(*fs, grow, isotropic);
    grownFs->setMask(mi->getMask().get(), "DETECTED");

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
     * Populate cellSet
     */
    for (afwDetection::FootprintSet<PixelT>::FootprintList::iterator ptr = fs->getFootprints().begin(),
             end = fs->getFootprints().end(); ptr != end; ++ptr) {
        afwImage::BBox const bbox = (*ptr)->getBBox();
        float const xc = (bbox.getX0() + bbox.getX1())/2.0;
        float const yc = (bbox.getY0() + bbox.getY1())/2.0;
        TestCandidate::Ptr tc(new TestCandidate(xc, yc, (*im->getImage())(int(xc + 0.5), int(yc + 0.5))));
        cellSet.insertCandidate(tc);
    }
    /*
     * OK, the SpatialCellList is populated.  Let's do something with it
     */
    TestCandidateVisitor visitor;

    cellSet.visitCandidates(&visitor);
    std::cout << boost::format("There are %d candidates\n") % visitor.getN();
    /*
     * Now label the first candidate in each cell as bad
     */
    for (unsigned int i = 0; i != cellSet.getCellList().size(); ++i) {
        afwMath::SpatialCell::Ptr cell = cellSet.getCellList()[i];

        (*cell->begin())->setStatus(afwMath::SpatialCellCandidate::BAD);
        cell->visitCandidates(&visitor);

        cell->setIgnoreBad(false);       // include BAD in cell.size()
        std::cout << boost::format("%s nobj=%d Ngood=%d\n") % cell->getLabel() % cell->size() % visitor.getN();
    }
}

/*
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def TestImageCandidate():
    cellSet = afwMath.SpatialCellSet(afwImage.BBox(afwImage.PointI(0, 0), 501, 501), 2, 3)

    # Test that we can use SpatialCellImageCandidate

    flux = 10
    cellSet.insertCandidate(testSpatialCellLib.TestImageCandidate(0, 0, flux))

    cand = cellSet.getCellList()[0][0]
    #
    # Swig doesn't know that we're a SpatialCellImageCandidate;  all it knows is that we have
    # a SpatialCellCandidate, and SpatialCellCandidates don't know about getImage;  so cast the
    # pointer to SpatialCellImageCandidate<Image<PixelT> > and all will be well;
    #
    # First check that we _can't_ cast to SpatialCellImageCandidate<MaskedImage<PixelT> >
    #
    assert(afwMath.cast_SpatialCellImageCandidateMF(cand), None)

    cand = afwMath.cast_SpatialCellImageCandidateF(cand)

    width, height = 15, 21
    cand.setWidth(width); cand.setHeight(height);

    im = cand.getImage()
    if False and display:
        ds9.mtv(im, title="Candidate", frame=1)
    assert(im.get(0,0), flux) # This is how TestImageCandidate sets its pixels
    assert(im.getWidth(), width)
    assert(im.getHeight(), height)
        
*/

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
/*
 * Run the tests
 */
int main() {
    std::pair<afwImage::MaskedImage<PixelT>::Ptr, afwDetection::FootprintSet<PixelT>::Ptr> data = readImage();
    SpatialCellSetDemo();
}
