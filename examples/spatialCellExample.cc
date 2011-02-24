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
#include "lsst/afw/geom.h"


#include "testSpatialCell.h"

namespace afwDetect = lsst::afw::detection;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace afwGeom = lsst::afw::geom;
typedef float PixelT;

std::pair<afwImage::MaskedImage<PixelT>::Ptr, afwDetect::FootprintSet<PixelT>::Ptr> readImage();

/************************************************************************************************************/
/*
 * A demonstration of the use of a SpatialCellSet
 */
void SpatialCellSetDemo() {
    afwImage::MaskedImage<PixelT>::Ptr im;
    afwDetect::FootprintSet<PixelT>::Ptr fs;
    boost::tie(im, fs) = readImage();
    /*
     * Create an (empty) SpatialCellSet
     */
    afwMath::SpatialCellSet cellSet(
        im->getBBox(afwImage::LOCAL),
        260, 200
    );
    /*
     * Populate the cellSet using the detected object in the FootprintSet
     */
    for (afwDetect::FootprintSet<PixelT>::FootprintList::iterator ptr = fs->getFootprints().begin(),
             end = fs->getFootprints().end(); ptr != end; ++ptr) {
        afwGeom::BoxI const bbox = (*ptr)->getBBox();
        float const xc = (bbox.getMinX() + bbox.getMaxX())/2.0;
        float const yc = (bbox.getMinY() + bbox.getMaxY())/2.0;
        ExampleCandidate::Ptr tc(
            new ExampleCandidate(xc, yc, 
                im->getImage(), 
                bbox
            )
        );
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

        for (afwMath::SpatialCell::iterator candidate = cell->begin(), candidateEnd = cell->end();
             candidate != candidateEnd; ++candidate) {
            afwGeom::BoxI box =
                dynamic_cast<ExampleCandidate *>((*candidate).get())->getBBox();
            
#if 0
            std::cout << boost::format("%d %5.2f %5.2f %d\n")
                % i % (*candidate)->getXCenter() % (*candidate)->getYCenter() % (w*h);
#endif
            if (box.getArea() < 75) {
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

    cellSet.setIgnoreBad(true);           // don't visit BAD candidates
    cellSet.visitCandidates(&visitor);
    std::cout << boost::format("There are %d good candidates\n") % visitor.getN();
}

/*
 * Read an image and background subtract it
 */
std::pair<afwImage::MaskedImage<PixelT>::Ptr, afwDetect::FootprintSet<PixelT>::Ptr>
readImage() {
    afwImage::MaskedImage<PixelT>::Ptr mi;

    try {
        std::string dataDir = lsst::utils::eups::productDir("afwdata");

        std::string filename = dataDir + "/CFHT/D4/cal-53535-i-797722_1";
        
        afwGeom::BoxI bbox = afwGeom::BoxI(
            afwGeom::PointI(270, 2530), 
            afwGeom::ExtentI(512, 512)
        );
        
        lsst::daf::base::PropertySet::Ptr md;
        mi.reset(new afwImage::MaskedImage<PixelT>(filename, 0, md, bbox));
        
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
    bctrl.getStatisticsControl()->setNumSigmaClip(3.0);
    bctrl.getStatisticsControl()->setNumIter(2);

    afwImage::Image<PixelT>::Ptr im = mi->getImage();
    try {
        *mi->getImage() -= *afwMath::makeBackground(*im, bctrl).getImage<PixelT>();
    } catch(std::exception &) {
        bctrl.setInterpStyle(afwMath::Interpolate::CONSTANT);
        *mi->getImage() -= *afwMath::makeBackground(*im, bctrl).getImage<PixelT>();
    }
    /*
     * Find sources
     */
    afwDetect::Threshold threshold(5, afwDetect::Threshold::STDEV);
    int npixMin = 5;                    // we didn't smooth
    afwDetect::FootprintSet<PixelT>::Ptr fs =
        afwDetect::makeFootprintSet(*mi, threshold, "DETECTED", npixMin);
    int const grow = 1;
    bool const isotropic = false;
    afwDetect::FootprintSet<PixelT>::Ptr grownFs = afwDetect::makeFootprintSet(*fs, grow, isotropic);
    grownFs->setMask(mi->getMask(), "DETECTED");

    return std::make_pair(mi, grownFs);
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
/*
 * Run the example
 */
int main() {
    std::pair<afwImage::MaskedImage<PixelT>::Ptr, afwDetect::FootprintSet<PixelT>::Ptr> data = readImage();
    assert (data.first != NULL);        // stop compiler complaining about data being unused

    SpatialCellSetDemo();
}
