#include <iostream>
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom.h"

namespace afwImage = lsst::afw::image;
namespace cameraGeom = lsst::afw::cameraGeom;

/*
 * Make an Amp
 */
cameraGeom::Amp::Ptr makeAmp(int const i // which amp? (i == 0 ? left : right)
                            )
{
    //
    // Layout of active and overclock pixels in the as-readout data  The layout is:
    //     Amp0 || extended | overclock | data || data | overclock | extended || Amp1
    // for each row; all rows are identical in drift-scan data
    //
    int const height = 1361;            // number of rows in a frame
    int const width = 1024;             // number of data pixels read out through one amplifier
    int const nExtended = 8;            // number of pixels in the extended register
    int const nOverclock = 32;          // number of (horizontal) overclock pixels 
    //
    // Construct the needed bounding boxes given that geometrical information.
    //
    // Note that all the offsets are relative to the origin of this amp, not to its eventual
    // position in the CCD
    //
    afwImage::BBox allPixels(afwImage::PointI(0,                                   0),
                             width + nExtended + nOverclock, height);
    afwImage::BBox biasSec(  afwImage::PointI(i == 0 ? nExtended : width,          0),
                             nOverclock,                     height);
    afwImage::BBox dataSec(  afwImage::PointI(i == 0 ? nExtended + nOverclock : 0, 0),
                             width,                          height);
    //
    // Electronic properties of amplifier
    //
    float const gain[] = {1.1, 1.2};
    float const  readNoise[] = {3.5, 4.5};
    float const saturationLevel[] = {65535, 65535};
    cameraGeom::ElectronicParams::Ptr eparams(new cameraGeom::ElectronicParams(gain[i], readNoise[i],
                                                                               saturationLevel[i]));
    
    return cameraGeom::Amp::Ptr(new cameraGeom::Amp(cameraGeom::Id(i), allPixels, biasSec, dataSec,
                                                    (i == 0 ? cameraGeom::Amp::LLC : cameraGeom::Amp::LRC),
                                                    eparams));
}

/*
 * Make a Ccd out of 2 Amps
 */
cameraGeom::Ccd::Ptr makeCcd(std::string const& name)
{
    const float pixelSize = 24e-3;     // pixel size in mm
    cameraGeom::Ccd::Ptr ccd(new cameraGeom::Ccd(cameraGeom::Id(name), pixelSize));

    for (int i = 0; i != 2; ++i) {
        ccd->addAmp(i, 0, *makeAmp(i));
    }

    return ccd;
}    

/************************************************************************************************************/

using namespace std;
//
// Print a Ccd
//
void printCcd(std::string const& title,
              cameraGeom::Ccd const& ccd)
{
    cout << title << "CCD: " << ccd.getId().getName() << endl;
    afwImage::BBox const allPixels = ccd.getAllPixels();
    cout << "Total size: " << allPixels.getWidth() << " " << allPixels.getHeight() << endl;
    for (cameraGeom::Ccd::const_iterator ptr = ccd.begin(); ptr != ccd.end(); ++ptr) {
        cameraGeom::Amp::ConstPtr amp = *ptr;

        afwImage::BBox const biasSec = amp->getBiasSec();
        afwImage::BBox const dataSec = amp->getDataSec();

        cout << "   Amp: " << amp->getId().getSerial() <<
            " gain: " << amp->getElectronicParams()->getGain() << endl;

        cout << "   bias sec: " <<
            biasSec.getWidth() << "x" << biasSec.getHeight() << "+" <<
            biasSec.getX0() << "+" << biasSec.getY0() << endl;

        cout << "   data sec: " <<
            dataSec.getWidth() << "x" << dataSec.getHeight() << "+" <<
            dataSec.getX0() << "+" << dataSec.getY0() << endl;

        if (ptr == ccd.begin()) {
            cout << endl;
        }
    }
}

int main()
{
    cameraGeom::Ccd::Ptr ccd = makeCcd("SDSS");

    printCcd("Raw ", *ccd);

    ccd->setTrimmed(true);

    cout << endl;
    printCcd("Trimmed ", *ccd);
}
