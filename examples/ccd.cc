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
 
#include <iostream>
#include <sstream>
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom.h"

namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;
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
    afwGeom::Box2I allPixels(
        afwGeom::Point2I(0, 0),
        afwGeom::Extent2I(width + nExtended + nOverclock, height)
    );
    afwGeom::Box2I biasSec(  
        afwGeom::Point2I(i == 0 ? nExtended : width, 0),
        afwGeom::Extent2I(nOverclock, height)
    );
    afwGeom::Box2I dataSec(  
        afwGeom::Point2I(i == 0 ? nExtended + nOverclock : 0, 0),
        afwGeom::Extent2I(width, height)
    );
    //
    // Electronic properties of amplifier
    //
    float const gain[] = {1.1, 1.2};
    float const  readNoise[] = {3.5, 4.5};
    float const saturationLevel[] = {65535, 65535};
    cameraGeom::ElectronicParams::Ptr eparams(new cameraGeom::ElectronicParams(gain[i], readNoise[i],
                                                                               saturationLevel[i]));
    
    return cameraGeom::Amp::Ptr(new cameraGeom::Amp(cameraGeom::Id(i), allPixels, biasSec, dataSec,
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

/*
 * Make a Raft (== SDSS dewar) out of 5 Ccds
 */
cameraGeom::Raft::Ptr makeRaft(std::string const& name)
{
    cameraGeom::Raft::Ptr dewar(new cameraGeom::Raft(cameraGeom::Id(name), 1, 5));

    std::string filters = "riuzg";
    for (int i = 0; i != 5; ++i) {
        std::stringstream ccdName;
        ccdName << filters[i] << name;
        dewar->addDetector(afwGeom::Point2I(0, i), cameraGeom::FpPoint(0.0, 25.4*2.1*(2.0 - i)),
                           cameraGeom::Orientation(0), makeCcd(ccdName.str()));
    }

    return dewar;
}    

/*
 * Make a Camera out of 6 Rafts (==dewars)
 */
cameraGeom::Camera::Ptr makeCamera(std::string const& name)
{
    cameraGeom::Camera::Ptr camera(new cameraGeom::Camera(cameraGeom::Id(name), 6, 1));

    for (int i = 0; i != 6; ++i) {
        std::stringstream dewarName;
        dewarName << i + 1;
        camera->addDetector(afwGeom::Point2I(i, 0), cameraGeom::FpPoint(25.4*2.5*(2.5 - i), 0.0),
                            cameraGeom::Orientation(0), makeRaft(dewarName.str()));
    }

    return camera;
}    

/************************************************************************************************************/

using namespace std;
//
// Print a Ccd
//
void printCcd(std::string const& title,
              cameraGeom::Ccd::ConstPtr ccd,
              std::string const& indent = ""
             )
{
    cout << indent <<title << "CCD: " << ccd->getId().getName() << endl;
    afwGeom::Box2I const allPixels = ccd->getAllPixels();
    cout << indent <<"Total size: " << allPixels.getWidth() << " " << allPixels.getHeight() << endl;
    for (cameraGeom::Ccd::const_iterator ptr = ccd->begin(); ptr != ccd->end(); ++ptr) {
        cameraGeom::Amp::ConstPtr amp = *ptr;

        afwGeom::Box2I const biasSec = amp->getBiasSec();
        afwGeom::Box2I const dataSec = amp->getDataSec();

        cout << indent <<"   Amp: " << amp->getId().getSerial() <<
            " gain: " << amp->getElectronicParams()->getGain() << endl;

        cout << indent <<"   bias sec: " <<
            biasSec.getWidth() << "x" << biasSec.getHeight() << "+" <<
            biasSec.getMinX() << "+" << biasSec.getMinY() << endl;

        cout << indent <<"   data sec: " <<
            dataSec.getWidth() << "x" << dataSec.getHeight() << "+" <<
            dataSec.getMinX() << "+" << dataSec.getMinY() << endl;

        if (ptr == ccd->begin()) {
            cout << endl;
        }
    }
}

//
// Print a Dewar
//
void printDewar(std::string const& title,
                cameraGeom::Raft::ConstPtr const& dewar,
                std::string const& indent = ""
               )
{
    cout << indent << title << "Dewar: " << dewar->getId().getName() << endl;

    for (cameraGeom::Raft::const_iterator ptr = dewar->begin(); ptr != dewar->end(); ++ptr) {
        cameraGeom::Detector::ConstPtr det = *ptr;
        cout << indent << det->getId().getName() << " " <<
            det->getAllPixels(true).getWidth() << "x" << det->getAllPixels(true).getHeight() <<
            "   centre (mm): " << det->getCenter().getMm() << endl;
    }
}

//
// Print a Camera
//
void printCamera(std::string const& title,
                cameraGeom::Camera::ConstPtr const& camera
               )
{
    cout << title << "Camera: " << camera->getId().getName() << endl;

    for (cameraGeom::Raft::const_iterator ptr = camera->begin(); ptr != camera->end(); ++ptr) {
        printDewar("\n", boost::dynamic_pointer_cast<cameraGeom::Raft>(*ptr), "    ");
    }
}

/************************************************************************************************************/

int main()
{
    cameraGeom::Ccd::Ptr ccd = makeCcd("SITe");

    printCcd("Raw ", ccd);

    ccd->setTrimmed(true);

    cout << endl;
    printCcd("Trimmed ", ccd);
    /*
     * The SDSS camera has 6 independent dewars, each with 5 CCDs mounted on a piece of invar
     */
    cameraGeom::Raft::Ptr dewar = makeRaft("1");

    cout << endl;
    printDewar("Single ", dewar);
    /*
     * On to the camera
     */
    cameraGeom::Camera::Ptr camera = makeCamera("SDSS");

    cout << endl;
    printCamera("", camera);
}
