// -*- LSST-C++ -*- // fixed format comment for emacs
/**
* \file wcsTest.cc
* \brief Simple test code for the WCS Class
*        Created on:    23-Jul-2007 12:28:00 PM PDT (by NMS)
* \author Nicole M. Silvestri
*         Last modified: 20-Aug-2007 (by NMS)
*
* LSST Legalese here...
*
*/

#include <iostream>
#include <sstream>
#include <string>

#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>
#include <vw/Image.h>
#include <vw/Math/BBox.h>

#include <lsst/pex/exceptions.h>
#include <lsst/pex/utils/Trace.h> // turn off by recompiling with 'LSST_NO_TRACE 0'
#include <lsst/afw/image.h>

/**
 * \brief This test code incorporates some very simple tests of the WCS Class
 * and its related classes.
 * 
 */

using namespace std;

int main(int argc, char **argv) {
    typedef double pixelType;

    const std::string inFilename(argv[1]);
    
   
    std::cout << "Opening file " << inFilename << std::endl;
    lsst::afw::image::MaskedImage<pixelType, lsst::afw::image::maskPixelType> mskdImage;
    mskdImage.readFits(inFilename);
    lsst::afw::image::WCS wcs(mskdImage.getImage()->getMetaData());
    
    // Testing input col, row values 

    lsst::afw::Coord2D minCoord(1.0,1.0);
    lsst::afw::Coord2D colRow(mskdImage.getCols(), mskdImage.getRows());

    lsst::afw::Coord2D sky1 = wcs.colRowToRaDec(minCoord);
    lsst::afw::Coord2D sky2 = wcs.colRowToRaDec(colRow);

    std::cout << "ra, decl of " << inFilename << " at ("<< minCoord[0] << " " << minCoord[1] <<") = " << endl;
    std::cout << "ra: " << sky1[0] << " decl: " << sky1[1] << endl << endl;
 
    std::cout << "ra, decl of " << inFilename << " at ("<< colRow[0] << " " << colRow[1]<<") = " << endl;
    std::cout << "ra: " << sky2[0] << " decl: " << sky2[1] << endl << endl;

    double pixArea0 = wcs.pixArea(minCoord);
    double pixArea1 = wcs.pixArea(colRow);

    std::cout << "pixel areas: " << pixArea0 << " " << pixArea1 << std::endl;

    // Testing input ra, dec values using output from above for now

    double miRa1 = sky1[0];
    double miDecl1 = sky1[1];
    double miRa2 = sky2[0];
    double miDecl2 = sky2[1];

    lsst::afw::Coord2D pix1 = wcs.raDecToColRow(miRa1, miDecl1);
    lsst::afw::Coord2D pix2 = wcs.raDecToColRow(miRa2, miDecl2);

    std::cout << "col, row of " << inFilename << " at ("<< miRa1 << " " << miDecl1<<") = " << endl;
    std::cout << "col: " << pix1[0] << " row: " << pix1[1] << endl << endl;

    std::cout << "col, row of " << inFilename << " at ("<< miRa2 << " " << miDecl2<<") = " << endl;
    std::cout << "col: " << pix2[0] << " row: " << pix2[1] << endl << endl;

    double raDecl1[2];
    raDecl1[0] = sky1[0];
    raDecl1[1] = sky1[1];

    double raDecl2[2];
    raDecl2[0] = sky2[0];
    raDecl2[1] = sky2[1];

    lsst::afw::Coord2D pix3 = wcs.raDecToColRow(raDecl1);
    lsst::afw::Coord2D pix4 = wcs.raDecToColRow(raDecl2);

        std::cout << "col, row of " << inFilename << " at ("<< raDecl1[0] << " " << raDecl1[1]<<") = " << endl;
    std::cout << "col: " << pix3[0] << " row: " << pix3[1] << endl << endl;

    std::cout << "col, row of " << inFilename << " at ("<< raDecl2[0] << " " << raDecl2[1]<<") = " << endl;
    std::cout << "col: " << pix4[0] << " row: " << pix4[1] << endl << endl;    

} // close main
