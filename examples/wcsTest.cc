// -*- LSST-C++ -*- // fixed format comment for emacs
/**
* @file
* @brief Simple test code for the Wcs Class
*        Created on:    23-Jul-2007 12:28:00 PM PDT (by NMS)
* @author Nicole M. Silvestri
*         Last modified: 20-Aug-2007 (by NMS)
*
* LSST Legalese here...
*
*/

#include <iostream>
#include <sstream>
#include <string>

#include "boost/format.hpp"
#include "boost/shared_ptr.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h" // turn off by recompiling with 'LSST_NO_TRACE 0'
#include "lsst/afw/image.h"

/**
 * @brief This test code incorporates some very simple tests of the Wcs Class
 * and its related classes.
 * 
 */

using namespace std;
namespace image = lsst::afw::image;

using lsst::daf::base::PropertySet;

int main(int argc, char **argv) {
    typedef double Pixel;

    std::string mimg;
    if (argc < 2) {
        std::string afwdata = getenv("AFWDATA_DIR");
        if (afwdata.empty()) {
            std::cerr << "I can take a default file from AFWDATA_DIR, but it's not defined." << std::endl;
            std::cerr << "Is afwdata set up?\n" << std::endl;
            exit(EXIT_FAILURE);
        } else {
            mimg = afwdata + "/small_MI";
            std::cerr << "Using " << mimg << std::endl;
        }
    } else {
        mimg = std::string(argv[1]);
    }

    const std::string inFilename(mimg);
    
    std::cout << "Opening file " << inFilename << std::endl;

    PropertySet::Ptr miMetadata(new PropertySet);
    int const hdu = 0;
    image::MaskedImage<Pixel> mskdImage(inFilename, hdu, miMetadata);
    image::Wcs wcs(miMetadata);
    
    // Testing input col, row values 

    image::PointD minCoord(1.0,1.0);
    image::PointD xy(mskdImage.getWidth(), mskdImage.getHeight());

    image::PointD sky1 = wcs.pixelToSky(minCoord);
    image::PointD sky2 = wcs.pixelToSky(xy);

    std::cout << "ra, decl of " << inFilename << " at ("<< minCoord[0] << " " << minCoord[1] <<") = " << endl;
    std::cout << "ra: " << sky1[0] << " decl: " << sky1[1] << endl << endl;
 
    std::cout << "ra, decl of " << inFilename << " at ("<< xy[0] << " " << xy[1]<<") = " << endl;
    std::cout << "ra: " << sky2[0] << " decl: " << sky2[1] << endl << endl;

    double pixArea0 = wcs.pixArea(minCoord);
    double pixArea1 = wcs.pixArea(xy);

    std::cout << "pixel areas: " << pixArea0 << " " << pixArea1 << std::endl;

    // Testing input ra, dec values using output from above for now

    double miRa1 = sky1[0];
    double miDecl1 = sky1[1];
    double miRa2 = sky2[0];
    double miDecl2 = sky2[1];

    image::PointD pix1 = wcs.skyToPixel(miRa1, miDecl1);
    image::PointD pix2 = wcs.skyToPixel(miRa2, miDecl2);

    std::cout << "col, row of " << inFilename << " at ("<< miRa1 << " " << miDecl1<<") = " << endl;
    std::cout << "col: " << pix1[0] << " row: " << pix1[1] << endl << endl;

    std::cout << "col, row of " << inFilename << " at ("<< miRa2 << " " << miDecl2<<") = " << endl;
    std::cout << "col: " << pix2[0] << " row: " << pix2[1] << endl << endl;

    double raDecl1[] = {sky1[0], sky1[1]};
    double raDecl2[] = {sky2[0], sky2[1]};

    image::PointD pix3 = wcs.skyToPixel(raDecl1);
    image::PointD pix4 = wcs.skyToPixel(raDecl2);

    std::cout << "col, row of " << inFilename << " at ("<< raDecl1[0] << " " << raDecl1[1]<<") = " << endl;
    std::cout << "col: " << pix3[0] << " row: " << pix3[1] << endl << endl;

    std::cout << "col, row of " << inFilename << " at ("<< raDecl2[0] << " " << raDecl2[1]<<") = " << endl;
    std::cout << "col: " << pix4[0] << " row: " << pix4[1] << endl << endl;    
}
