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

//  -*- lsst-c++ -*-
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include "boost/filesystem.hpp"
#include "lsst/afw/image.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/fits.h"
#include "lsst/utils/Utils.h"

using namespace std;

namespace dafBase = lsst::daf::base;
namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace afwFits = lsst::afw::fits;

typedef afwImage::Image<float> ImageF;
typedef afwImage::MaskedImage<float> MaskedImageF;
typedef afwImage::Exposure<float> ExposureF;

namespace cfitsio {
#if !defined(DOXYGEN)
extern "C" {
#include "fitsio.h"
}
#endif
}

static string gFilename = "", gFilename2 = "";
static string gFilenameStripped = "", gFilename2Stripped = "";
static string gFilter = "r";
static string gQueryBounds = "37.8_39_1_1.3";  // This query is good for pairing 2570/6/0199 with 5902/6/0677

//================================================================================
// tools

std::shared_ptr<afwFits::MemFileManager> readFile(string filename) {
    std::shared_ptr<afwFits::MemFileManager> result;
    std::size_t fileLen = 0;
    ifstream ifs;
    ifs.open(filename.c_str(), ios::in | ios::binary | ios::ate);
    if (!ifs) throw runtime_error("Failed to open ifstream: " + filename);
    if (ifs) {
        fileLen = ifs.tellg();
        result.reset(new afwFits::MemFileManager(fileLen));
        ifs.seekg(0, ios::beg);
        ifs.read(reinterpret_cast<char *>(result->getData()), result->getLength());
        ifs.close();
    }

    cout << "Filename/length: " << gFilename << " / " << fileLen << " bytes" << endl;

    return result;
}

string stripHierarchyFromPath(string filepath) {
    cout << "filepath A: " << filepath << endl;

    size_t lastSlash = filepath.rfind("/");
    if (lastSlash != string::npos) filepath = filepath.substr(lastSlash + 1);

    cout << "filepath B: " << filepath << endl;

    return filepath;
}

/**
 @internal Read a FITS file into an Image, write the Image to a RAM FITS file, then write the RAM FITS file to
 disk.
 */
void test6() {
    if (gFilename == "") throw runtime_error("Must specify SDSS image filename on command line");

    // Read FITS file from disk into an Image
    std::shared_ptr<dafBase::PropertySet> miMetadata(new dafBase::PropertySet);
    std::shared_ptr<ImageF> image(new ImageF(gFilename, afwFits::DEFAULT_HDU, miMetadata));

    // Write the Image to a RAM FITS file
    image->writeFits(string(gFilenameStripped + "_imageOut.fit").c_str());
    afwFits::MemFileManager manager;
    image->writeFits(manager);

    // Write the RAM FITS file to disk
    ofstream ofs;
    ofs.open(string(gFilenameStripped + "_imageRamOut.fit").c_str());
    if (ofs) ofs.write(reinterpret_cast<char *>(manager.getData()), manager.getLength());
    ofs.close();
}

/**
 @internal Read a FITS file into an Exposure, write the Exposure to a RAM FITS file, then write the RAM FITS
 file to disk.
 */
void test7() {
    if (gFilename == "") throw runtime_error("Must specify SDSS image filename on command line");

    // Read FITS file from disk into an Exposure
    std::shared_ptr<dafBase::PropertySet> miMetadata(new dafBase::PropertySet);
    std::shared_ptr<ImageF> image = std::shared_ptr<ImageF>(new ImageF(gFilename, afwFits::DEFAULT_HDU,
                                                                       miMetadata));
    MaskedImageF maskedImage(image);
    auto wcsFromFITS = std::make_shared<afwGeom::SkyWcs>(*miMetadata);
    ExposureF exposure(maskedImage, wcsFromFITS);

    // Write the Exposure to a RAM FITS file
    afwFits::MemFileManager manager;
    exposure.writeFits(manager);

    // Write the RAM FITS file to disk
    ofstream ofs;
    ofs.open(string(gFilenameStripped + "_exposureRamOut.fit").c_str());
    if (ofs) ofs.write(reinterpret_cast<char *>(manager.getData()), manager.getLength());
    ofs.close();
}

//================================================================================
// test entry point

/**
 @internal Run one test as specified by ftn.
 */
int test(void (*ftn)(void), string label) {
    cout << endl << "Running test " << label << "..." << endl;

    try {
        (*ftn)();
        cout << "  Test succeeded" << endl;
    } catch (exception &e) {
        cerr << "  Caught the following exception:" << endl;
        cerr << "    " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "  Caught a default exception" << endl;
        return 2;
    }

    return 0;
}

string GetGFilenamePath(int argc, char **argv) {
    string inImagePath;
    if (argc < 2) {
        try {
            string dataDir = lsst::utils::getPackageDir("afwdata");
            // inImagePath = dataDir + "/data/fpC-002570-r6-0199_sub.fits"; //Also works - this one was not
            // used at all in the previous avatar of this test.
            inImagePath = dataDir + "/data/fpC-005902-r6-0677_sub.fits";
        } catch (lsst::pex::exceptions::NotFoundError) {
            cerr << "Usage: maskedImage1 [inputBaseName1] [inputBaseName2] [outputBaseName1] "
                    "[outputBaseName2]"
                 << endl;
            cerr << "Warning: tests not run! Setup afwdata if you wish to use the default fitsFile." << endl;
            exit(EXIT_SUCCESS);
        }
    } else {
        inImagePath = string(argv[1]);
    }
    return inImagePath;
}

//================================================================================
// main

int main(int argc, char **argv) {
    gFilename = GetGFilenamePath(argc, argv);
    gFilenameStripped = "./tests/ramFitsIO_" + stripHierarchyFromPath(gFilename);

    int numerrs = 0;

    cout << "Testing RAM FITS..." << endl;

    numerrs += test(&test6, "6") ? 1 : 0;
    if (numerrs != 0) return EXIT_FAILURE;
    numerrs += test(&test7, "7") ? 1 : 0;
    if (numerrs != 0) return EXIT_FAILURE;

    cout << "Done testing.  Num failed tests: " << numerrs << endl;

    return EXIT_SUCCESS;
}
