/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
//  -*- lsst-c++ -*-
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "boost/shared_ptr.hpp"
#include "boost/filesystem.hpp"
#include "lsst/afw/image.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/fits.h"

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
#       include "fitsio.h"
    }
#endif
}

static int gDebugData = 0;
static string gFilename = "", gFilename2 = "";
static string gFilenameStripped = "", gFilename2Stripped = "";
static string gFilter = "r";
static string gQueryBounds = "37.8_39_1_1.3";	//This query is good for pairing 2570/6/0199 with 5902/6/0677

//================================================================================
//tools

PTR(afwFits::MemFileManager) readFile(string filename)
{
    PTR(afwFits::MemFileManager) result;
    std::size_t fileLen = 0;
	ifstream ifs;
	ifs.open(filename.c_str(), ios::in|ios::binary|ios::ate);
	if (!ifs)
		throw runtime_error("Failed to open ifstream: " + filename);
	if (ifs)
	{
		fileLen = ifs.tellg();
        result.reset(new afwFits::MemFileManager(fileLen));
		ifs.seekg(0, ios::beg);
		ifs.read(reinterpret_cast<char*>(result->getData()), result->getLength());
		ifs.close();
	}
	
	cout << "Filename/length: " << gFilename << " / " << fileLen << " bytes" << endl;
	
	return result;
}

string stripHierarchyFromPath(string filepath)
{
	cout << "filepath A: " << filepath << endl;
	
	size_t lastSlash = filepath.rfind("/");
	if (lastSlash != string::npos)
		filepath = filepath.substr(lastSlash + 1);
	
	cout << "filepath B: " << filepath << endl;
	
	return filepath;
}

/**
 Read a FITS file into an Image, write the Image to a RAM FITS file, then write the RAM FITS file to disk.
 */
void test6()
{
	if (gFilename == "")
		throw runtime_error("Must specify SDSS image filename on command line");
	
	//Read FITS file from disk into an Image
	PTR(dafBase::PropertySet) miMetadata(new dafBase::PropertySet);
	PTR(ImageF) image(new ImageF(gFilename, 0, miMetadata));
	
	//Write the Image to a RAM FITS file
	image->writeFits(string(gFilenameStripped + "_imageOut.fit").c_str());
    afwFits::MemFileManager manager;
	image->writeFits(manager);
	
	//Write the RAM FITS file to disk
	ofstream ofs;
	ofs.open(string(gFilenameStripped + "_imageRamOut.fit").c_str());
	if (ofs)
		ofs.write(reinterpret_cast<char*>(manager.getData()), manager.getLength());
	ofs.close();
}

/**
 Read a FITS file into an Exposure, write the Exposure to a RAM FITS file, then write the RAM FITS file to disk.
 */
void test7()
{
	if (gFilename == "")
		throw runtime_error("Must specify SDSS image filename on command line");
	
	//Read FITS file from disk into an Exposure
	dafBase::PropertySet::Ptr miMetadata(new dafBase::PropertySet);
	ImageF::Ptr image = ImageF::Ptr(new ImageF(gFilename, 0, miMetadata));
	MaskedImageF maskedImage(image);
	afwImage::Wcs::Ptr wcsFromFITS = afwImage::makeWcs(miMetadata); 
	ExposureF exposure(maskedImage, wcsFromFITS);
	
	//Write the Exposure to a RAM FITS file
    afwFits::MemFileManager manager;
	exposure.writeFits(manager);
	
	//Write the RAM FITS file to disk
	ofstream ofs;
	ofs.open(string(gFilenameStripped + "_exposureRamOut.fit").c_str());
	if (ofs)
		ofs.write(reinterpret_cast<char*>(manager.getData()), manager.getLength());
	ofs.close();
}

//================================================================================
//test entry point

/**
 Run one test as specified by ftn.
 */
int test(void(*ftn)(void), string label)
{
	cout << endl << "Running test " << label << "..." << endl;
	
	try
	{
		(*ftn)();
		cout << "  Test succeeded" << endl;
	}
	catch (exception &e)
	{
		cerr << "  Caught the following exception:" << endl;
		cerr << "    " << e.what() << endl;
		return 1;
	}
	catch (...)
	{
		cerr << "  Caught a default exception" << endl;
		return 2;
	}
	
	return 0;
}

//================================================================================
//main

int main(int argc, char **argv)
{
	if (argc >= 2)
	{
		stringstream ss;
		ss << argv[1];
		ss >> gDebugData;
	}
	
	if (argc >= 3)
	{
		stringstream ss;
		ss << argv[2];
		ss >> gFilename;
		gFilenameStripped = "./tests/ramFitsIO_" + stripHierarchyFromPath(gFilename);
	}
	
	if (argc >= 4)
	{
		stringstream ss;
		ss << argv[3];
		ss >> gFilename2;
		gFilename2Stripped = "./tests/ramFitsIO_" + stripHierarchyFromPath(gFilename2);
	}
	
	cout << "gDebugData:         " << gDebugData << endl;
	cout << "gFilename:          " << gFilename << endl;
	cout << "gFilename2:         " << gFilename2 << endl;
	cout << "gFilenameStripped:  " << gFilenameStripped << endl;
	cout << "gFilename2Stripped: " << gFilename2Stripped << endl;
	
	int numerrs = 0;
	
	cout << "Testing RAM FITS..." << endl;

	numerrs += test(&test6, "6") ? 1 : 0;
	if (numerrs != 0)
		return EXIT_FAILURE;
	numerrs += test(&test7, "7") ? 1 : 0;
	if (numerrs != 0)
		return EXIT_FAILURE;
	
	cout << "Done testing.  Num failed tests: " << numerrs << endl;
	
	return EXIT_SUCCESS;
}
