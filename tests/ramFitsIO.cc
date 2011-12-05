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
#include "boost/shared_ptr.hpp"
#include "boost/filesystem.hpp"
#include "lsst/afw/image.h"
#include "lsst/afw/image/Image.h"

using namespace std;

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

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

pair<boost::shared_ptr<char>, long> readFile(string filename)
{
	boost::shared_ptr<char> fileContents;
	long fileLen = 0;
	ifstream ifs;
	ifs.open(filename.c_str(), ios::in|ios::binary|ios::ate);
	if (!ifs)
		throw runtime_error("Failed to open ifstream: " + filename);
	if (ifs)
	{
		fileLen = ifs.tellg();
		fileContents = boost::shared_ptr<char>(new char[fileLen]);
		ifs.seekg(0, ios::beg);
		ifs.read(fileContents.get(), fileLen);
		ifs.close();
	}
	
	cout << "Filename/length: " << gFilename << " / " << fileLen << " bytes" << endl;
	
	return pair<boost::shared_ptr<char>, long>(fileContents, fileLen);
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

//================================================================================
//tests

/**
 Read a FITS file from disk into a ram buffer.  Read the ram buffer into a cfitsio object.
 Copy the cfitsio object to another object.  Dump the cfitsio object to a FITS file.
 */
void test1()
{
	if (gFilename == "")
		throw runtime_error("Must specify SDSS image filename on command line");
	
	pair<boost::shared_ptr<char>, size_t> file = readFile(gFilename);
	
	cfitsio::fitsfile *ff_in = 0; 
	int status = 0;
	char *fileContents = file.first.get();
	if (fits_open_memfile(&ff_in, "UnusedFilenameParameter", READONLY, (void**)&fileContents,
						  &file.second, 0,
						  NULL/*Memory allocator unnecessary for READONLY*/, &status) != 0)
	{
		cout << "fits_open_memfile err" << endl;
	}
	else
	{
		cfitsio::fitsfile *ff_out = 0;
		if (fits_create_file(&ff_out, string("!" + gFilenameStripped + "_cfitsioInOut.fit").c_str(), &status) != 0)
		{
			cout << "fits_create_file err" << endl;
		}
		else
		{
			if (fits_copy_file(ff_in, ff_out, 1, 1, 1, &status) != 0)
			{
				cout << "fits_copy_file err" << endl;
			}
			else
			{
				if (fits_close_file(ff_out, &status) != 0)
				{
					cout << "fits_close_file err" << endl;
				}
			}
		}
	}
}

/**
Read a FITS file from disk into a ram buffer.  Read the ram buffer into an Image.  Dump the Image to a FITS file.
 */
void test2()
{
	if (gFilename == "")
		throw runtime_error("Must specify SDSS image filename on command line");
	
	pair<boost::shared_ptr<char>, size_t> file = readFile(gFilename);
	
	char* ramFile = file.first.get();
	dafBase::PropertySet::Ptr miMetadata(new dafBase::PropertySet);
	ImageF::Ptr image = ImageF::Ptr(new ImageF(&ramFile, &file.second, 0, miMetadata));
	
	image->writeFits(string("!" + gFilenameStripped + "_imageInOut.fit").c_str());
}

/**
 Test Coadd::map().
 */
/*
void test3()
{
	if (gFilename == "")
		throw runtime_error("Must specify SDSS image filename on command line");
	
	pair<boost::shared_ptr<char>, long> file = readFile(gFilename);
	
	string queriesStrsSerialized = string("QUERY_1.099e-4_0_1000000_") + gFilter + "_" + gQueryBounds + "\n";
	
	if (file.first != NULL)
	{
		vector<pair<char*, int> > intersections =
			Coadd::map(queriesStrsSerialized, file.first.get(), file.second, gFilename, gDebugData);
		
		cout << "Map done, Intersection details per query:" << endl;
		for (int i = 0; i < intersections.size(); ++i)
		{
			if (intersections[i].first)
				cout << "  Intersection ramFITS length: " << intersections[i].second << endl;
			else cout << "  Empty intersection" << endl;
		}
	}
}
*/

/**
 Read in a FITS file from disk, copy it to another FITS file, then write the copy to disk.
 */
void test4()
{
	if (gFilename == "")
		throw runtime_error("Must specify SDSS image filename on command line");
	
	cfitsio::fitsfile *ff_in = 0; 
	int status = 0;
	if (fits_open_file(&ff_in, gFilename.c_str(), READONLY, &status) != 0)
		throw runtime_error("fits_open_file err");
	
	cfitsio::fitsfile *ff_out = 0;
	if (fits_create_file(&ff_out, string("!" + gFilenameStripped + "_fitsOut.fit").c_str(), &status) != 0)
	{
		cout << "status: " << status << endl;
		throw runtime_error("fits_create_file err");
	}
	
	if (fits_copy_file(ff_in, ff_out, 1, 1, 1, &status) != 0)
	{
		cout << "status: " << status << endl;
		throw runtime_error("fits_copy_file err");
	}
	
	if (fits_close_file(ff_out, &status) != 0)
	{
		cout << "status: " << status << endl;
		throw runtime_error("fits_close_file err");
	}
}

/**
 Read in a FITS file from disk, copy it to a RAM FITS file, then write the RAM FITS file to disk.
 */
void test5()
{
	if (gFilename == "")
		throw runtime_error("Must specify SDSS image filename on command line");
	
	//Open a FITS file from disk
	cfitsio::fitsfile *ff_in = 0; 
	int status = 0;
	if (fits_open_file(&ff_in, gFilename.c_str(), READONLY, &status) != 0)
		throw runtime_error("fits_open_file err");
	
	//Create a RAM FITS file
	cfitsio::fitsfile *ff_out = 0;
	size_t ramFileLen = 2880;	//Initial buffer size (file length)
	size_t deltaSize = 0;	//0 is a flag that this parameter will be ignored and the default 2880 used instead
	char *ramFile = new char[ramFileLen];	//Mem allocation will be handled by cfitsio
	
	cout << "ramFile: " << (long)ramFile << endl;
	cout << "ramFileLen: " << ramFileLen << endl;
	
	if (fits_create_memfile(&ff_out, (void**)&ramFile,
		&ramFileLen, deltaSize, &realloc, &status) != 0)
	{
		cout << "status: " << status << endl;
		throw runtime_error("fits_create_memfile err");
	}
	
	cout << "ramFile: " << (long)ramFile << endl;
	cout << "ramFileLen: " << ramFileLen << endl;
	
	//Copy the file read from disk to the RAM file
	if (fits_copy_file(ff_in, ff_out, 1, 1, 1, &status) != 0)
	{
		cout << "status: " << status << endl;
		throw runtime_error("fits_copy_file err");
	}
	
	cout << "ramFile: " << (long)ramFile << endl;
	cout << "ramFileLen: " << ramFileLen << endl;
	
	//This must be done after fits_copy_file() or the subsequently reading the FITS buffer will not retrieve all the data
	if (fits_flush_file(ff_out, &status) != 0)
	{
		cout << "status: " << status << endl;
		throw runtime_error("fits_flush_file err");
	}
	
	//Write the RAM FITS file to disk
	ofstream ofs;
        std::string oFilename = boost::filesystem::path(gFilename).filename().stem().native();
        oFilename += "_fitsRamOut.fit";

	ofs.open(oFilename.c_str());
	if (ofs)
		ofs.write(ramFile, ramFileLen);
	ofs.close();
        ::unlink(oFilename.c_str());
	
	cout << "ramFile: " << (long)ramFile << endl;
	cout << "ramFileLen: " << ramFileLen << endl;
	
	if (fits_close_file(ff_out, &status) != 0)
	{
		cout << "status: " << status << endl;
		throw runtime_error("fits_close_file err");
	}
}

/**
 Read a FITS file into an Image, write the Image to a RAM FITS file, then write the RAM FITS file to disk.
 */
void test6()
{
	if (gFilename == "")
		throw runtime_error("Must specify SDSS image filename on command line");
	
	//Read FITS file from disk into an Image
	dafBase::PropertySet::Ptr miMetadata(new dafBase::PropertySet);
	ImageF::Ptr image = ImageF::Ptr(new ImageF(gFilename, 0, miMetadata));
	
	//Write the Image to a RAM FITS file
	char *ramFile = NULL;
	size_t ramFileLen = 0;
	image->writeFits(string(gFilenameStripped + "_imageOut.fit").c_str());
	image->writeFits(&ramFile, &ramFileLen);
	
	//Write the RAM FITS file to disk
	ofstream ofs;
	ofs.open(string(gFilenameStripped + "_imageRamOut.fit").c_str());
	if (ofs)
		ofs.write(ramFile, ramFileLen);
	ofs.close();
}

/**
 Read a FITS file into an Exposure, write the Exposure to a RAM FITS file, then write the RAM FITS file to disk.
 */
void test7()
{
	if (gFilename == "")
		throw runtime_error("Must specify SDSS image filename on command line");
	
	char *ramFile = NULL;
	size_t ramFileLen = 0;
	
	//Read FITS file from disk into an Exposure
	dafBase::PropertySet::Ptr miMetadata(new dafBase::PropertySet);
	ImageF::Ptr image = ImageF::Ptr(new ImageF(gFilename, 0, miMetadata));
	MaskedImageF maskedImage(image);
	afwImage::Wcs::Ptr wcsFromFITS = afwImage::makeWcs(miMetadata); 
	ExposureF exposure(maskedImage, *wcsFromFITS);
	
	//Write the Exposure to a RAM FITS file
	exposure.writeFits(&ramFile, &ramFileLen);
	
	//Write the RAM FITS file to disk
	ofstream ofs;
	ofs.open(string(gFilenameStripped + "_exposureRamOut.fit").c_str());
	if (ofs)
		ofs.write(ramFile, ramFileLen);
	ofs.close();
}

/**
 Test Coadd::reduce().
 */
 /*
void test8()
{
	cout << "This test will work best if given two images and a query that all overlap." << endl;
	cout << "  e.g.:   fpC-002570-r6-0199.fit    fpC-005902-r6-0677.fit    37.8-39 x 1-1.3" << endl;
	
	if (gFilename == "" || gFilename2 == "")
		throw runtime_error("Must specify two SDSS image filenames on command line");
	
	pair<boost::shared_ptr<char>, long> file1 = readFile(gFilename);
	pair<boost::shared_ptr<char>, long> file2 = readFile(gFilename2);
	
	vector<pair<char*, int> > intersections1, intersections2;	//Vectors by query of intersections for one input image
	
	string queriesStrsSerialized = string("QUERY_1.099e-4_0_1000000_") + gFilter + "_" + gQueryBounds + "\n";
	
	if (file1.first != NULL)
	{
		intersections1 = Coadd::map(queriesStrsSerialized, file1.first.get(), file1.second, gFilename, gDebugData);
		
		cout << "Map done, File 1 intersection lengths (bytes):" << endl;
		for (int i = 0; i < intersections1.size(); ++i)
			cout << intersections1[i].second << endl;
	}
	if (file2.first != NULL)
	{
		intersections2 = Coadd::map(queriesStrsSerialized, file2.first.get(), file2.second, gFilename2, gDebugData);
		
		cout << "Map done, File 2 intersection lengths (bytes):" << endl;
		for (int i = 0; i < intersections2.size(); ++i)
			cout << intersections2[i].second << endl;
	}
	
	vector<pair<char*, int> > intersections;	//Vector by input image of intersections for one query
	
	if (intersections1[0].first)
		intersections.push_back(intersections1[0]);
	if (intersections2[0].first)
		intersections.push_back(intersections2[0]);
	
	Coadd::reduce(intersections);
}
*/

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
	
	numerrs += test(&test1, "1") ? 1 : 0;
	if (numerrs != 0)
		return EXIT_FAILURE;
	numerrs += test(&test2, "2") ? 1 : 0;
	if (numerrs != 0)
		return EXIT_FAILURE;
	//numerrs += test(&test3, "3") ? 1 : 0;
	//if (numerrs != 0)
	//	return EXIT_FAILURE;
	numerrs += test(&test4, "4") ? 1 : 0;
	if (numerrs != 0)
		return EXIT_FAILURE;
	numerrs += test(&test5, "5") ? 1 : 0;
	if (numerrs != 0)
		return EXIT_FAILURE;
	numerrs += test(&test6, "6") ? 1 : 0;
	if (numerrs != 0)
		return EXIT_FAILURE;
	numerrs += test(&test7, "7") ? 1 : 0;
	if (numerrs != 0)
		return EXIT_FAILURE;
	//numerrs += test(&test8, "8") ? 1 : 0;
	//if (numerrs != 0)
	//	return EXIT_FAILURE;
	
	cout << "Done testing.  Num failed tests: " << numerrs << endl;
	
	return EXIT_SUCCESS;
}
