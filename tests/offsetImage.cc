// -*- LSST-C++ -*-
//
// A machine-precision test of offsetImage(), complementary to the tests in offsetImage.py
//
// TODO: only bilinear interpolation is fully tested for now; should add lanczos and nearest-neighbor

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
#include <boost/random.hpp>

#include "lsst/afw/geom.h"
#include "lsst/afw/math.h"
#include "lsst/afw/image.h"
#include "lsst/afw/detection.h"

using namespace std;
using namespace boost;
using namespace lsst::afw::geom;
using namespace lsst::afw::math;
using namespace lsst::afw::image;


static random::mt19937 rng(0);  // RNG deliberately initialized with same seed every time
static random::uniform_int_distribution<> uni_int(0,100);
static random::uniform_01<> uni_double;


static void fill_random(Image<double> &im)
{
    int nx = im.getWidth();
    int ny = im.getHeight();

    double sum = 0.0;
    for (int i = 0; i < nx; i++) {
	for (int j = 0; j < ny; j++) {
	    im(i,j) = uni_double(rng);
	    sum += im(i,j);
	}
    }

    // image must be normalized to sum 1, due to current confusion in Psf::computeImage()
    for (int i = 0; i < nx; i++)
	for (int j = 0; j < ny; j++)
	    im(i,j) /= sum;
}


static inline void show_image(const Image<double> &im)
{
    int nx = im.getWidth();
    int ny = im.getHeight();

    cerr << "(x0,y0)=(" << im.getX0() << "," << im.getY0() << ")\n";
    for (int i = 0; i < nx; i++) {
	for (int j = 0; j < ny; j++)
	    cerr << " " << im(i,j);
	cerr << endl;
    }
}


// -------------------------------------------------------------------------------------------------
//
// testFlipOffset(): offsetting an image by dx and (-dx) should be related by the appropriate flip symmetry


static PTR(Image<double>) xflip(const Image<double> &im)
{
    int nx = im.getWidth();
    int ny = im.getHeight();
   
    PTR(Image<double>) ret = make_shared<Image<double> >(nx,ny);
    ret->setXY0(-im.getX0()-nx+1, im.getY0());

    for (int i = 0; i < nx; i++)
	for (int j = 0; j < ny; j++)
	    (*ret)(i,j) = im(nx-1-i,j);

    return ret;
}


static PTR(Image<double>) yflip(const Image<double> &im)
{
    int nx = im.getWidth();
    int ny = im.getHeight();
   
    PTR(Image<double>) ret = make_shared<Image<double> >(nx,ny);
    ret->setXY0(im.getX0(), -im.getY0()-ny+1);

    for (int i = 0; i < nx; i++)
	for (int j = 0; j < ny; j++)
	    (*ret)(i,j) = im(i,ny-1-j);

    return ret;
}


// returns 0 on success, 1 on failure
static int testOneFlipOffset(int nx, int ny, int x0, int y0, double dx, double dy, const string &algorithmName, unsigned int buffer)
{
    // identifier string which will be displayed on failure
    ostringstream os;
    os << "testOneFlipOffset(" << nx << "," << ny << "," << x0 << "," << y0 << "," << dx << "," << dy << "," << algorithmName << "," << buffer << ")";
    string s = os.str();

    // initialize PSF to random image
    Image<double> src(nx, ny);
    src.setXY0(x0, y0);
    fill_random(src);

    Image<double>::Ptr dst = offsetImage(src, dx, dy, algorithmName, buffer);

    Image<double>::Ptr dst2 = xflip(src);
    dst2 = offsetImage(*dst2, -dx, dy, algorithmName, buffer);
    dst2 = xflip(*dst2);

    Image<double>::Ptr dst3 = yflip(src);
    dst3 = offsetImage(*dst3, dx, -dy, algorithmName, buffer);
    dst3 = yflip(*dst3);

    if ((dst->getX0() != dst2->getX0()) || (dst->getX0() != dst3->getX0()) || (dst->getY0() != dst2->getY0()) || (dst->getY0() != dst3->getY0())) {
	cerr << s << "XY0 mismatch\n";
	return 1;
    }

    for (int i = 0; i < nx; i++) {
	for (int j = 0; j < ny; j++) {
	    // Note that offsetImage() is partly in single precision, so we test with 10^-6 threshold
	    if ((fabs((*dst)(i,j) - (*dst2)(i,j)) > 1.0e-6) || (fabs((*dst)(i,j) - (*dst3)(i,j)) > 1.0e-6)) {
		cerr << s << ": image mismatch\n";
#if 0
		cerr << "source image follows\n";
		show_image(src);
		cerr << "offset image follows\n";
		show_image(*dst);
		cerr << "xflip->offset->xflip image follows\n";
		show_image(*dst2);
		cerr << "yflip->offset->yflip image follows\n";
		show_image(*dst3);
#endif
		return 1;
	    }
	}
    }

    cerr << s << ": pass\n";
    return 0;
}


static int testFlipOffset()
{
    int n = 0;
    n += testOneFlipOffset(5, 5, 2, 2, 0.3, 0.2, "bilinear", 1);
    n += testOneFlipOffset(5, 5, 2, 2, -0.8, 0.3, "bilinear", 1);
    n += testOneFlipOffset(5, 5, 2, 2, -10.2, -8.2, "bilinear", 1);
    n += testOneFlipOffset(5, 5, 2, 2, 0.3, 0.2, "lanczos5", 5);
    n += testOneFlipOffset(5, 5, 2, 2, -0.8, 0.3, "lanczos5", 5);
    n += testOneFlipOffset(5, 5, 2, 2, -10.2, -8.2, "lanczos5", 5);
    return n;
}


// -------------------------------------------------------------------------------------------------
//
// a precise test of afw::math::offsetImage() with bilinear interpolation, by comparing with
// quick-and-dirty handcoded bilinear interpolation


// returns 0 on success, 1 on failure
static int testOneOffsetBilinear(int nx, int ny, int x0, int y0, double dx, double dy)
{
    // identifier string which will be displayed on failure
    ostringstream os;
    os << "testOneOffsetBilinear(" << nx << "," << ny << "," << x0 << "," << y0 << "," << dx << "," << dy << ")";
    string s = os.str();

    // initialize PSF to random image
    Image<double> src(nx, ny);
    src.setXY0(x0, y0);
    fill_random(src);

    Image<double>::Ptr dst = offsetImage(src, dx, dy, "bilinear", 1);

    // test correctness...
    // Note that offsetImage() is partly in single precision, so we test with 10^-6 threshold

    if ((dst->getWidth() != nx) || (dst->getHeight() != ny)) {
	cerr << s << ": mismatched dst dimensions\n";
	return 1;
    }

    int dst_x0 = dst->getX0();
    int dst_y0 = dst->getY0();

    int expected_dst_x0 = (fabs(dx) < 1.0) ? x0 : round(x0+dx);
    int expected_dst_y0 = (fabs(dy) < 1.0) ? y0 : round(y0+dy);

    if ((dst_x0 != expected_dst_x0) || (dst_y0 != expected_dst_y0)) {
	cerr << s << ": wrong image location\n";
	return 1;
    }

    for (int i = 0; i < nx; i++) {
	double si = i + dst_x0 - dx - x0;  // location in src image
	int si0 = floor(si);
	double sx = si-si0;

	for (int j = 0; j < ny; j++) {
	    double sj = j + dst_y0 - dy - y0;  // location in src image
	    int sj0 = floor(sj);
	    double sy = sj-sj0;
	    
	    // quick-and-dirty bilinear interpolation by hand
	    double t00 = (si0 >= 0 && sj0 >= 0) ? src(si0,sj0) : 0.0;
	    double t01 = (si0 >= 0 && sj0 < ny-1) ? src(si0,sj0+1) : 0.0;
	    double t10 = (si0 < nx-1 && sj0 >= 0) ? src(si0+1,sj0) : 0.0;
	    double t11 = (si0 < nx-1 && sj0 < ny-1) ? src(si0+1,sj0+1) : 0.0;
	    double t = t00*(1-sx)*(1-sy) + t01*(1-sx)*sy + t10*sx*(1-sy) + t11*sx*sy;
	    
	    if (fabs((*dst)(i,j)-t) > 1.0e-6) {
		cerr << s << ": incorrect output image at i=" << i << ", j=" << j << endl;
		cerr << "got " << (*dst)(i,j) << ", expected " << t << endl;
#if 0
		cerr << "t00=" << t00 << " t01=" << t01 << " t10=" << t10 << " t11=" << t11 << endl;
		cerr << "complete source image follows\n";
		show_image(src);
#endif
		return 1;
	    }
	}
    }

    cerr << s << ": pass\n";
    return 0;
}


static int testOffsetBilinear()
{
    int n = 0;
    n += testOneOffsetBilinear(5, 5, 2, 2, 10.0, 10.0);
    n += testOneOffsetBilinear(5, 5, 2, 2, 9.71, 9.65);
    n += testOneOffsetBilinear(5, 5, 2, 2, 9.9, 10.2);
    n += testOneOffsetBilinear(5, 5, 2, 2, 10.1, 9.8);
    n += testOneOffsetBilinear(5, 5, 2, 2, 10.1, 10.2);
    n += testOneOffsetBilinear(5, 5, 2, 2, 0.9, -0.8);
    return n;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    int n1 = testFlipOffset();
    if (n1 > 0)
	cerr << "testFlipOffset: " << n1 << " failures\n";

    int n2 = testOffsetBilinear();
    if (n2 > 0)
	cerr << "testOffsetBilinear: " << n2 << " failures\n";

    return (n1 > 0) || (n2 > 0);
}

/*
 * Local variables:
 *  c-basic-offset: 4
 * End:
 */
