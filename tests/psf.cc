// -*- LSST-C++ -*-

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
using namespace lsst::afw::detection;


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


static void show_image(const Image<double> &im)
{
    int nx = im.getWidth();
    int ny = im.getHeight();

    for (int i = 0; i < nx; i++) {
	for (int j = 0; j < ny; j++)
	    cerr << " " << im(i,j);
	cerr << endl;
    }
}


// -------------------------------------------------------------------------------------------------
//
// testPsfOffsets


// returns 0 on success, 1 on failure
static int testOnePsfOffset(int dst_nx, int dst_ny, int src_nx, int src_ny, int ctrx, int ctry)
{
    // identifier string which will be displayed on failure
    ostringstream os;
    os << "testOnePsfOffset(" << dst_nx << "," << dst_ny << "," << src_nx << "," << src_ny << "," << ctrx << "," << ctry << ")";
    string s = os.str();

    // initialize PSF to random image
    Image<double> src(src_nx, src_ny);
    fill_random(src);

    // construct PSF object
    Kernel::Ptr ker = make_shared<FixedKernel>(src);
    ker->setCtr(Point2I(ctrx,ctry));
    KernelPsf psf(ker);

    // coordinates where we will ask for the PSF (these must be integers but otherwise arbitrary)
    int x = uni_int(rng);
    int y = uni_int(rng);

    // ask for PSF image
    Image<double>::Ptr dst = psf.computeImage(Point2D(x,y),
					      Extent2I(dst_nx,dst_ny),
					      false,    // normalizePeak
					      false);   // distort

    // test correctness...

    if ((dst->getWidth() != dst_nx) || (dst->getHeight() != dst_ny)) {
	cerr << s << ": mismatched dst dimensions\n";
	return 1;
    }

    int x0 = dst->getX0();
    int y0 = dst->getY0();

    if (x0 > x || x0+dst_nx <= x) {
        cerr << s << ": psf image does not include base point (x,y)\n";
        return 1;
    }

    if (y0 > y || y0+dst_ny <= y) {
        cerr << s << ": psf image does not include base point (x,y)\n";
        return 1;
    }        

#if 0
    cerr << s << endl;
    cerr << "source image follows\n";
    show_image(src);

    cerr << "dst image follows (x,y)=(" << x << "," << y << ")  (x0,y0)=(" << x0 << "," << y0 << ")\n";
    show_image(*dst);
#endif    

    for (int i = 0; i < dst_nx; i++) {
	for (int j = 0; j < dst_ny; j++) {
	    // location in src image corresponding to (i,j)
	    int sx = i+x0-x+ctrx;
	    int sy = j+y0-y+ctry;
	    double t = ((sx >= 0) && (sx < src_nx) && (sy >= 0) && (sy < src_ny)) ? src(sx,sy) : 0.0;
	    
	    if (fabs((*dst)(i,j) - t) > 1.0e-13) {
		cerr << s << ": incorrect output image\n";
		return 1;
	    }
	}
    }

    cerr << s << ": pass\n";
    return 0;
}
	
// returns number of failures
static int testPsfOffsets()
{
    int n = 0;
    n += testOnePsfOffset(5, 5, 5, 5, 2, 2);
    n += testOnePsfOffset(5, 5, 5, 5, 1, 2);
    n += testOnePsfOffset(5, 5, 5, 5, 2, 3);
    n += testOnePsfOffset(3, 5, 5, 5, 2, 2);
    n += testOnePsfOffset(5, 3, 5, 5, 2, 2);
    n += testOnePsfOffset(7, 5, 5, 5, 2, 2);
    n += testOnePsfOffset(5, 7, 5, 5, 2, 2);
    n += testOnePsfOffset(8, 4, 3, 10, 2, 4);
    n += testOnePsfOffset(4, 9, 3, 1, 0, 0);
    n += testOnePsfOffset(1, 1, 3, 1, 0, 0);
    n += testOnePsfOffset(12, 15, 10, 11, 3, 7);
    n += testOnePsfOffset(5, 6, 12, 10, 1, 8);
    n += testOnePsfOffset(6, 4, 10, 8, 9, 0);
    n += testOnePsfOffset(3, 2, 7, 9, 3, 4);
    return n;
}


// -------------------------------------------------------------------------------------------------
//
// testPsfRecenter
//

// returns 0 on success, 1 on failure
static int testOnePsfRecenter(int nx, int ny, int ctrx, int ctry, double px, double py)
{
    // identifier string which will be displayed on failure
    ostringstream os;
    os << "testOnePsfRecenter(" << nx << "," << ny << "," << ctrx << "," << ctry << "," << px << "," << py << ")";
    string s = os.str();

    // initialize PSF to random image
    PTR(Image<double>) src = make_shared<Image<double> >(nx,ny);
    fill_random(*src);

    // use bilinear interpolation for simplest machine-precision test 
    // (can't go through Psf::computeImage() since this will use lanczos5)
    PTR(Image<double>) dst = Psf::recenterKernelImage(src, Point2I(ctrx,ctry), Point2D(px,py), "bilinear", 1);

    // test correctness...

    if ((dst->getWidth() != nx) || (dst->getHeight() != ny)) {
	cerr << s << ": mismatched dst dimensions\n";
	return 1;
    }

    int x0 = dst->getX0();
    int y0 = dst->getY0();

    if ((fabs(x0+ctrx-px) > 0.5+1.0e-13) || (fabs(y0+ctry-py) > 0.5+1.0e-13)) {
	cerr << s << ": wrong image location\n";
	return 1;
    }

    for (int i = 1; i < nx-1; i++) {
	double si = i+x0-px+ctrx;  // location in src image
	int si0 = floor(si);
	double sx = si-si0;

	for (int j = 1; j < ny-1; j++) {
	    double sj = j+y0-py+ctry;  // location in src image
	    int sj0 = floor(sj);
	    double sy = sj-sj0;
	    
	    // quick-and-dirty bilinear interpolation by hand
	    double t00 = (si0 >= 0 && sj0 >= 0) ? (*src)(si0,sj0) : 0.0;
	    double t01 = (si0 >= 0 && sj0 < ny-1) ? (*src)(si0,sj0+1) : 0.0;
	    double t10 = (si0 < nx-1 && sj0 >= 0) ? (*src)(si0+1,sj0) : 0.0;
	    double t11 = (si0 < nx-1 && sj0 < ny-1) ? (*src)(si0+1,sj0+1) : 0.0;
	    double t = t00*(1-sx)*(1-sy) + t01*(1-sx)*sy + t10*sx*(1-sy) + t11*sx*sy;
	    
	    // offsetImage() includes some single-precision arithmetic, so threshold is 10^-6 here
	    if (fabs((*dst)(i,j)-t) > 1.0e-6) {
		cerr << s << ": incorrect output image at i=" << i << ", j=" << j << endl;
		cerr << setprecision(17) << "got " << (*dst)(i,j) << ", expected " << t << endl;
		cerr << setprecision(17) << "t00=" << t00 << " t01=" << t01 << " t10=" << t10 << " t11=" << t11 << endl;
		cerr << "complete source image follows\n";
		show_image(*src);
		return 1;
	    }
	}
    }

    cerr << s << ": pass\n";
    return 0;
}


static int testPsfRecenter()
{
    int n = 0;
    n += testOnePsfRecenter(5, 5, 2, 2, 10.0, 10.0);
    n += testOnePsfRecenter(5, 5, 2, 2, 9.71, 9.65);
    n += testOnePsfRecenter(5, 5, 2, 2, 9.9, 10.2);
    n += testOnePsfRecenter(5, 5, 2, 2, 10.1, 9.8);
    n += testOnePsfRecenter(5, 5, 2, 2, 10.1, 10.2);
    return n;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    int n1 = testPsfOffsets();
    if (n1 > 0)
	cerr << "testPsfOffsets: " << n1 << " failures\n";

    int n2 = testPsfRecenter();
    if (n2 > 0)
	cerr << "testPsfRecenter: " << n2 << " failures\n";

    return (n1+n2 > 0);
}

/*
 * Local variables:
 *  c-basic-offset: 4
 * End:
 */
