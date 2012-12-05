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
using namespace lsst::afw::geom;
using namespace lsst::afw::math;
using namespace lsst::afw::image;
using namespace lsst::afw::detection;


static boost::random::mt19937 rng(0);  // RNG deliberately initialized with same seed every time
static boost::random::uniform_int_distribution<> uni_int(0,100);
static boost::random::uniform_01<> uni_double;


// returns 0 on success, 1 on failure
static int testOnePsfOffset(int dst_nx, int dst_ny, int src_nx, int src_ny, int ctrx, int ctry)
{
    // identifier string which will be displayed on failure
    ostringstream os;
    os << "testOnePsfOffset(" << dst_nx << "," << dst_ny << "," << src_nx << "," << src_ny << "," << ctrx << "," << ctry << ")";
    string s = os.str();

    // initialize PSF to random image
    // image must be normalized to sum 1, due to confusion in Psf::computeImage()

    Image<double> src(src_nx, src_ny);

    double sum = 0.0;
    for (int i = 0; i < src_nx; i++) {
	for (int j = 0; j < src_ny; j++) {
	    src(i,j) = uni_double(rng);
	    sum += src(i,j);
	}
    }
    for (int i = 0; i < src_nx; i++)
	for (int j = 0; j < src_ny; j++)
	    src(i,j) /= sum;

    // construct PSF object
    Kernel::Ptr ker = boost::make_shared<FixedKernel>(src);
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

#if 0
    cerr << s << endl;
    cerr << "source image follows\n";
    for (int i = 0; i < src_nx; i++) {
	for (int j = 0; j < src_ny; j++)
	    cerr << " " << src(i,j);
	cerr << endl;
    }
    cerr << "(x,y)=(" << x << "," << y << ")  (x0,y0)=(" << x0 << "," << y0 << ")\n";
    for (int i = 0; i < dst_nx; i++) {
	for (int j = 0; j < dst_ny; j++)
	    cerr << " " << (*dst)(i,j);
	cerr << endl;
    }
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

    cerr << s << ": pass\n";  // remove
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
    n += testOnePsfOffset(8, 4, 3, 10, 3, 4);
    n += testOnePsfOffset(4, 9, 3, 1, 0, 3);
    return n;
}


int main(int argc, char **argv)
{
    int n = testPsfOffsets();
    if (n > 0)
	cerr << "testPsfOffsets: " << n << " failures\n";

    return (n > 0);
}

/*
 * Local variables:
 *  c-basic-offset: 4
 * End:
 */
