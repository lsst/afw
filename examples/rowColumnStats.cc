// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
/**
 * @file rowColumnStats.cc
 * @author Steve Bickerton
 * @brief An example executible which calls the statisticsStack function
 *
 */
#include <iostream>

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Stack.h"
#include "lsst/afw/image/ImageSlice.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;
namespace geom = lsst::afw::geom;
typedef image::Image<float> ImageF;
typedef image::ImageSlice<float> ImageSliceF;
typedef image::MaskedImage<float> MImageF;
typedef std::vector<float> VecF;
typedef boost::shared_ptr<VecF> VecFPtr;

int main(int argc, char **argv) {

    int const nX = 8;
    int const nY = 8;

    // fill an image with a gradient
    // - we want something different in x and y so we can see the different projections
    ImageF::Ptr img = ImageF::Ptr (new ImageF(geom::Extent2I(nX, nY), 0));
    for (int y = 0; y < img->getHeight(); ++y) {
        int x = 0;
        for (ImageF::x_iterator ptr = img->row_begin(y), end = img->row_end(y); ptr != end; ++ptr, ++x) {
            *ptr = 1.0*x + 2.0*y;
        }
    }

    // collapse with a MEAN over 'x' (ie. avg all columns to one), then 'y' (avg all rows to one)
    MImageF::Ptr imgProjectCol = math::statisticsStack(*img, math::MEAN, 'x');
    MImageF::Ptr imgProjectRow = math::statisticsStack(*img, math::MEAN, 'y');

    ImageSliceF slc = ImageSliceF(*(imgProjectCol->getImage()));

    ImageF::Ptr opColPlus(new ImageF(*img, true));
    *opColPlus += slc;
    ImageF::Ptr opColMinus = *img - slc;

    ImageF::Ptr opColMult  = *img * slc;
    ImageF::Ptr opColDiv   = *img / slc;

    
    ImageSliceF rowSlice = ImageSliceF(*(imgProjectRow->getImage()));
    std::vector<ImageF::Ptr> rows;
    rows.push_back(*img + rowSlice);
    rows.push_back(*img - rowSlice);
    rows.push_back(*img * rowSlice);
    rows.push_back(*img / rowSlice);


    // output the pixel values and show the statistics projections

    for (unsigned int i = 0; i < rows.size(); ++i) {
        ImageF::x_iterator end = rows[i]->row_end(0);
        printf("%26s", " ");
        for (ImageF::x_iterator ptr = rows[i]->row_begin(0); ptr != end; ++ptr) {
            printf("%5.2f ", static_cast<float>(*ptr));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    int y = 0;
    MImageF::y_iterator colEnd = imgProjectCol->col_end(0);
    for (MImageF::y_iterator pCol = imgProjectCol->col_begin(0); pCol != colEnd; ++pCol, ++y) {
        printf("%5.1f %5.1f %5.1f %5.2f : ", 
               (*opColPlus)(0, y), (*opColMinus)(0, y), (*opColMult)(0, y), (*opColDiv)(0, y));
        for (ImageF::x_iterator ptr = img->row_begin(y), end = img->row_end(y); ptr != end; ++ptr) {
            printf("%5.2f ", static_cast<float>(*ptr));
        }
        printf(" ==> %5.2f +/- %5.3f\n", pCol.image(), sqrt(pCol.variance()));
    }
    std::cout << std::endl;

    MImageF::x_iterator rowEnd = imgProjectRow->row_end(0);
    printf("%26s", " ");
    for (MImageF::x_iterator ptr = imgProjectRow->row_begin(0); ptr != rowEnd; ++ptr) {
        printf("%5.2f ", static_cast<float>(ptr.image()));
    }

    std::cout << std::endl;
}
