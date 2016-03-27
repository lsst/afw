/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <cstdio>
#include <string>
#include <algorithm>

#include "lsst/afw/image/Image.h"

namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;

template <typename PixelT>
void print(afwImage::Image<PixelT>& src, const std::string& title = "") {
    typedef typename afwImage::Image<PixelT>::x_iterator XIter;
    if (title.size() > 0) {
        printf("%s:\n", title.c_str());
    }

    printf("%3s ", "");
    for (int x = 0; x != src.getWidth(); ++x) {
        printf("%4d ", x);
    }
    printf("\n");

    for (int y = src.getHeight() - 1; y >= 0; --y) {
        printf("%3d ", y);
        for (XIter src_it = src.row_begin(y), src_end = src.row_end(y); src_it != src_end; ++src_it) {
            printf("%4g ", static_cast<float>((*src_it)[0]));
        }
        printf("\n");
    }
}

template <typename PixelT>
void printT(afwImage::Image<PixelT>& src, const std::string& _title = "") {
    std::string title =_title;
    if (title.size() > 0) {
        title += " ";
    }
    title += "transposed";
    printf("%s:\n", title.c_str());

    printf("%3s ", "");
    for (int r = 0; r != src.getHeight(); ++r) {
        printf("%4d ", r);
    }
    printf("\n");

    for (int c = 0; c != src.getWidth(); ++c) {
        printf("%3d ", c);

#if 1   // print the column from the top (there's no reverse iterator)
        typename afwImage::Image<PixelT>::y_iterator src_it = src.col_begin(c);
        for (int r = src.getHeight() - 1; r >= 0; --r) {
            printf("%4g ", static_cast<float>(src_it[r][0]));
        }
#else  // print the column from the bottom (i.e. upside down)
        for (typename afwImage::Image<PixelT>::y_iterator src_it = src.col_begin(c); src_it != src.col_end(c);
            ++src_it) {
            printf("%4g ", static_cast<float>((*src_it)[0]));
        }
#endif
        
        printf("\n");
    }
}

/************************************************************************************************************/

template <typename PixelT>
void y_gradient(const afwImage::Image<PixelT>& src, const afwImage::Image<PixelT>& dst) {
    assert(src.getDimensions() == dst.getDimensions());

    typedef typename afwImage::Image<PixelT>::const_xy_locator xy_loc;
    xy_loc src_loc = src.xy_at(0, 1);

#define USE_CACHE_LOCATION 1
#if USE_CACHE_LOCATION
    typename xy_loc::cached_location_t above = src_loc.cache_location(0,  1);
    typename xy_loc::cached_location_t below = src_loc.cache_location(0, -1);
#endif

    for (int r = 1; r < src.getHeight() - 1; ++r) {
        for (typename afwImage::Image<PixelT>::x_iterator dst_it = dst.row_begin(r);
            dst_it != dst.row_end(r); ++dst_it, ++src_loc.x()) {            
#if USE_CACHE_LOCATION                  // this version is faster
            *dst_it = (src_loc[above] - src_loc[below])/2;
#else  // but this is possible too, and more general (but slower)
            *dst_it = (src_loc(0, 1) - src_loc(0, -1))/2;
#endif
        }
        
        src_loc += afwImage::detail::difference_type(-src.getWidth(), 1);
    }
}

/************************************************************************************************************/

int main() {
    afwImage::Image<float> img(afwGeom::Extent2I(10, 6));
    // This is equivalent to img = 100:
    for (afwImage::Image<float>::iterator ptr = img.begin(); ptr != img.end(); ++ptr) {
        (*ptr)[0] = 100;
    }
    // so is this, but fills backwards
    for (afwImage::Image<float>::reverse_iterator ptr = img.rbegin(); ptr != img.rend(); ++ptr) {
        (*ptr)[0] = 100;
    }
    // so is this, but tests a different way of choosing begin()
    for (afwImage::Image<float>::iterator ptr = img.at(0, 0); ptr != img.end(); ++ptr) {
        (*ptr)[0] = 100;
    }

    afwImage::Image<float> jmg = img;

    printf("%dx%d\n", img.getWidth(), img.getHeight());

    *img.y_at(7, 2) = 999;
    *img.x_at(0, 0) = 0;
    img(img.getWidth() - 1, img.getHeight() - 1) = -100;

    print(img, "img");
    printT(img, "img");
    print(jmg, "jmg");

    afwImage::Image<float> kmg = jmg;
    kmg(0,0) = 111;
    kmg += 222;
    kmg -= 222;
    kmg += jmg;
    kmg *= 10;
    {
        afwImage::Image<float> tmp(kmg.getDimensions());
        tmp = 10;
        print(tmp, "tmp");
        kmg /= tmp;
    }
    print(kmg, "kmg");

    afwImage::Image<float> lmg(img);
    print(lmg, "lmg");

    afwImage::Image<float> mmg(img, true);
    mmg = -1;                           // shouldn't modify img
    
    printf("sub images\n");

    // img will be modified
    afwImage::Image<float> simg1(
        img, afwGeom::Box2I(
            afwGeom::Point2I(1, 1), 
            afwGeom::Extent2I(7, 3)
        ),
        afwImage::LOCAL
    );
    afwImage::Image<float> simg(
        simg1, afwGeom::Box2I(
            afwGeom::Point2I(0, 0), 
            afwGeom::Extent2I(5, 2)
        ),
        afwImage::LOCAL
    );


    {
        afwImage::Image<float> nimg(afwGeom::Extent2I(5, 2));
        nimg = 1;
        simg.assign(nimg);
    }

    print(simg, "simg");
    print(img, "img");

    printf("\n");
    for (int r = 0; r != img.getHeight(); ++r) {
        std::fill(img.row_begin(r), img.row_end(r), 100*(1 + r));
    }
    print(img, "ramp img");
    
    afwImage::Image<float> grad_y(img.getDimensions());
    grad_y = 0;
    y_gradient(img, grad_y);

    print(grad_y, "grad_y");
    
    afwImage::Image<unsigned short> u16(img.getDimensions());
    u16 = 100;
    afwImage::Image<float> fl32(u16, true); // must be true as all type conversions are deep
    print(fl32, "Float from U16");

    try {
        afwImage::Image<float> fl32(u16, false);  // will throw
    } catch(lsst::pex::exceptions::InvalidParameterError &e) {
        printf("Correctly threw exception: %s\n", e.what());
    }

    return 0;
}
