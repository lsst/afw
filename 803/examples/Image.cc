#include <cstdio>
#include <string>
#include <algorithm>
#include "lsst/afw/image/Image.h"

using namespace lsst::afw::image;

template <typename PixelT>
void print(Image<PixelT>& src, const std::string& title = "") {
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
        for (typename Image<PixelT>::x_iterator src_it = src.row_begin(y); src_it != src.row_end(y); ++src_it) {
            printf("%4g ", static_cast<float>((*src_it)[0]));
        }
        printf("\n");
    }
}

template <typename PixelT>
void printT(Image<PixelT>& src, const std::string& _title = "") {
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

#if 1                                   // print the column from the top (there's no reverse iterator)
        typename Image<PixelT>::y_iterator src_it = src.col_begin(c);
        for (int r = src.getHeight() - 1; r >= 0; --r) {
            printf("%4g ", static_cast<float>(src_it[r][0]));
        }
#else  // print the column from the bottom (i.e. upside down)
        for (typename Image<PixelT>::y_iterator src_it = src.col_begin(c); src_it != src.col_end(c); ++src_it) {
            printf("%4g ", static_cast<float>((*src_it)[0]));
        }
#endif
        
        printf("\n");
    }
}

/************************************************************************************************************/

template <typename PixelT>
void y_gradient(const Image<PixelT>& src, const Image<PixelT>& dst) {
    assert(src.getDimensions() == dst.getDimensions());

    typedef typename Image<PixelT>::const_xy_locator xy_loc;
    xy_loc src_loc = src.xy_at(0, 1);

#define USE_CACHE_LOCATION 1
#if USE_CACHE_LOCATION
    typename xy_loc::cached_location_t above = src_loc.cache_location(0,  1);
    typename xy_loc::cached_location_t below = src_loc.cache_location(0, -1);
#endif

    for (int r = 1; r < src.getHeight() - 1; ++r) {
        for (typename Image<PixelT>::x_iterator dst_it = dst.row_begin(r);
             						dst_it != dst.row_end(r); ++dst_it, ++src_loc.x()) {            
#if USE_CACHE_LOCATION                  // this version is faster
            *dst_it = (src_loc[above] - src_loc[below])/2;
#else  // but this is possible too, and more general (but slower)
            *dst_it = (src_loc(0, 1) - src_loc(0, -1))/2;
#endif
        }
        
        src_loc += detail::difference_type(-src.getWidth(), 1);
    }
}

/************************************************************************************************************/

int main() {
    Image<float> img(10, 6);
    // This is equivalent to img = 100:
    for (Image<float>::iterator ptr = img.begin(); ptr != img.end(); ++ptr) {
        (*ptr)[0] = 100;
    }
    // so is this, but fills backwards
    for (Image<float>::reverse_iterator ptr = img.rbegin(); ptr != img.rend(); ++ptr) {
        (*ptr)[0] = 100;
    }
    // so is this, but tests a different way of choosing begin()
    for (Image<float>::iterator ptr = img.at(0, 0); ptr != img.end(); ++ptr) {
        (*ptr)[0] = 100;
    }

    Image<float> jmg = img;

    printf("%dx%d\n", img.getWidth(), img.getHeight());

    *img.y_at(7, 2) = 999;
    *img.x_at(0, 0) = 0;
    img(img.getWidth() - 1, img.getHeight() - 1) = -100;

    print(img, "img");
    printT(img, "img");
#if 1
    print(jmg, "jmg");

    Image<float> kmg = jmg;
    kmg(0,0) = 111;
    kmg += 222;
    kmg -= 222;
    kmg += jmg;
    kmg *= 10;
#if 1
    {
        Image<float> tmp(kmg.getDimensions());
        tmp = 10;
        print(tmp, "tmp");
        kmg /= tmp;
    }
#endif
    print(kmg, "kmg");

    Image<float> lmg(img);
    print(lmg, "lmg");

    Image<float> mmg(img, true);
    mmg = -1;                           // shouldn't modify img
#endif
    
    printf("sub images\n");
#if 0
    Image<float> simg = Image<float>(img, BBox(PointI(1, 1), 5, 2)); // img will be modified
#elif 0
    Image<float> simg = Image<float>(img, BBox(PointI(1, 1), 5, 2), true); // img won't be modified
#else
    Image<float> simg1 = Image<float>(img, BBox(PointI(1, 1), 7, 3)); // img will be modified
    Image<float> simg = Image<float>(simg1, BBox(PointI(0, 0), 5, 2));
#endif

#if 0
    simg = 0;
#elif 1
    {
        Image<float> nimg = Image<float>(5, 2);
        nimg = 1;
        simg <<= nimg;
    }
#endif    

    print(simg, "simg");
    print(img, "img");

    printf("\n");
    for (int r = 0; r != img.getHeight(); ++r) {
        std::fill(img.row_begin(r), img.row_end(r), 100*(1 + r));
    }
    print(img, "ramp img");
    
    Image<float> grad_y(img.getDimensions());
    grad_y = 0;
    y_gradient(img, grad_y);

    print(grad_y, "grad_y");
    
    Image<unsigned short> u16(img.getDimensions()); u16 = 100;
    Image<float> fl32(u16, true); // must be true as all type conversions are deep
    print(fl32, "Float from U16");

    try {
        Image<float> fl32(u16, false);  // will throw
    } catch(lsst::pex::exceptions::InvalidParameterException &e) {
        printf("Correctly threw exception: %s\n", e.what());
    }

    return 0;
}
