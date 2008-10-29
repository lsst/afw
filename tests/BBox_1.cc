#include <iostream>
#include "lsst/afw/image/Image.h"

namespace image = lsst::afw::image;

int main() {
    int code = 0;                       // return code

     image::BBox bb;
     image::PointI point(1.0, 1.0);

     bb.grow(point);

     if (bb.contains(point)) {
	  std::cout << "OK" << std::endl;
     } else {
	  std::cout << "Fails" << std::endl;
          code++;
     }

     return code;
}

     
