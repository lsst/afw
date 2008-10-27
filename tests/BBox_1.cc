#include <vw/Math/BBox.h>
#include <iostream>

int main() {
    int code = 0;                       // return code

     vw::BBox2i bb;
     vw::math::Vector<float,2> point;

     point[0] = 1.0;
     point[1] = 1.0;

     bb.grow(point);

     if (bb.contains(point)) {
	  std::cout << "OK" << std::endl;
     } else {
	  std::cout << "Fails" << std::endl;
          code++;
     }

     return code;
}

     
