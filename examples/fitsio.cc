#include <iostream>
#include "DiskImageResourceFITS.h"
#include "Exception.h"

using namespace lsst::fits;

int main(int ac, char **av) {
    typedef vw::ImageView<float> my_image_type;

    my_image_type image;
    try {
        vw::read_image(image, av[1]);   // is the filetype registered?
    } catch(vw::Exception &e) {
        try {
            std::cerr << av[1] << " is not a registered file type; trying FITS\n";
            lsst::fits::read(image, av[1]);
        } catch(vw::Exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }

    int r = 1;
    int c = 42;
    for (int dr = 1; dr >= -1; dr--) {
        std::cout << r + dr << "  ";
        for (int dc = -1; dc <= 1; dc++) {
            std::cout << image(c + dc, r + dr) << " ";
        }
        std::cout << "\n";
    }

#if 0
    for (my_image_type::iterator iter = image.begin(); iter != image.end(); iter++) {
        std::cout << *iter << "\n";
    }
#endif

    if (ac <= 2) {
        return 0;
    }
    //
    // Test output
    //
    write_image(av[2], image);

    my_image_type image2;
    try {
        lsst::fits::read(image2, av[2]);
    } catch(vw::Exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    for (int dr = 1; dr >= -1; dr--) {
        std::cout << r + dr << "  ";
        for (int dc = -1; dc <= 1; dc++) {
            std::cout << image(c + dc, r + dr) << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
