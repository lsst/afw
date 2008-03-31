#include <iostream>
#include <vw/Image/Manipulation.h>
#include "lsst/pex/exceptions.h"

int main(int ac, char **av) {
    if (ac < 2) {
        std::cerr << "Please provide a filename for me to read" << std::endl;
        return 1;
    }

#if 0
    typedef vw::ImageView<float> my_image_type;
#else
    typedef vw::ImageView<unsigned int> my_image_type;
#endif
    my_image_type image;
    try {
        vw::read_image(image, av[1]);   // is the filetype registered?
    } catch(vw::Exception &e) {
        try {
            std::cerr << av[1] << " is not a registered file type; trying FITS\n";
            read(image, av[1]);
        } catch(vw::Exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }

    int r = 1;
    int c = 41;
    const int nrow = 3;
    const int ncol = 3;
    my_image_type::pixel_accessor row(vw::crop(image, c - 1, r - 1, ncol, nrow).impl().origin());
    for (int dr = -1; dr < nrow - 1; dr++) {
        my_image_type::pixel_accessor col = row;
        std::cout << r + dr << "  ";
        for (int dc = -1; dc < ncol - 1; dc++) {
            std::cout << *col << " ";
            col.next_col();
        }
        std::cout << "\n";
        row.next_row();
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
        read(image2, av[2]);
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
