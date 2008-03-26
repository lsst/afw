#include <stdexcept>
#include <sstream>
#include <vector>

#include <vw/Math.h>
#include <vw/Image.h>

using namespace std;

int main() {
    vw::ImageView<double> im(3, 3);
    
    vw::EdgeExtensionBase extension = vw::ZeroEdgeExtension();
    
    cout << "image cols = " << im.cols() << "; rows = " << im.rows() << endl;
    double rampVal = 0.0;
    for (unsigned col = 0; col < im.cols(); col++) {
        for (unsigned row = 0; row < im.rows(); row++) {
            im(col, row) = rampVal;
            ++rampVal;
        }
    }
    
    for (unsigned col = 0; col < im.cols(); col++) {
        for (unsigned row = 0; row < im.rows(); row++) {
            cout << "im(" << col << ", " << row << ") = " << im(col, row) << endl;
        }
    }

    
    cout << "image(0,0) = " << im(0,0) << " after setting ramp" << endl;
    
    int extcol = 2, extrow = 0;
    vw::EdgeExtensionView<vw::ImageView<double>, vw::ConstantEdgeExtension> extim(
        im, -extcol, -extrow, im.cols() + (2 * extcol), im.rows() + (2 * extrow)
    );
    cout << "extim cols = " << extim.cols() << "; rows = " << extim.rows() << endl;
    
    for (unsigned col = 0; col < extim.cols(); col++) {
        for (unsigned row = 0; row < extim.rows(); row++) {
            cout << "extim(" << col << ", " << row << ") = " << extim(col, row) << endl;
        }
    }
}
