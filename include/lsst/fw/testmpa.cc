#include <stdexcept>
#include <sstream>
#include <vector>

#include "testmpa.h"

using namespace std;

int main() {
    vw::ImageView<double> im(3, 3);
    vw::ImageView<int> mask(3, 3);
    
    cout << "image cols = " << im.cols() << "; rows = " << im.rows() << endl;
    int rampVal = 0;
    for (unsigned row = 0; row < im.rows(); row++) {
        for (unsigned col = 0; col < im.cols(); col++) {
            im(col, row) = static_cast<double>(rampVal);
            mask(col, row) = rampVal;
            ++rampVal;
        }
    }
    
    for (unsigned row = 0; row < im.rows(); row++) {
        for (unsigned col = 0; col < im.cols(); col++) {
            cout << "im(" << col << ", " << row << ") = " << im(col, row)
                 << "; mask(" << col << ", " << row << ") = " << mask(col, row) << endl;
        }
    }
    
    lsst::fw::MaskedPixelAccessor<double, int> mpa(im.origin(), mask.origin(), im.origin());
    
    lsst::fw::MaskedPixelAccessor<double, int> mpaRow = mpa;
    for (unsigned row = 0; row < im.rows(); row++) {
        lsst::fw::MaskedPixelAccessor<double, int> mpaCol = mpaRow;
        for (unsigned col = 0; col < im.cols(); col++) {
            cout << "im(" << col << ", " << row << ") = " << *mpaCol.camera
                 << "; mask(" << col << ", " << row << ") = " << *mpaCol.mask << endl;
            mpaCol.nextCol();
        }
        mpaRow.nextRow();
    }
}
