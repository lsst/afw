#include <lsst/afw/math/FourierCutout.h>

namespace afwMath = lsst::afw::math;
    /**
     * @brief copy constructor
     * creates a shallow copy of another FourierCutout
     */
afwMath::FourierCutout::FourierCutout(afwMath::FourierCutout const & other) :
    _imageDimensions(other._imageDimensions),
    _fourierWidth(other._fourierWidth),
    _data(other._data), 
    _owner(other._owner) 
{ }

/**
 * Shift the center of the cutout by (dx, dy)
 */
void afwMath::FourierCutout::shift(double dx, double dy) { 
    int yMid = (getImageHeight()+1) /2;
    int xMid = (getImageWidth()+1)/2;
    bool evenHeight = (getImageHeight() % 2) == 0; 
    bool evenWidth = (getImageWidth() % 2) == 0;
    
    RealT u = -2.0*M_PI*dx/getImageWidth();
    RealT v = -2.0*M_PI*dy/getImageHeight();

    int rowStep = getFourierWidth();
    iterator rowStart=begin();
    //iterate over the bottom half
    for(int y = 0; y < yMid; ++y, rowStart += rowStep) {
        Complex ey = std::polar(1.0, v * y);
        iterator xIter = rowStart;
        for(int x = 0; x < xMid; ++x, ++xIter) {
            (*xIter) *= std::polar(1.0, u * x) * ey;    
        }
        if (evenWidth) {
            (*xIter) *= std::cos(u * xMid) * ey;
        }
    }
    //if even number of columns,
    //operate on the middle column
    if(evenHeight) {
        Complex ey = std::cos(v *  yMid);
        iterator xIter = rowStart;
        for(int x = 0; x< xMid; ++x, ++xIter) {
            (*xIter) *= std::polar(1.0, u * x) * ey;    
        }
        if (evenWidth) {
            (*xIter) *= std::cos(u * xMid) * ey;
        }
        ++yMid;
        rowStart += rowStep;
    }
    //operate on top half
    for(int y = yMid - getImageHeight(); y < 0; ++y, rowStart+=rowStep) {
        Complex ey = std::polar(1.0, v*y); 
        iterator xIter = rowStart;
        for(int x = 0; x< xMid; ++x, ++xIter) {
            (*xIter) *= std::polar(1.0, u * x) * ey;    
        }
        if (evenWidth) {
            (*xIter) *= std::cos(u * xMid) * ey;
        }
    }
    
}


/**
 * @brief take the derivative with respect to X
 */
void afwMath::FourierCutout::differentiateX() { 
    int yMid = (getImageHeight() +1) /2;
    int xMid = (getImageWidth()+1)/2;

    RealT u = -2.0*M_PI/getImageWidth();
    bool evenHeight = (getImageHeight() % 2) == 0;
    bool evenWidth = (getImageWidth() % 2) == 0;

    int rowStep = getFourierWidth();
    iterator rowStart = begin();
    for( int y = 0; y < yMid; ++y, rowStart += rowStep) {
        iterator xIter = rowStart;
        for(int x = 0; x < xMid; ++x, ++xIter) {
            (*xIter) *= Complex(0.0, u * x);    
        }
        if( evenWidth) {
            (*xIter) = 0.0;    
        }
    }
    if(evenHeight) {
        iterator xIter = rowStart;
        for(int x = 0; x < xMid; ++x, ++xIter) {
            (*xIter) = 0.0;
        }    

        ++yMid;
        rowStart += rowStep;
    }
    for( int y = yMid - getImageHeight(); y < 0; ++y, rowStart += rowStep) {
        iterator xIter = rowStart;
        for(int x = 0; x < xMid; ++x, ++xIter) {
            (*xIter) *= Complex(0.0, u * x);    
        }
        if( evenWidth) {
            (*xIter) = 0.0;    
        }
    }
}


/**
 * @brief take the derivative with respect to Y
 */
void afwMath::FourierCutout::differentiateY() { 
    int yMid = (getImageHeight() +1) /2;
    int xMid = (getImageWidth()+1)/2;

    RealT v = -2.0*M_PI/getImageHeight();
    bool evenHeight = (getImageHeight() % 2) == 0;
    bool evenWidth = (getImageWidth() % 2) == 0;

    int rowStep = getFourierWidth();
    iterator rowStart = begin();
    for( int y = 0; y < yMid; ++y, rowStart += rowStep) {
        Complex ey(0.0, v * y);
        iterator xIter = rowStart;
        for(int x = 0; x < xMid; ++x, ++xIter) {
            (*xIter) *= ey;    
        }
        if( evenWidth) {
            (*xIter) = 0.0;    
        }
    }
    if(evenHeight) {
        iterator xIter = rowStart;
        for(int x = 0; x < xMid; ++x, ++xIter) {
            (*xIter) = 0.0;
        }    

        ++yMid;
        rowStart += rowStep;
    }

    for( int y = yMid - getImageHeight(); y < 0; ++y, rowStart += rowStep) {
        Complex ey(0.0, v * y);
        iterator xIter = rowStart;
        for(int x = 0; x < xMid; ++x, ++xIter) {
            (*xIter) *= ey;    
        }
        if( evenWidth) {
            (*xIter) = 0.0;    
        }
    }
}

/**
 * @brief Copy the cutout to a new size, adding padding as needed.
 * @param output The dimensions of this FourierCutout will determine the 
 *   amount of "padding" rows and columns that are present. 
 * 
 * When scaling up, output will be a copy of this in the corners, with additional rows
 * and columns padding between real data.
 *
 * When scaling down, the output will not have any padding, and may loose "real" data. 
 * For that reason, one should only scale down a cutout which is known to have padding.
 * In that case, the FourierCutout::scale method, becomes a way to remove that padding
 *
 * If the dimensions of output are the same as this, and there is no need to scale,
 * a deep copy of the data will still be performed. The prefered method for deep copying
 * data is the <<= operator.
 */
void afwMath::FourierCutout::scale(afwMath::FourierCutout & output) const {
    int const rowStep = getFourierWidth();
    int const outputRowStep = output.getFourierWidth();
    int const width = std::min(rowStep, outputRowStep);

    int const inputHeight = getFourierHeight();
    int const outputHeight = output.getFourierHeight();
    int halfHeight = std::min(inputHeight, outputHeight)/2;
    
    const_iterator rowStart = begin();
    const_iterator rowEnd = rowStart + width;
    iterator outRow = output.begin();
    
    //clear the output FourierCutout
    output = 0;

    //copy the top half of the input image
    for (int i = 0; i < halfHeight; ++i) {
        std::copy(rowStart, rowEnd, outRow);
        rowStart += rowStep;
        rowEnd += rowStep;
        outRow += outputRowStep;
    }
          
    //skip the middle rows of the 'taller' image 
    if(inputHeight < outputHeight) {
        //output taller than input, skip rows in output
        outRow = output.row_begin(outputHeight - halfHeight);
        halfHeight = inputHeight - halfHeight;
    }
    else if(inputHeight > outputHeight) {
        //input taller than output, skip rows in input
        rowStart = row_begin(inputHeight - halfHeight);    
        rowEnd = rowStart + width;
        halfHeight = outputHeight-halfHeight;
    }
    else halfHeight = inputHeight - halfHeight;

    //copy the bottom half of the input image
    for (int i = 0; i < halfHeight; ++i) {
        std::copy(rowStart, rowEnd, outRow);
        rowStart += rowStep;
        rowEnd += rowStep;
        outRow += outputRowStep;
    }
    std::copy(rowStart, rowEnd, outRow);    
}

/**
 * @brief Set all pixels to a scalar value
 * @param scalar value to set pixels equal to
 */ 
afwMath::FourierCutout & afwMath::FourierCutout::operator=(RealT scalar) {
    std::fill(begin(),end(),scalar);
    return *this;
}

/**
 * @brief Multiply all pixels by a scalar value
 */
afwMath::FourierCutout & afwMath::FourierCutout::operator*=(RealT scalar) {
    for (iterator i=begin(); i!=end(); ++i) 
        (*i) *= scalar;
    return *this;
}

/**
 * @brief Increment all pixels by a scalar value
 */
afwMath::FourierCutout & afwMath::FourierCutout::operator+=(RealT scalar) {
    for (iterator i=begin(); i!=end(); ++i) 
        (*i) += scalar;
    return *this;
}

/**
 * @brief Decrement all pixels by a scalar value
 */
afwMath::FourierCutout & afwMath::FourierCutout::operator-=(RealT scalar) {
    for (iterator i=begin(); i!=end(); ++i) 
        (*i) -= scalar;
    return *this;
}

/** 
 * @brief Shallow assignment to a different FourierCutout
 * 
 * This is a shallow assignment, to perform a deep copy, use the <<= operator
 */
afwMath::FourierCutout & afwMath::FourierCutout::operator=(afwMath::FourierCutout const & other) {
    FourierCutout tmp(other);
    swap(tmp);

    return *this;
}

/**
 * @brief Pixel-wise assignment to a different FourierCutout.
 * This performs a deep copy of a FourierCutout, and thus requires that both cutouts have 
 * the same dimensions
 */
afwMath::FourierCutout & afwMath::FourierCutout::operator<<=(afwMath::FourierCutout const & other) {
    if(other.getImageDimensions() != getImageDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                "Deep copy requires a FourierCutout of equal dimensions.");
    }
    std::copy(other.begin(), other.end(), begin());
    return *this;
}
/**
 * @brief Pixel-wise multiplication, both cutouts must have the same dimensions
 */
afwMath::FourierCutout & afwMath::FourierCutout::operator*=(afwMath::FourierCutout const & other) {
    if(other.getImageDimensions() != getImageDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                "FourierCutout multiplication requires a FourierCutout of equal dimensions.");
    }

    const_iterator otherIter = other.begin();
    const_iterator const otherEnd(other.end());
    for (iterator iter = begin(); otherIter != otherEnd; ++otherIter, ++iter) {
        (*iter) *= (*otherIter);
    }
    return *this;
}

/**
 * @brief Pixel-wise addition, both cutouts must have the same dimensions
 */
afwMath::FourierCutout & afwMath::FourierCutout::operator+=(afwMath::FourierCutout const & other) {    
    if(other.getImageDimensions() != getImageDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                "FourierCutout addition requires a FourierCutout of equal dimensions.");
    }

    const_iterator otherIter = other.begin();
    const_iterator const otherEnd(other.end());
    for (iterator iter = begin(); otherIter != otherEnd; ++otherIter, ++iter) {
        (*iter) += (*otherIter);
    }
    return *this;
}

/**
 * @brief Pixel-wise subtraction, both cutouts must have the same dimensions
 */
afwMath::FourierCutout & afwMath::FourierCutout::operator-=(afwMath::FourierCutout const & other) {
    if(other.getImageDimensions() != getImageDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                "FourierCutout subtraction requires a FourierCutout of equal dimensions.");
    }

    const_iterator otherIter = other.begin();
    const_iterator const otherEnd(other.end());
    for (iterator iter = begin(); otherIter != otherEnd; ++otherIter, ++iter) {
        (*iter) -= (*otherIter);
    }
    return *this;
}
