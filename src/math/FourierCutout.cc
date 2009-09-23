#include <lsst/afw/math/FourierCutout.h>

namespace afwMath = lsst::afw::math;

struct Shift {
    typedef afwMath::FourierCutout::RealT RealT;
    typedef afwMath::FourierCutout::Complex Complex;

    RealT u;
    RealT v;
    
    Shift(RealT dx, RealT dy, int size) : u(-2*M_PI*dx / size), v(-2*M_PI*dy / size) {}
    
    inline Complex operator()(int kx, int ky) const { 
        return std::polar<RealT>(1,u*kx + v*ky); 
    }
};

struct DifferentiateX {
    typedef afwMath::FourierCutout::RealT RealT;
    typedef afwMath::FourierCutout::Complex Complex;

    Complex u;

    explicit DifferentiateX(int size) : u(0,2*M_PI/size) {}

    inline Complex operator()(int kx, int ky) const { return u * RealT(kx); }
};
    
struct DifferentiateY {
    typedef afwMath::FourierCutout::RealT RealT;
    typedef afwMath::FourierCutout::Complex Complex;
    
    Complex u;
    
    explicit DifferentiateY(int size) : u(0,2*M_PI/size) {}

    inline Complex operator()(int kx, int ky) const { return u * RealT(ky); }
};


template <typename FunctorT>
void afwMath::FourierCutout::apply(FunctorT functor) {
    int height = getFourierHeight();
    int width = getFourierWidth();

    iterator i = begin();
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x, ++i) {
            (*i) *= functor(x,y);
        }
    }
}

void afwMath::FourierCutout::shift(RealT dx, RealT dy) { 
    apply(Shift(dx,dy,getFourierSize()));
}

void afwMath::FourierCutout::differentiateX() { 
    apply(DifferentiateX(getFourierSize())); 
}

void afwMath::FourierCutout::differentiateY() { apply(DifferentiateY(getFourierSize())); }

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

afwMath::FourierCutout & afwMath::FourierCutout::operator=(RealT scalar) {
    std::fill(begin(),end(),scalar);
    return *this;
}

afwMath::FourierCutout & afwMath::FourierCutout::operator*=(RealT scalar) {
    for (iterator i=begin(); i!=end(); ++i) 
        (*i) *= scalar;
    return *this;
}

afwMath::FourierCutout & afwMath::FourierCutout::operator+=(RealT scalar) {
    for (iterator i=begin(); i!=end(); ++i) 
        (*i) += scalar;
    return *this;
}

afwMath::FourierCutout & afwMath::FourierCutout::operator-=(RealT scalar) {
    for (iterator i=begin(); i!=end(); ++i) 
        (*i) -= scalar;
    return *this;
}

/** 
 * shallow assignment
 */
afwMath::FourierCutout & afwMath::FourierCutout::operator=(afwMath::FourierCutout const & other) {
    _data = other._data;
    _owner = other._owner;
    _imageDimensions = other._imageDimensions;
    return *this;
}


afwMath::FourierCutout & afwMath::FourierCutout::operator<<=(afwMath::FourierCutout const & other) {
    if(other._imageDimensions != _imageDimensions) {
        ///TODO: throw exception: invalid argument: mismatch cutout sizes
        return *this;    
    }
    std::copy(other.begin(), other.end(), begin());
    return *this;
}
afwMath::FourierCutout & afwMath::FourierCutout::operator*=(afwMath::FourierCutout const & other) {
    if(other._imageDimensions != _imageDimensions) {
        ///TODO: throw exceptioin: invalid argument: mismatch cutout sizes
        return *this;    
    }

    const_iterator otherIter = other.begin();
    const_iterator const otherEnd(other.end());
    for (iterator iter = begin(); otherIter != otherEnd; ++otherIter, ++iter) {
        (*iter) *= (*otherIter);
    }
    return *this;
}

afwMath::FourierCutout & afwMath::FourierCutout::operator+=(afwMath::FourierCutout const & other) {    
     if(other._imageDimensions != _imageDimensions) {
        ///TODO: throw exceptioin: invalid argument: mismatch cutout sizes
        return *this;    
    }

    const_iterator otherIter = other.begin();
    const_iterator const otherEnd(other.end());
    for (iterator iter = begin(); otherIter != otherEnd; ++otherIter, ++iter) {
        (*iter) += (*otherIter);
    }
    return *this;
}

afwMath::FourierCutout & afwMath::FourierCutout::operator-=(afwMath::FourierCutout const & other) {
     if(other._imageDimensions != _imageDimensions) {
        ///TODO: throw exceptioin: invalid argument: mismatch cutout sizes
        return *this;    
    }

    const_iterator otherIter = other.begin();
    const_iterator const otherEnd(other.end());
    for (iterator iter = begin(); otherIter != otherEnd; ++otherIter, ++iter) {
        (*iter) -= (*otherIter);
    }
    return *this;
}
