#ifndef LSST_AFW_MATH_FOURIER_CUTOUT_H
#define LSST_AFW_MATH_FOURIER_CUTOUT_H

#include <complex>
#include <algorithm>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace lsst {
namespace afw {
namespace math {

class FourierCutoutStack;

class FourierCutout {
public:
    typedef double RealT;
    typedef std::complex<RealT> Complex;
    typedef boost::shared_ptr<FourierCutout> Ptr;
    typedef boost::shared_ptr<FourierCutout const> ConstPtr;
   
    typedef Complex * iterator;
    typedef Complex const * const_iterator;

    // Allocate and set to a constant.
    explicit FourierCutout(std::pair<int, int> const & imageDimensions) :
        _imageDimensions(imageDimensions),
        _data(new Complex[getFourierSize()]),
        _owner(_data, Deleter())
    { }

    // Shallow copy.
    explicit FourierCutout(FourierCutout const & other) :
        _imageDimensions(other._imageDimensions), 
        _data(other._data), 
        _owner(other._owner) 
    {}

    std::pair<int,int> getImageDimensions() const {return _imageDimensions;}
    std::pair<int,int> getFourierDimensions() const {
        return std::make_pair<int,int> (getFourierWidth(), getFourierHeight());
    }
    int getImageWidth() const {return _imageDimensions.first;}
    int getImageHeight() const {return _imageDimensions.second;}
    int getImageSize() const {return getImageHeight()*getImageWidth();}
    int getFourierWidth() const {return computeFourierWidth(_imageDimensions.first);}
    int getFourierHeight() const {return _imageDimensions.second;}
    int getFourierSize() const {return getFourierHeight()*getFourierWidth();}
    
    iterator begin() { return _data; }
    iterator end() { return _data + getFourierSize(); }
    iterator row_begin(int i) {return _data + getFourierWidth()*i;}
    iterator row_end(int i) {return row_begin(i+1);}

    const_iterator begin() const { return _data; }
    const_iterator end() const { return _data + getFourierSize(); }
    const_iterator row_begin(int i) const {return _data + getFourierWidth()*i;}
    const_iterator row_end(int i) const {return row_begin(i+1);}
    
    void shift(RealT dx, RealT dy);
    void differentiateX();
    void differentiateY();

    void scale(FourierCutout & output) const;

    FourierCutout & operator=(RealT scalar);
    FourierCutout & operator=(FourierCutout const & other);
    FourierCutout & operator<<=(FourierCutout const & other);
    FourierCutout & operator*=(RealT scalar);
    FourierCutout & operator*=(FourierCutout const & other);
    FourierCutout & operator+=(RealT scalar);
    FourierCutout & operator+=(FourierCutout const & other);
    FourierCutout & operator-=(RealT scalar);
    FourierCutout & operator-=(FourierCutout const & other);

private:
    static int computeFourierWidth(int const &width) {
        return width/2+1;
    }

    struct Deleter {
        void operator()(Complex * p) const { delete [] p; }
    };

    std::pair<int, int> _imageDimensions;
    Complex * _data;
    boost::shared_ptr<Complex> _owner;


    template <typename FunctorT> 
    void apply(FunctorT functor);

    friend class FourierCutoutStack;
    explicit FourierCutout(std::pair<int, int> imageDimensions, Complex * data, boost::shared_ptr<Complex> owner) :
        _imageDimensions(imageDimensions),
        _data(data),
        _owner(owner)
    {}
};

class FourierCutoutStack {
    typedef FourierCutout::Complex Complex;
    typedef FourierCutout::Deleter Deleter;

public:
    typedef std::vector<FourierCutout::Ptr> FourierCutoutVector;
    typedef boost::shared_ptr<FourierCutoutStack> Ptr;
    typedef boost::shared_ptr<FourierCutoutStack const> ConstPtr;
    
    FourierCutout::Ptr getCutout(int i) const {
        if( i< 0 || i > _depth)
            ///TODO:throw exception, array out of bounds
            return FourierCutout::Ptr();

        return FourierCutout::Ptr(new FourierCutout(
                _imageDimensions,
                _stackData.get() + getCutoutSize()*i,
                _stackData
        ));
    }
    FourierCutoutVector getCutoutVector(int begin = 0, int n = 0) const {                
        if(begin < 0 || begin > _depth)
            return FourierCutoutVector();
        
        if (n <= 0)
            n = _depth;

        int end = std::min(begin+n, _depth);
        if( end <= begin)
            return FourierCutoutVector();        
        
        n = end - begin;
        FourierCutoutVector cutoutVector(n);
        FourierCutoutVector::iterator i = cutoutVector.begin();
        for(int cutoutId = begin; cutoutId < end; ++cutoutId, ++i) {            
            (*i) = getCutout(cutoutId);
        }

        return cutoutVector;
    }
    explicit FourierCutoutStack(int stackDepth, std::pair<int, int> imageDimensions) :
        _depth(stackDepth), 
        _imageDimensions(imageDimensions),        
        _stackData(new Complex[getCutoutSize()*_depth], Deleter())
    { }
    explicit FourierCutoutStack() : 
        _depth(0),
        _imageDimensions(0,0)
    { }

    FourierCutoutStack(FourierCutoutStack const & other) :
        _depth(other._depth),
        _imageDimensions(other._imageDimensions),
        _stackData(other._stackData)
    {}

    int getStackDepth() const {return _depth;}
    int getCutoutSize() const {
        int width = FourierCutout::computeFourierWidth(_imageDimensions.first);
        return width*_imageDimensions.second;
    }

    FourierCutout::Ptr operator[](int const i) const {return getCutout(i);}
    FourierCutoutStack const & operator=(FourierCutoutStack const & other) {
        _depth = other._depth;
        _imageDimensions = other._imageDimensions,
        _stackData = other._stackData;
        return *this;
    }

    boost::shared_ptr<Complex> getData() const {return _stackData;}

private:
    int _depth;
    std::pair<int, int> _imageDimensions;    
    boost::shared_ptr<Complex> _stackData;
};

}}} //end namespace lsst::afw::math

#endif // !LSST_AFW_MATH_FOURIER_CUTOUT_H
