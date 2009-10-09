#ifndef LSST_AFW_MATH_FOURIER_CUTOUT_H
#define LSST_AFW_MATH_FOURIER_CUTOUT_H



#include <complex>
#include <algorithm>
#include <vector>
#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/make_shared.hpp>

#include <lsst/pex/exceptions/Runtime.h>

namespace lsst {
namespace afw {
namespace math {

class FourierCutoutStack;

class FourierCutout {
public:
    static int computeFourierWidth(int const &width) {
        return width/2+1;
    }

    typedef double RealT;
    typedef std::complex<RealT> Complex;
    typedef boost::shared_ptr<FourierCutout> Ptr;
    typedef boost::shared_ptr<FourierCutout const> ConstPtr;
   
    typedef Complex * iterator;
    typedef Complex const * const_iterator;

    /**
     * @brief default constructor
     */
    explicit FourierCutout() :
        _imageDimensions(0,0),
        _fourierWidth(0),
        _data(0)
    {}

    /**
     * @brief Construct a FourierCutout from image dimensions.
     * The dimensions of the cutout will be (image width /2 +1) by image height
     * @param width real-space width of the image
     * @param height real-space height of the image     
     */
    explicit FourierCutout(int const & width, int const & height) :
        _imageDimensions(height, width),
        _fourierWidth(computeFourierWidth(width)),
        _data(new Complex[getFourierSize()]),
        _owner(_data, Deleter())
    { }

    /**
     * @brief copy constructor
     * creates a shallow copy of another FourierCutout
     */
    FourierCutout(FourierCutout const & other);

    /**
     * @brief get the dimensions of the image as a (height, width) pair
     */
    std::pair<int,int> getImageDimensions() const {return _imageDimensions;}
    /**
     * @brief get the dimensions of the FourierCutout as a (height, width) pair
     */
    std::pair<int,int> getFourierDimensions() const {
        return std::make_pair<int,int> (getFourierWidth(), getFourierHeight());
    }
    int getImageWidth() const {return _imageDimensions.second;}
    int getImageHeight() const {return _imageDimensions.first;}
    int getImageSize() const {return getImageHeight()*getImageWidth();}
    int getFourierWidth() const {return _fourierWidth;}
    int getFourierHeight() const {return _imageDimensions.first;}
    int getFourierSize() const {return getFourierHeight()*getFourierWidth();}
      

    iterator begin() {return _data;}
    iterator end() {return _data + getFourierSize();}
    iterator row_begin(int i) {return _data + getFourierWidth()*i;}
    iterator row_end(int i) {return row_begin(i+1);}
    iterator at(int x, int y) {return row_begin(y)+ x;}

    const_iterator begin() const {return _data;}
    const_iterator end() const {return _data + getFourierSize();}
    const_iterator row_begin(int i) const {return _data + getFourierWidth()*(i);}
    const_iterator row_end(int i) const {return row_begin(i+1);}
    const_iterator at(int x, int y) const {return row_begin(y)+ x;}
    
    void shift(double dx, double dy);
    void differentiateX();
    void differentiateY();

    void scale(FourierCutout & output) const;

    Complex operator()(int x, int y) const {        
        return *at(x,y);   
    }
    FourierCutout & operator=(RealT scalar);
    FourierCutout & operator=(FourierCutout const & other);
    FourierCutout & operator<<=(FourierCutout const & other);
    FourierCutout & operator*=(RealT scalar);
    FourierCutout & operator*=(FourierCutout const & other);
    FourierCutout & operator+=(RealT scalar);
    FourierCutout & operator+=(FourierCutout const & other);
    FourierCutout & operator-=(RealT scalar);
    FourierCutout & operator-=(FourierCutout const & other);

    boost::shared_ptr<Complex> getOwner() const {return _owner;}

    void swap(FourierCutout & other) {
        boost::swap(_owner, other._owner);
        std::swap(_data, other._data);
        std::swap(_imageDimensions.first, other._imageDimensions.first);
        std::swap(_imageDimensions.second, other._imageDimensions.second);
        std::swap(_fourierWidth, other._fourierWidth);
    }
private:
    std::pair<int, int> _imageDimensions;
    int _fourierWidth;
    Complex * _data;
    boost::shared_ptr<Complex> _owner;

#if !defined SWIG

    friend class FourierCutoutStack;
    explicit FourierCutout(int const & width, int const & height, Complex * data, boost::shared_ptr<Complex> owner) :
        _imageDimensions(height, width),
        _fourierWidth(computeFourierWidth(width)),
        _data(data),
        _owner(owner)
    {}

    struct Deleter {
        void operator()(Complex * p) {delete [] p;}
    };
#endif
};

class FourierCutoutStack {
    typedef FourierCutout::Deleter Deleter;
public:
    typedef FourierCutout::Complex Complex;
    typedef std::vector<FourierCutout::Ptr> FourierCutoutVector;
    typedef boost::shared_ptr<FourierCutoutStack> Ptr;
    typedef boost::shared_ptr<FourierCutoutStack const> ConstPtr;
    
    FourierCutout::Ptr getCutout(int i) const {
        if( i< 0 || i >= _depth) {
            if(_depth == 0) {
                throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException, 
                        "Zero-depth FourierCutoutStack object");
            }
            
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeException,
                   (boost::format("Index %1% must be in range [0, %2%)")
                    % i % _depth).str()
            );
        }

        //cannot use boost::make_shared because constructor used is private 
        //memeber of FourierCutout
        return FourierCutout::Ptr(
                new FourierCutout(
                        _imageDimensions.second, _imageDimensions.first,
                        _stackData.get() + _cutoutSize*i,
                        _stackData
                )
        );
    }
    
    FourierCutoutVector getCutoutVector(int begin = 0, int n = 0) const {
        if(begin < 0 || begin >= _depth) {
            if(_depth == 0) {
                throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException, 
                        "Zero-depth FourierCutoutStack object");
            }

            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeException,
                   (boost::format("Start index %1% must be in range [0, %2%)")
                    % begin % _depth).str()
            );
        } 

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
    explicit FourierCutoutStack(int const & width, int const & height, int const & depth) :
        _depth(depth), 
        _cutoutSize(FourierCutout::computeFourierWidth(width)*height),
        _imageDimensions(height, width),        
        _stackData(new Complex[_cutoutSize*_depth], Deleter())
    { }
    explicit FourierCutoutStack() : 
        _depth(0),
        _cutoutSize(0),
        _imageDimensions(0,0)
    { }

    FourierCutoutStack(FourierCutoutStack const & other) :
        _depth(other._depth),
        _cutoutSize(other._cutoutSize),
        _imageDimensions(other._imageDimensions),
        _stackData(other._stackData)
    { }

    /**
     * @brief Retrieve the number of cutouts in the tack
     */
    int getStackDepth() const {return _depth;}

    /**
     * @brief Retrieve the size of a single cutout on the stack
     */
    int getCutoutSize() const {return _cutoutSize;}
    
    /**
     * @brief Retrieve the image space dimensions
     */
    std::pair<int,int> getImageDimensions() const {return _imageDimensions;}

    /**
     *@brief Shallow assignment
     */
    FourierCutoutStack const & operator=(FourierCutoutStack const & other) {
        FourierCutoutStack tmp(other);        
        swap(tmp);

        return *this;
    }

    /**
     *@brief Retrieve a shared_ptr to the underlying memory
     */
    boost::shared_ptr<Complex> getData() const {return _stackData;}

    /** 
     * Swap with another FourierCutoutStack
     */
    void swap(FourierCutoutStack & other) {
        boost::swap(_stackData, other._stackData);
        std::swap(_imageDimensions.first, other._imageDimensions.first);
        std::swap(_imageDimensions.second, other._imageDimensions.second);
        std::swap(_depth, other._depth);
        std::swap(_cutoutSize, other._cutoutSize);
    }
private:
    int _depth;
    int _cutoutSize;
    std::pair<int, int> _imageDimensions;    
    boost::shared_ptr<Complex> _stackData;
};

}}} //end namespace lsst::afw::math

#endif // !LSST_AFW_MATH_FOURIER_CUTOUT_H
