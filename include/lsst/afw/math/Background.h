#if !defined(LSST_AFW_MATH_BACKGROUND_H)
#define LSST_AFW_MATH_BACKGROUND_H
/**
 * \file
 * \brief ImageT Background
 */

#include "lsst/afw/math/Interpolate.h"

namespace lsst { namespace afw { namespace math {

/// \brief Pass parameters to a Background object
class BackgroundControl {
public:
    BackgroundControl(
                      int nxSample = 5,
                      int nySample = 5,
                      double numSigmaClip = 3.0, ///< number of standard deviations to clip at
                      int numIter = 3   ///< Number of iterations
                     ) : _nxSample(nxSample), _nySample(nySample),
                         _numSigmaClip(numSigmaClip), _numIter(numIter) {
        assert(_numSigmaClip > 0);
        assert(_numIter > 0);
        assert(_nxSample > 0);
        assert(_nxSample > 0);
    }
    int getNxSample() const { return _nxSample; }
    int getNySample() const { return _nySample; }
    double getNumSigmaClip() const { return _numSigmaClip; }
    int getNumIter() const { return _numIter; }

    void setNxSample (int nxSample) { assert(nxSample > 0); _nxSample = nxSample; }
    void setNySample (int nySample) { assert(nySample > 0); _nySample = nySample; }
    void setNumSigmaClip(double numSigmaClip) { assert(numSigmaClip > 0); _numSigmaClip = numSigmaClip; }
    void setNumIter(int numIter) { assert(numIter > 0); _numIter = numIter; }

private:
    int _nxSample;
    int _nySample;
    double _numSigmaClip;                 // Number of standard deviations to clip at
    int _numIter;                         // Number of iterations
};

    
/**
 * A class to evaluate %image background levels
 *
 * The basic strategy is to construct a Background object from an Image and
 * a statement of what we want to know.  The desired results can then be
 * returned using Background methods:
 * \code
        math::Background<ImageT> stats = math::make_Background(*img, math::NPOINT | math::MEAN);
        
        double const n = stats.getValue(math::NPOINT);
        std::pair<double, double> const mean = stats.getResult(math::MEAN); // Returns (value, error)
        double const meanError = stats.getError(math::MEAN);                // just the error
 * \endcode
 *
 * (Note that we used a helper function, \c make_Background, rather that the constructor directly so that
 * the compiler could deduce the types -- cf. \c std::make_pair)
 */
template<typename ImageT>
class Background {
public:

    explicit Background(ImageT const& img, BackgroundControl const& bgCtrl);
    typename ImageT::Pixel getPixel(int const x, int const y) const;
    ImageT getFrame() const;
    //~Background() { delete _grid; };
    
private:
    int _n;                             // number of pixels in the image
    double _meanclip;                   // n-sigma clipped mean
    ImageT _img;
    int _nxSample;
    int _nySample;
    int _subimgWidth;
    int _subimgHeight;
    std::vector<int> _xcen;
    std::vector<int> _ycen;
    std::vector<int> _xorig;
    std::vector<int> _yorig;
    std::vector<std::vector<typename ImageT::Pixel> > _grid;
    std::vector<std::vector<typename ImageT::Pixel> > _gridcolumns;
};

/// A convenience function that uses function overloading to make the correct type of Background
/// cf. std::make_pair()
template<typename ImageT>
Background<ImageT> make_Background(ImageT const& img, BackgroundControl const& bgCtrl=BackgroundControl()) { ///< ImageT (or MaskedImage) whose properties we want
    return Background<ImageT>(img, bgCtrl);
};
    
}}}

#endif  //   LSST_AFW_MATH_BACKGROUND_H
