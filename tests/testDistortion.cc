#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DISTORTION
#include <boost/test/unit_test.hpp>

#include <boost/random.hpp>
#include <boost/make_shared.hpp>

#include "lsst/afw.h"

using namespace std;
using namespace boost;
using namespace Eigen;
using namespace lsst::afw::geom;
using namespace lsst::afw::math;
using namespace lsst::afw::image;
using namespace lsst::afw::detection;
using namespace lsst::afw::cameraGeom;
using namespace lsst::afw::geom::ellipses;


static random::mt19937 rng(0);  // RNG deliberately initialized with same seed every time
static random::uniform_int_distribution<> uni_int(0,100);
static random::uniform_01<> uni_double;


// -------------------------------------------------------------------------------------------------
//
// Helper functions


static inline Point2D randpt()
{
    // returns randomly located point in [-100,100] x [-100,100]
    return Point2D(200*uni_double(rng)-100, 200*uni_double(rng)-100);
}

static inline double dist(const Point2D &p1, const Point2D &p2)
{
    double dx = p1.getX() - p2.getX();
    double dy = p1.getY() - p2.getY();
    return sqrt(dx*dx + dy*dy);
}

static inline double dist(const AffineTransform &a1, const AffineTransform &a2)
{
    double ret = 0.0;
    for (int i = 0; i < 6; i++)
        ret += (a1[i]-a2[i]) * (a1[i]-a2[i]);
    return sqrt(ret);
}

static inline double compare(const Image<double> &im1, const Image<double> &im2)
{
    assert(im1.getWidth() == im2.getWidth());
    assert(im1.getHeight() == im2.getHeight());
    assert(im1.getX0() == im2.getX0());
    assert(im1.getY0() == im2.getY0());

    double t11 = 0.0;
    double t12 = 0.0;
    double t22 = 0.0;

    int nx = im1.getWidth();
    int ny = im1.getHeight();

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double x = im1(i,j);
            double y = im2(i,j);
            t11 += x*x;
            t12 += (x-y)*(x-y);
            t22 += y*y;
        }
    }

    assert(t11 > 0.0);
    assert(t22 > 0.0);
    return sqrt(fabs(t12) / sqrt(t11*t22));
}

static shared_ptr<RadialXYTransform> makeRandomRadialXYTransform()
{
    vector<double> coeffs(7, 0.0);
    double t = 1.0;

    for (unsigned int i = 1; i < coeffs.size(); i++) {
        t *= 0.01;
        coeffs[i] = t * (uni_double(rng) - 0.5);
    }

    coeffs[1] += 1.0;
    return make_shared<RadialXYTransform> (coeffs);
}


// -------------------------------------------------------------------------------------------------
//
// End-to-end test of afw::geom::makeAffineTransformFromTriple()


BOOST_AUTO_TEST_CASE(affineTransformFromTriple)
{
    Point2D p1 = randpt();
    Point2D p2 = randpt();
    Point2D p3 = randpt();

    Point2D q1 = randpt();
    Point2D q2 = randpt();
    Point2D q3 = randpt();

    AffineTransform a = makeAffineTransformFromTriple(p1,p2,p3,q1,q2,q3);

    BOOST_CHECK(dist(a(p1),q1) < 1.0e-10);
    BOOST_CHECK(dist(a(p2),q2) < 1.0e-10);
    BOOST_CHECK(dist(a(p3),q3) < 1.0e-10);
}


// -------------------------------------------------------------------------------------------------
//
// End-to-end test of static member function RadialXYTransform::makeAffineTransform()


BOOST_AUTO_TEST_CASE(radialAffineTransform)
{
    Point2D p = randpt();
    double rnew = 50 * uni_double(rng);
    double rprime = uni_double(rng);

    AffineTransform a = RadialXYTransform::makeAffineTransform(p.getX(), p.getY(), rnew, rprime);

    double x = p.getX();
    double y = p.getY();
    double r = sqrt(x*x + y*y);
    
    Point2D q = a(Point2D(x,y));
    Point2D q2 = Point2D(rnew*x/r, rnew*y/r);

    // test base point of transform
    BOOST_CHECK(dist(q,q2) < 1.0e-10);

    q = a(Point2D(2*x, 2*y));
    q2 = Point2D(rnew*x/r + rprime*x, rnew*y/r + rprime*y);

    // test radial direction of transform
    BOOST_CHECK(dist(q,q2) < 1.0e-10);

    q = a(Point2D(x+y, y-x));
    q2 = Point2D(rnew*(x+y)/r, rnew*(y-x)/r);

    // test tangential direction of transform
    BOOST_CHECK(dist(q,q2) < 1.0e-10);
}


// -------------------------------------------------------------------------------------------------


class ToyDetector : public Detector
{
public:
    explicit ToyDetector(AffineTransform const &a)
        : Detector(Id()), _a(a), _ainv(a.invert()) { }

    virtual Point2D getPixelFromPosition(FpPoint const &pos) const
    {
        return _a(pos.getMm());
    }

    virtual FpPoint getPositionFromPixel(Point2D const &pix, bool const isTrimmed) const
    {
        return FpPoint(_ainv(pix));
    }

    static PTR(ToyDetector) makeRandom()
    {
        AffineTransform a;
        a[0] = uni_double(rng) + 1.0;
        a[1] = uni_double(rng);
        a[2] = uni_double(rng);
        a[3] = uni_double(rng) + 1.0;
        a[4] = 100 * (uni_double(rng) - 0.5);
        a[5] = 100 * (uni_double(rng) - 0.5);

        return make_shared<ToyDetector> (a);
    }

protected:
    AffineTransform _a, _ainv;
};


BOOST_AUTO_TEST_CASE(detectorTransform)
{
    PTR(Detector) det = ToyDetector::makeRandom();
    
    Point2D p = randpt();
    FpPoint q = det->getPositionFromPixel(p);

    // test round trip
    BOOST_CHECK(dist(det->getPixelFromPosition(q),p) < 1.0e-10);

    AffineTransform a = det->linearizePositionFromPixel(p);
    AffineTransform b = det->linearizePixelFromPosition(q);
    Point2D r = randpt();
    
    // test linearization
    BOOST_CHECK(dist(det->getPositionFromPixel(r).getMm(), a(r)) < 1.0e-10);
    BOOST_CHECK(dist(det->getPixelFromPosition(FpPoint(r)), b(r)) < 1.0e-10);
}


// -------------------------------------------------------------------------------------------------


static double linearizationResidual(const XYTransform &tr, const Point2D &p, 
                                    double step, bool forward)
{
    Point2D px = p + Extent2D(step,0);
    Point2D py = p + Extent2D(0,step);

    Point2D q = forward ? tr.forwardTransform(p) : tr.reverseTransform(p);
    Point2D qx = forward ? tr.forwardTransform(px) : tr.reverseTransform(px);
    Point2D qy = forward ? tr.forwardTransform(py) : tr.reverseTransform(py);

    AffineTransform a = forward ? tr.linearizeForwardTransform(p) : tr.linearizeReverseTransform(p);
    return dist(a(p),q) + dist(a(px),qx) + dist(a(py),qy);
}

//
// Tests some invariants of class XYTransform
//
static void testXYTransform(const XYTransform &tr, const Point2D &p, 
                            bool uses_default_linearization, bool uses_exact_derivatives)
{
    Point2D tp = tr.forwardTransform(p);
    AffineTransform afwd = tr.linearizeForwardTransform(p);
    AffineTransform arev = tr.linearizeReverseTransform(tp);

    // round trip
    BOOST_CHECK(dist(p, tr.reverseTransform(tp)) < 1.0e-10);

    // affine transforms should map p -> t(p)
    BOOST_CHECK(dist(afwd(p),tp) < 1.0e-9);
    BOOST_CHECK(dist(arev(tp),p) < 1.0e-9);

    if (uses_default_linearization) {
        BOOST_CHECK(linearizationResidual(tr,p,1.0,true) < 1.0e-9);
        BOOST_CHECK(linearizationResidual(tr,tp,1.0,false) < 1.0e-9);
    }

    if (uses_exact_derivatives) {
        // forward/reverse transforms should be inverses
        BOOST_CHECK(dist(arev,afwd.invert()) < 1.0e-9);
    }

    if (uses_exact_derivatives) {
        //
        // The following sequence should be decreasing by a factor of ~100 each time 
        // (a factor ~10 would count as failure)
        //
        double t0 = linearizationResidual(tr, p, 1.0, true);
        double t1 = linearizationResidual(tr, p, 0.1, true);
        double t2 = linearizationResidual(tr, p, 0.01, true);
        double t3 = linearizationResidual(tr, p, 0.001, true);

        BOOST_CHECK(t1 < 0.04*t0);
        BOOST_CHECK(t2 < 0.04*t1);
        BOOST_CHECK(t3 < 0.04*t2);
    }

    PTR(XYTransform) tr_inv = tr.invert();

    // consistency of inverse linearized transforms
    BOOST_CHECK(dist(afwd, tr_inv->linearizeReverseTransform(p)) < 1.0e-9);
    BOOST_CHECK(dist(arev, tr_inv->linearizeForwardTransform(tp)) < 1.0e-9);
}


//
// General quadratic distortion of the form
//   x' = x + Ax + By + Cx^2 + Dxy + Ey^2
//   y' = y + Fx + Gy + Hx^2 + Ixy + Jy^2
//
class ToyXYTransform : public XYTransform
{
public:
    ToyXYTransform(double A, double B, double C, double D, double E, 
                   double F, double G, double H, double I, double J)
        : XYTransform(false), _A(A), _B(B), _C(C), _D(D), _E(E), _F(F), _G(G), _H(H), _I(I), _J(J)
    { }

    virtual ~ToyXYTransform() { }

    virtual PTR(XYTransform) clone() const
    {
        return PTR(XYTransform) (new ToyXYTransform(_A,_B,_C,_D,_E,_F,_G,_H,_I,_J));
    }

    virtual Point2D forwardTransform(Point2D const &pixel) const
    {
        double x = pixel.getX();
        double y = pixel.getY();
        
        return Point2D(x + _A*x + _B*y + _C*x*x + _D*x*y + _E*y*y,
                       y + _F*x + _G*y + _H*x*x + _I*x*y + _J*y*y);
    }

    virtual Point2D reverseTransform(Point2D const &pixel) const
    {
        static const int maxiter = 1000;
        Point2D ret = pixel;
        
        // very slow and boneheaded iteration scheme but OK for testing purposes
        for (int i = 0; i < maxiter; i++) {
            Point2D q = forwardTransform(ret);
            double dx = q.getX() - pixel.getX();
            double dy = q.getY() - pixel.getY();

#if 0
            cerr << "iteration " << i << ": (" << dx << "," << dy << ")\n";
#endif

            if (dx*dx + dy*dy < 1.0e-24)
                return ret;

            ret = Point2D(ret.getX() - dx, ret.getY() - dy);
        }

        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, 
                          "max iterations exceeded");
    }
    
    // factory function
    static shared_ptr<ToyXYTransform> makeRandom()
    {
        double A = 0.1 * (uni_double(rng)-0.5);
        double B = 0.1 * (uni_double(rng)-0.5);
        double C = 0.0001 * (uni_double(rng)-0.5);
        double D = 0.0001 * (uni_double(rng)-0.5);
        double E = 0.0001 * (uni_double(rng)-0.5);
        double F = 0.1 * (uni_double(rng)-0.5);
        double G = 0.1 * (uni_double(rng)-0.5);
        double H = 0.0001 * (uni_double(rng)-0.5);
        double I = 0.0001 * (uni_double(rng)-0.5);
        double J = 0.0001 * (uni_double(rng)-0.5);

        return PTR(ToyXYTransform) (new ToyXYTransform(A,B,C,D,E,F,G,H,I,J));
    }

protected:
    double _A, _B, _C, _D, _E, _F, _G, _H, _I, _J;
};


BOOST_AUTO_TEST_CASE(XYTransforms)
{
    cerr << "testing random ToyXYTransform...\n";
    PTR(XYTransform) t = ToyXYTransform::makeRandom();
    testXYTransform(*t, randpt(), true, false);

    cerr << "testing random RadialXYTransform...\n";
    t = makeRandomRadialXYTransform();
    testXYTransform(*t, randpt(), false, true);

    cerr << "testing random DetectorXYTransform...\n";
    PTR(Detector) d = ToyDetector::makeRandom();
    t = make_shared<DetectorXYTransform> (t,d);
    testXYTransform(*t, randpt(), false, true);
}


// -------------------------------------------------------------------------------------------------
//
// ToyPsf: general PDF of the form
//   exp(-ax^2/2 - bxy - cy^2/2)
//
// where
//   a = 0.1 (1 + Ax + By)
//   b = 0.1 (Cx + Dy)
//   c = 0.1 (1 + Ex + Fy)
//


//
// Helper function which fills an image with a normalized 2D Gaussian of the form
//   exp(-a(x-px)^2/2 - b(x-px)(y-py) - c(y-py)^2/2)
//
static PTR(Image<double>) fill_gaussian(double a, double b, double c, double px, double py, 
                                        int nx, int ny, int x0, int y0)
{
    // smallest eigenvalue
    double lambda = 0.5 * (a+c + sqrt((a-c)*(a-c) + b*b));

    // approximate size of box needed to hold kernel
    double width = sqrt(20/lambda);

    assert(lambda > 1.0e-10);
    assert(x0-px <= -width && x0-px+nx-1 >= width);
    assert(y0-py <= -width && y0-py+ny-1 >= width);

    PTR(Image<double>) im = make_shared<Image<double> >(nx, ny);
    im->setXY0(x0, y0);

    double imSum = 0.0;

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double x = i+x0-px;
            double y = j+y0-py;
            double t = exp(-0.5*a*x*x - b*x*y - 0.5*c*y*y);
            (*im)(i,j) = t;
            imSum += t;
        }
    }

    (*im) /= imSum;
    return im;
}


struct ToyPsf : public Psf
{    
    double _A, _B, _C, _D, _E, _F;

    ToyPsf(double A, double B, double C, double D, double E, double F)
        : _A(A), _B(B), _C(C), _D(D), _E(E), _F(F) 
    { }

    virtual ~ToyPsf() { }
    
    virtual PTR(Psf) clone() const 
    { 
        return make_shared<ToyPsf>(_A,_B,_C,_D,_E,_F); 
    }

    void evalABC(double &a, double &b, double &c, Point2D const &p) const
    {
        double x = p.getX();
        double y = p.getY();

        a = 0.1 * (1.0 + _A*x + _B*y);
        b = 0.1 * (_C*x + _D*y);
        c = 0.1 * (1.0 + _E*x + _F*y);
    }
    
    virtual PTR(Image) doComputeKernelImage(
        Color const &color, Point2D const &ccdXY, bool normalizePeak
    ) const {
        static const int nside = 100;

        double a, b, c;
        this->evalABC(a, b, c, ccdXY);

        return fill_gaussian(a, b, c, 0, 0, 2*nside+1, 2*nside+1, -nside, -nside);
    }
    
    // factory function
    static shared_ptr<ToyPsf> makeRandom()
    {
        double A = 0.005 * (uni_double(rng)-0.5);
        double B = 0.005 * (uni_double(rng)-0.5);
        double C = 0.005 * (uni_double(rng)-0.5);
        double D = 0.005 * (uni_double(rng)-0.5);
        double E = 0.005 * (uni_double(rng)-0.5);
        double F = 0.005 * (uni_double(rng)-0.5);

        return make_shared<ToyPsf> (A,B,C,D,E,F);
    }

};


BOOST_AUTO_TEST_CASE(warpedPsf)
{
    PTR(XYTransform) distortion = ToyXYTransform::makeRandom();

    PTR(ToyPsf) unwarped_psf = ToyPsf::makeRandom();
    PTR(WarpedPsf) warped_psf = make_shared<WarpedPsf> (unwarped_psf, distortion);

    Point2D p = randpt();
    Point2D q = distortion->reverseTransform(p);

    // warped image
    PTR(Image<double>) im = warped_psf->computeImage(p, false);
    int nx = im->getWidth();
    int ny = im->getHeight();
    int x0 = im->getX0();
    int y0 = im->getY0();

    double a, b, c;
    unwarped_psf->evalABC(a, b, c, q);

    Eigen::Matrix2d m0;
    m0 << a, b,
          b, c;
    
    AffineTransform atr = distortion->linearizeReverseTransform(p);

    Eigen::Matrix2d md;
    md << atr.getLinear()[0], atr.getLinear()[2],
          atr.getLinear()[1], atr.getLinear()[3];   // LinearTransform transposed index convention

    Eigen::Matrix2d m1 = md.transpose() * m0 * md;

    // this should be the same as the warped image, up to artifacts from warping/pixelization
    PTR(Image<double>) im2 = fill_gaussian(m1(0,0), m1(0,1), m1(1,1), 
                                           p.getX(), p.getY(), nx, ny, x0, y0);

    // TODO: improve this test; the ideal thing would be to repeat with 
    // finer resolutions and more stringent threshold
    BOOST_CHECK(compare(*im,*im2) < 0.005);
}


// -------------------------------------------------------------------------------------------------
//
// End-to-end test of XYTransform member functions for quadrupole distortion


static double contract(Quadrupole const &q, Extent2D const &x)
{
    Vector2d v;
    v[0] = x.getX();
    v[1] = x.getY();

    return v.transpose() * q.getMatrix().inverse() * v;
}

BOOST_AUTO_TEST_CASE(quadrupoleDistortion)
{
    PTR(XYTransform) t = ToyXYTransform::makeRandom();

    Point2D p = randpt();
    Quadrupole q(uni_double(rng)+1.0, uni_double(rng)+1.0, uni_double(rng));
    Quadrupole qfwd = t->forwardTransform(p,q);
    Quadrupole qrev = t->reverseTransform(p,q);
    
    Extent2D e(uni_double(rng)-0.5, uni_double(rng)-0.5);
    Extent2D efwd = t->linearizeForwardTransform(p)(e);
    Extent2D erev = t->linearizeReverseTransform(p)(e);

    BOOST_CHECK_CLOSE(contract(qfwd,efwd), contract(q,e), 1.0e-10);
    BOOST_CHECK_CLOSE(contract(qrev,erev), contract(q,e), 1.0e-10);
}
