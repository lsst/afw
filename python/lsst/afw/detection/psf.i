// -*- lsst-C++ -*-

SWIG_SHARED_PTR_DERIVED(PSFPtrT, lsst::daf::data::LsstBase, lsst::meas::algorithms::PSF);
//
// We need this macro so as to avoid having commas in the 2nd argument to SWIG_SHARED_PTR_DERIVED,
// which confuses the swig parser.  It's also convenient
//
%define %MASKEDIMAGE(PIXTYPE)
lsst::afw::image::MaskedImage<PIXTYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>
%enddef
//
// Must go Before the %include
//
%define %PsfCandidatePtr(NAME, TYPE)
SWIG_SHARED_PTR_DERIVED(PsfCandidate##NAME,
                        lsst::afw::math::SpatialCellImageCandidate<%MASKEDIMAGE(TYPE)>,
                        lsst::meas::algorithms::PsfCandidate<%MASKEDIMAGE(TYPE)>);
/*
 * Swig doesn't like the TMP used to make makePsfCandidate able to deduce its image type, and thus be
 * easily usable from C++.  Here we define a simpler version for swig where we explicitly instantiate
 * the python version anyway.  Note that we %ignore the C++ version
 */
%inline %{
namespace lsst { namespace meas { namespace algorithms { namespace lsstSwig {
template <typename ImageT>
typename PsfCandidate<ImageT>::Ptr
makePsfCandidateForSwig(lsst::afw::detection::Source const& source, ///< The detected Source
                 typename ImageT::ConstPtr image ///< The image wherein lies the object
                ) {
    
    return typename PsfCandidate<ImageT>::Ptr(new PsfCandidate<ImageT>(source, image));
}
}}}}
%}

%ignore makePsfCandidate;
%enddef
//
// Must go After the %include
//
%define %PsfCandidate(NAME, TYPE)
%template(PsfCandidate##NAME) lsst::meas::algorithms::PsfCandidate<%MASKEDIMAGE(TYPE)>;
%template(makePsfCandidate) lsst::meas::algorithms::lsstSwig::makePsfCandidateForSwig<%MASKEDIMAGE(TYPE)>;
//
// When swig sees a SpatialCellImageCandidates it doesn't know about PsfCandidates; all it knows is that it has
// a SpatialCellImageCandidate, and SpatialCellCandidates don't know about e.g. getSource().
//
// We therefore provide a cast to PsfCandidate<TYPE> and swig can go from there;  In fact,
// we can cast all the way from the ultimate base class, so let's do that.
//
%inline %{
    lsst::meas::algorithms::PsfCandidate<%MASKEDIMAGE(TYPE)>::Ptr
        cast_PsfCandidate##NAME(lsst::afw::math::SpatialCellCandidate::Ptr candidate) {
        return boost::shared_dynamic_cast<lsst::meas::algorithms::PsfCandidate<%MASKEDIMAGE(TYPE)> >(candidate);
    }
%}
%enddef

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

%PsfCandidatePtr(F, float);

%ignore PsfFactoryBase;

%include "lsst/meas/algorithms/PSF.h"
%include "lsst/meas/algorithms/SpatialModelPsf.h"
//
// N.b. Swig won't will be able to resolve the overload for *FromPsfCandidates
// if you define another image type (there are no dependent parameters); so you'll have to
// append some type marker (e.g. "I") to the name
//
%template(pair_Psf_vector_double) std::pair<lsst::afw::math::LinearCombinationKernel::Ptr, std::vector<double> >;
%template(pair_bool_double) std::pair<bool, double>;

%PsfCandidate(F, float);
%template(createKernelFromPsfCandidates) lsst::meas::algorithms::createKernelFromPsfCandidates<float>;
%template(fitSpatialKernelFromPsfCandidates) lsst::meas::algorithms::fitSpatialKernelFromPsfCandidates<float>;
%template(subtractPsf) lsst::meas::algorithms::subtractPsf<%MASKEDIMAGE(float)>;
