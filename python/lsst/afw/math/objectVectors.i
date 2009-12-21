// -*- lsst-c++ -*-
/**
* Define vectors of objects (not POTS=plain old types) that are required for functions.

Call this last to avoid problems for overloaded constructors (or other functions, I assume).
If one of these lines appears before %importing the file declaring the type
and if the associated type is used as an argument in an overloaded constructor
that can alternatively be a vector of POTS, then:
- Calling the constructor with a python list containing a mix of the objects and types causes an abort
- SWIG will warn about a shadowed overloaded constructor (or function, presumably)
*/
%template(Function1FList) std::vector<boost::shared_ptr<lsst::afw::math::Function1<float> > >;
%template(Function1DList) std::vector<boost::shared_ptr<lsst::afw::math::Function1<double> > >;
%template(Function2FList) std::vector<boost::shared_ptr<lsst::afw::math::Function2<float> > >;
%template(Function2DList) std::vector<boost::shared_ptr<lsst::afw::math::Function2<double> > >;

%template(KernelList) std::vector<boost::shared_ptr<lsst::afw::math::Kernel> >;
