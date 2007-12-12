// -*- lsst-c++ -*-
%define fwCatalog_DOCSTRING
"
Access to persistable C++ objects for catalog data. Currently supported are:
    - DiaSource
    - MovingObjectPrediction
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.fw.Core", docstring=fwCatalog_DOCSTRING) fwCatalog

// Suppress swig complaints
// I had trouble getting %warnfilter to work; hence the pragmas
#pragma SWIG nowarn=314                 // print is a python keyword (--> _print)
#pragma SWIG nowarn=362                 // operator=  ignored

%{
#include "lsst/fw/DiaSource.h"
#include "lsst/fw/MovingObjectPrediction.h"
#include "lsst/fw/formatters/Utils.h"
%}

%inline %{
namespace boost { namespace filesystem {} }
%}

%init %{
%}

%pythoncode %{
import lsst.fw.exceptions
%}

%include "lsst/mwi/p_lsstSwig.i"
%include "lsst/mwi/persistenceMacros.i"
%import "lsst/mwi/persistence/Persistable.h"

%import "lsst/mwi/data/Citizen.h"
%import "lsst/mwi/policy/Policy.h"
%import "lsst/mwi/persistence/LogicalLocation.h"
%import "lsst/mwi/persistence/Persistence.h"
%import "lsst/mwi/persistence/Storage.h"
%import "lsst/mwi/data/DataProperty.h"


%include <stdint.i>
%include <std_vector.i>
%include <typemaps.i>

%rename(MopsPred)      lsst::fw::MovingObjectPrediction; 
%rename(MopsPredVec)   lsst::fw::MovingObjectPredictionVector;
%rename(DiaSourceVec)  lsst::fw::DiaSourceVector;

%include "lsst/fw/DiaSource.h"
%include "lsst/fw/MovingObjectPrediction.h"
%include "lsst/fw/formatters/Utils.h"


// Provide semi-useful printing of catalog records

%extend lsst::fw::DiaSource {
    std::string toString() {
        std::ostringstream os;
        os << "DiaSource " << $self->getId();
        os.precision(9);
        os << " (" << $self->getRa() << ", " << $self->getDec() << ")";
        return os.str();
    }
};

%extend lsst::fw::MovingObjectPrediction {
    std::string toString() {
        std::ostringstream os;
        os << "DiaSource " << $self->getId();
        os.precision(9);
        os << " (" << $self->getRa() << ", " << $self->getDec() << ")";
        return os.str();
    }
};

%pythoncode %{
MopsPred.__str__  = MopsPred.toString
DiaSource.__str__ = DiaSource.toString
%}

// Could define a single large SWIG macro to take care of all of this, but for now
// leave it as is for clarity and easy experimentation

%fragment("DiaSourceVectorTraits","header",fragment="StdSequenceTraits")
%{
namespace swig {
    template <>
    struct traits_asptr<lsst::fw::DiaSourceVector>  {
        static int asptr(PyObject *obj, lsst::fw::DiaSourceVector **vec) {
            return traits_asptr_stdseq<lsst::fw::DiaSourceVector>::asptr(obj, vec);
        }
    };
    
    template <>
    struct traits_from<lsst::fw::DiaSourceVector> {
        static PyObject *from(const lsst::fw::DiaSourceVector& vec) {
            return traits_from_stdseq<lsst::fw::DiaSourceVector>::from(vec);
        }
    };
}
%}

%fragment("MovingObjectPredictionVectorTraits","header",fragment="StdSequenceTraits")
%{
namespace swig {
    template <>
    struct traits_asptr<lsst::fw::MovingObjectPredictionVector>  {
        static int asptr(PyObject *obj, lsst::fw::MovingObjectPredictionVector **vec) {
            return traits_asptr_stdseq<lsst::fw::MovingObjectPredictionVector>::asptr(obj, vec);
        }
    };

    template <>
    struct traits_from<lsst::fw::MovingObjectPredictionVector> {
        static PyObject *from(const lsst::fw::MovingObjectPredictionVector& vec) {
            return traits_from_stdseq<lsst::fw::MovingObjectPredictionVector>::from(vec);
        }
    };
}
%}


// Define an analogue to %std_vector_methods()
// required since we have no access to std::vector::get_allocator()
%define %lsst_vector_methods(vec...)
  vec();
  vec(const vec&);
  vec(size_type size);
  vec(size_type size, const value_type& value);

  bool empty() const;
  size_type size() const;
  void clear();

  void swap(vec& v);

  #ifdef SWIG_EXPORT_ITERATOR_METHODS
  class iterator;
  class reverse_iterator;
  class const_iterator;
  class const_reverse_iterator;

  iterator begin();
  const_iterator begin() const;
  iterator end();
  const_iterator end() const;
  reverse_iterator rbegin();
  const_reverse_iterator rbegin() const;
  reverse_iterator rend();
  const_reverse_iterator rend() const;

  iterator erase(iterator pos);
  iterator erase(iterator first, iterator last);

  iterator insert(iterator pos, const value_type& x);
  void insert(iterator pos, size_type n, const value_type& x);
  #endif

  void pop_back();
  void push_back(const value_type& x);  

  const value_type& front() const;
  const value_type& back() const;
 
  void assign(size_type n, const value_type& x);
  void resize(size_type new_size);
  void resize(size_type new_size, const value_type& x);
 
  void reserve(size_type n);
  size_type capacity() const;

  bool operator==(vec const & v);
  bool operator!=(vec const & v);
%enddef


// Apply SWIG std::vector machinery to catalog classes. These classes contain a std::vector and look 
// like a std::vector, but do not derive from one (since std::vector has a non-virtual destructor).
namespace lsst {
namespace fw {

class DiaSourceVector : public lsst::mwi::persistence::Persistable {
public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef DiaSource value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef DiaSource& reference;
    typedef const DiaSource& const_reference;
    typedef std::allocator<lsst::fw::DiaSource> allocator_type;

    %traits_swigtype(lsst::fw::DiaSource);

    %fragment(SWIG_Traits_frag(lsst::fw::DiaSourceVector), "header",
              fragment=SWIG_Traits_frag(lsst::fw::DiaSource),
              fragment="DiaSourceVectorTraits") {
        namespace swig {
            template <>  struct traits<lsst::fw::DiaSourceVector> {
                typedef pointer_category category;
                static const char* type_name() {
                    return "lsst::fw::DiaSourceVector";
                }
            };
        }
    }

    %typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, lsst::fw::DiaSourceVector);

    #ifdef %swig_vector_methods
    // Add swig/language extra methods
    %swig_vector_methods(lsst::fw::DiaSourceVector);
    #endif

    %lsst_vector_methods(lsst::fw::DiaSourceVector);
};


class MovingObjectPredictionVector : public lsst::mwi::persistence::Persistable {
public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef MovingObjectPrediction value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef MovingObjectPrediction& reference;
    typedef const MovingObjectPrediction& const_reference;
    typedef std::allocator<MovingObjectPrediction> allocator_type;

    %traits_swigtype(lsst::fw::MovingObjectPrediction);

    %fragment(SWIG_Traits_frag(lsst::fw::MovingObjectPredictionVector), "header",
              fragment=SWIG_Traits_frag(lsst::fw::MovingObjectPrediction),
              fragment="MovingObjectPredictionVectorTraits") {
        namespace swig {
            template <>  struct traits<lsst::fw::MovingObjectPredictionVector> {
                typedef pointer_category category;
                static const char* type_name() {
                    return "lsst::fw::MovingObjectPredictionVector";
                }
            };
        }
    }

    %typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, lsst::fw::MovingObjectPredictionVector);

    #ifdef %swig_vector_methods
    // Add swig/language extra methods
    %swig_vector_methods(lsst::fw::MovingObjectPredictionVector);
    #endif

    %lsst_vector_methods(lsst::fw::MovingObjectPredictionVector);
};

}}

// Make sure SWIG generates type information for boost::shared_ptr<lsst::mwi::Persistable> *,
// even though that type is actually wrapped in the persistence module
%types(boost::shared_ptr<lsst::mwi::persistence::Persistable> *);

// Export instantiations of boost::shared_ptr for persistable data vectors
%lsst_persistable_shared_ptr(MopsPredVecSharedPtr, lsst::fw::MovingObjectPredictionVector);
%lsst_persistable_shared_ptr(DiaSourceVecSharedPtr, lsst::fw::DiaSourceVector);

%template(DiaSourceSharedPtr) boost::shared_ptr<lsst::fw::DiaSource>;

%pythoncode %{

def DiaSourceVecPtr(*args):
    """return a DiaSourceVecSharedPtr that owns its own DiaSourceVec"""
    v = DiaSourceVec(*args)
    v.this.disown()
    out = DiaSourceVecSharedPtr(v)
    return out

def MopsPredVecPtr(*args):
    """return a MopsPredVecSharedPtr that owns its own MopsPredVec"""
    v = MopsPredVec(*args)
    v.this.disown()
    out = MopsPredVecSharedPtr(v)
    return out

def DiaSourcePtr(*args):
    """return a DiaSourceSharedPtr that owns its own DiaSource"""
    ds = DiaSource(*args)
    ds.this.disown()
    out = DiaSourceSharedPtr(ds)
    return out

%}

