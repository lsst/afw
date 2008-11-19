// -*- lsst-c++ -*-

%{
#include "lsst/afw/detection/Source.h"
#include "lsst/afw/formatters/Utils.h"
#include <sstream>
%}

%rename(SourceVec)  lsst::afw::detection::SourceVector;

SWIG_SHARED_PTR(Source, lsst::afw::detection::Source);
SWIG_SHARED_PTR_DERIVED(SourceVec, lsst::daf::base::Persistable, lsst::afw::detection::SourceVector);

%include "lsst/afw/detection/Source.h"
%include "lsst/afw/formatters/Utils.h"

// Provide semi-useful printing of catalog records
%extend lsst::afw::detection::Source {
    std::string toString() {
        std::ostringstream os;
        os << "Source " << $self->getId();
        os.precision(9);
        os << " (" << $self->getRa() << ", " << $self->getDec() << ")";
        return os.str();
    }
};

%pythoncode %{
Source.__str__ = Source.toString
%}

// SourceVector delegates rather than derives from std::vector, and applying
// SWIG std::vector machinery doesn't interact well with SWIG_SHARED_PTR.
// Therefore, wrap the simple parts of the SourceVector API. Support for
// python iteration and a subset of the python list functions is then added
// by hand.
namespace lsst {
namespace afw {
namespace detection {

class SourceVector : public lsst::daf::base::Persistable,
                     public lsst::daf::base::Citizen {
public:
    SourceVector();
    SourceVector(SourceVector const&);
    SourceVector(size_t size);
    SourceVector(size_t size, Source const& value);

    bool empty() const;
    size_t size() const;
    void clear();

    void swap(SourceVector& v);
    void pop_back();

    %rename(append) push_back;
    void push_back(Source const& x);

    void resize(size_t new_size);
    void resize(size_t new_size, Source const& x);

    void reserve(size_t n);
    size_t capacity() const;

    bool operator==(SourceVector const & v);
    bool operator!=(SourceVector const & v);
};

}}} // namespace lsst::afw::detection

%extend lsst::afw::detection::SourceVector {
    %newobject get(int);

    Source * get(int i) throw (std::out_of_range) {
        if (i < 0 || i >= static_cast<int>(self->size())) {
            throw std::out_of_range("SourceVec index out of range");
        }
        return new lsst::afw::detection::Source((*self)[i]);
    }

    Source & at(int i) {
        return (*self)[i];
    }

    void set(int i, Source const& s) throw (std::out_of_range) {
        if (i < 0 || i >= static_cast<int>(self->size())) {
            throw std::out_of_range("SourceVec index out of range");
        }
        (*self)[i] = s;
    }

    void insert(int i, Source const& s) throw (std::out_of_range) {
        if (i < 0 || i > static_cast<int>(self->size())) {
            throw std::out_of_range("SourceVec index out of range");
        }
        self->insert(self->begin() + i, s);
    }

    void erase(int i) throw (std::out_of_range) {
        if (i < 0 || i >= static_cast<int>(self->size())) {
            throw std::out_of_range("SourceVec index out of range");
        }
        self->erase(self->begin() + i);
    }

    void erase(int i, int j) throw (std::out_of_range) {
        int size = static_cast<int>(self->size());
        if (i < 0 || j < 0 || i >= size || j > size) {
            throw std::out_of_range("SourceVec index out of range");
        }
        self->erase(self->begin() + i, self->begin() + j);
    }
}

%pythoncode {

class _SourceVecIterator(object):
    def __init__(self, vec, index = 0):
        self.vec = vec
        self.index = index

    def next(self):
        if self.index == len(self.vec):
            raise StopIteration
        source = self.vec[self.index]
        self.index += 1
        return source

def _SourceVec___len__(vec):
    return vec.size()

def _SourceVec___iter__(vec):
    return _SourceVecIterator(vec)

def _SourceVec_pop(vec, index=None):
    if index is None:
        if vec.size() == 0:
            raise IndexError, "pop from empty SourceVec"
        s = vec.get(vec.size() - 1)
        vec.pop_back()
        return s
    elif isinstance(index, int):
        s = vec.get(index)
        vec.erase(index)
        return s
    else:
        raise TypeError, "SourceVec index not an int"

def _SourceVec___getitem__(vec, key):
    if isinstance(key, int):
        return vec.at(key)
    elif isinstance(key, slice):
        (start, stop, step) = key.indices(vec.size())
        v = SourceVec()
        while start < stop:
            v.append(vec.at(start))
            start += step
        return v
    else:
        raise TypeError, "SourceVec key not an int or slice"

def _SourceVec___setitem__(vec, key, value):
    if isinstance(key, int):
        vec.set(key, value)
    elif isinstance(key, slice):
        raise TypeError, "SourceVec slice assignment not implemented"
    else:
        raise TypeError, "SourceVec key not an int or slice"

def _SourceVec___delitem__(vec, key):
    if isinstance(key, int):
        vec.erase(key)
    elif isinstance(key, slice):
        (start, stop, step) = key.indices(vec.size())
        ndel = 0
        if step == 1:
            vec.erase(start, stop)
        else:
            while start < stop:
                vec.erase(start - ndel)
                start += step
                ndel += 1
    else:
        raise TypeError, "SourceVec key not an int or slice"

def _SourceVec___str__(vec):
    s = ["["]
    for i in xrange(vec.size()):
        if i != 0:
            s.append(",")
        s.append(str(vec.at(i)))
    s.append("]")
    return ''.join(s)

SourceVec.__len__     = _SourceVec___len__
SourceVec.__iter__    = _SourceVec___iter__
SourceVec.__getitem__ = _SourceVec___getitem__
SourceVec.__setitem__ = _SourceVec___setitem__
SourceVec.__delitem__ = _SourceVec___delitem__
SourceVec.__str__     = _SourceVec___str__
SourceVec.pop         = _SourceVec_pop

}

%lsst_persistable(lsst::afw::detection::SourceVector);

