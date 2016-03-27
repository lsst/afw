// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

// This is a bit of swig-only C++ code that ideally we'd put in a %inline block,
// but it turns out %inline blocks don't get pulled in by %import statements,
// causing problems downstream problems in any module that %imports ellipsesLib.i
// or PixelRegion.i.  So instead, we separate it into a .h file that must be
// #included inside a %{ %} block in any module that includes ellipsesLib.i.
// Still ugly, but at least it makes it possible for downstream modules to include
// ellipsesLib.i.  If someone has a better solution, I'd love to hear about it.
// Note that we also have to put it in include/ rather than python/, because we
// need it to be in the include path when we compile SWIG modules, not just when
// we generate them.

class PyPixelRegionIterator : private boost::noncopyable {
public:

    lsst::afw::geom::Span get() const { return *_current; }

    void increment() { ++_current; }

    bool atEnd() const { return _current == _end; }

    PyPixelRegionIterator(
        lsst::afw::geom::ellipses::PixelRegion::Iterator const & begin,
        lsst::afw::geom::ellipses::PixelRegion::Iterator const & end,
        PyObject * owner = NULL
    ) : _current(begin), _end(end), _owner(owner) {
        Py_XINCREF(_owner);
    }

    ~PyPixelRegionIterator() { Py_XDECREF(_owner); }

private:
    lsst::afw::geom::ellipses::PixelRegion::Iterator _current;
    lsst::afw::geom::ellipses::PixelRegion::Iterator _end;
    PyObject * _owner;
};
