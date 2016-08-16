// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
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

class PyPixelRegionIterator {
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

    // No copying
    PyPixelRegionIterator (const PyPixelRegionIterator&) = delete;
    PyPixelRegionIterator& operator=(const PyPixelRegionIterator&) = delete;

    // No moving
    PyPixelRegionIterator (PyPixelRegionIterator&&) = delete;
    PyPixelRegionIterator& operator=(PyPixelRegionIterator&&) = delete;

private:
    lsst::afw::geom::ellipses::PixelRegion::Iterator _current;
    lsst::afw::geom::ellipses::PixelRegion::Iterator _end;
    PyObject * _owner;
};
