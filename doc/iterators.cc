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

/**
 * \file
 * \brief A set of pseudo-code to permit me to document the Image iterator API.
 *
 * This is needed as the actual implementation is opaque, based on the boost::gil
 * libraries.
 */
namespace lsst { namespace afw { namespace image {
/// An ImageBase iterator
class imageIterator {
public:
    /// Dereference an iterator, returning a reference to a Pixel
    Image::Pixel& operator*();
    /// Advance an iterator (prefix)
    void operator++();
    /// Advance an iterator (postfix)
    void operator++(int);
    /// Increment the iterator by \c delta
    void operator+=(std::ptrdiff_t delta ///< how far to move the iterator
                   );
    /// Decrement the iterator by \c delta
    void operator-=(std::ptrdiff_t delta ///< how far to move the iterator
                   );
    /// Return true if the lhs equals the rhs
    bool operator==(imageIterator const& rhs ///< right hand side
                   );
    /// Return true if the lhs doesn't equal the rhs
    bool operator!=(imageIterator const& rhs ///< right hand side
                   );
    /// Return true if the lhs is less than the rhs
    bool operator<(imageIterator const& rhs); ///< right hand side
};

/// An ImageBase locator
class imageLocator {
public:
    /// type to store relative location for efficient repeated access (not really void)
    typedef void cached_location_t;
    /// Dereference a locator, returning a reference to a Pixel
    Image::Pixel& operator*();
    /// Dereference a %pixel offset by <tt>(x, y)</tt> from the current locator, returning a reference to a Pixel
    ///
    /// As an \c locator ptr is moved through the image, expressions such as <tt>ptr(-1, -1)</tt> can
    /// be used to examine or set neighbouring pixels
    Image::Pixel& operator()(int x, int y);
    /// Return an \c x_iterator that can move an \c xy_locator
    xy_x_iterator x();
    /// Return an \c y_iterator that can move an \c xy_locator
    xy_y_iterator y();

    /// An \c x_iterator created from an \c xy_locator
    struct xy_x_iterator {
        /// Dereference an iterator, returning a reference to a Pixel
        Image::Pixel& operator*();
        /// Advance the iterator (prefix)
        void operator++();
        /// Advance the iterator (postfix)
        void operator++(int);
    };

    /// An \c y_iterator created from an \c xy_locator
    struct xy_y_iterator {
        /// Dereference an iterator, returning a reference to a Pixel
        Image::Pixel& operator*();
        /// Advance the iterator (prefix)
        void operator++();
        /// Advance the iterator (postfix)
        void operator++(int);
    };

    /// Store a relative location for faster repeated access
    cached_location_t cache_location(int x, int y);
    /// Dereference a \c cached_location_t, returning a reference to a Pixel
    Image::Pixel& operator[](cached_location_t const&);
};

}}}
