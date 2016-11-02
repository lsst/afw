// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See COPYRIGHT file at the top of the source tree.
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program. If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#ifndef LSST_AFW_GEOM_POINT2DLIST_H_
#define LSST_AFW_GEOM_POINT2DLIST_H_

#include <iterator>
#include <type_traits>
#include <vector>

#include "ndarray_fwd.h"

#include "lsst/afw/geom/Point.h"

// Temporary, remove when all template methods implemented
#include <stdexcept>

namespace lsst {
namespace afw {
namespace geom {

/**
 * Mutable list of Points, with support for C-style updates.
 *
 * This class allows multiple Point objects to be passed to and from
 * algorithms. It behaves much like `std::vector`, and shall be
 * compatible with any generic code that accepts STL containers, except
 * that it does not support custom allocators.
 *
 * `Point2DList` provides support (through the data() method) for
 * treating points as arrays of floating-point values. Do not intersperse
 * calls to data() or its return value and calls to other modifying
 * operations like insert(), as doing so may lead to data corruption.
 *
 * @note This class does not follow LSST conventions for the names of its
 * member types or methods in order to maximize compatibility with generic
 * code written for STL containers.
 */
class Point2DList {
public:
    /*
     * Member types
     */

    /**
     * Iterates over the collection without allowing element modification.
     *
     * This class shall be a random access iterator. Behavior when
     * default-constructed, when moving before the first element, or
     * when moving after the past-the-end element is **undefined**.
     */
    class const_iterator : public std::iterator<std::random_access_iterator_tag, Point2D> {
    public:
        /// Reference to an unmodifiable element
        using const_reference =
                std::add_lvalue_reference<std::add_const<std::remove_reference<reference>::type>::type>;
        /// Modifiable pointer to an unmodifiable element
        using const_pointer = std::add_pointer<std::add_const<std::remove_pointer<pointer>::type>::type>;

        /*
         * Non-throwing constructors
         */

        const_iterator() noexcept;
        const_iterator(const_iterator const &) noexcept;
        const_iterator &operator=(const_iterator const &) noexcept;

        /*
         * Iteration
         */
        const_iterator &operator++();
        const_iterator operator++(int);
        const_iterator &operator+=(difference_type);
        const_iterator operator+(difference_type) const;
        const_iterator &operator--();
        const_iterator operator--(int);
        const_iterator &operator-=(difference_type);
        const_iterator operator-(difference_type) const;
        difference_type operator-(const const_iterator &) const;

        /*
         * Element access
         */

        value_type operator*() const;
        const_pointer operator->() const;
        value_type operator[](difference_type) const;

        /*
         * Comparisons
         */

        bool operator==(const const_iterator &) const;
        bool operator!=(const const_iterator &) const;
        bool operator<(const const_iterator &) const;
        bool operator>(const const_iterator &) const;
        bool operator<=(const const_iterator &) const;
        bool operator>=(const const_iterator &) const;
    };

    /**
     * Iterates over the collection, allowing element modification.
     *
     * @copydetails const_iterator
     */
    class iterator : public const_iterator {
    public:
        /*
         * Non-throwing constructors
         */

        iterator() noexcept;
        iterator(const_iterator const &) noexcept;
        iterator &operator=(const_iterator const &) noexcept;

        /*
         * Iteration
         */

        // Deliberately masking, not inheriting -- return type determined by variable's declared type
        iterator &operator++();
        iterator operator++(int);
        iterator &operator+=(difference_type);
        iterator operator+(difference_type) const;
        iterator &operator--();
        iterator operator--(int);
        iterator &operator-=(difference_type);
        iterator operator-(difference_type) const;

        /*
         * Element access
         */

        reference operator*() const;
        pointer operator->() const;
        reference operator[](difference_type) const;
    };

    /// Type of element stored in this container.
    using value_type = std::iterator_traits<iterator>::value_type;

    /// Allocator used by this container.
    using allocator_type = std::allocator<value_type>;

    /// Reference to modifiable elements in this container.
    using reference = std::iterator_traits<iterator>::reference;

    /// Reference to unmodifiable elements in this container.
    // TODO: replace with reference once safe proxy available
    // using const_reference = const std::iterator_traits<iterator>::reference;
    using const_reference = value_type;

    /// Pointer to modifiable elements in this container.
    using pointer = std::allocator_traits<allocator_type>::pointer;

    /// Pointer to unmodifiable elements in this container.
    using const_pointer = std::allocator_traits<allocator_type>::const_pointer;

    /// Random-access iterator to value_type.
    using reverse_iterator = std::reverse_iterator<iterator>;

    /// Random-access iterator to `const value_type`.
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    /// Difference between two iterators. Shall be a signed integer.
    using difference_type = std::iterator_traits<iterator>::difference_type;

    /**
     * Indices, lengths, and non-negative values of difference_type. Shall be an unsigned integer.
     */
    using size_type = std::size_t;

    /*
     * List definition
     */

    /**
     * Construct an empty list.
     *
     * @throws MemoryError Thrown if the list could not prepare an element
     *                     buffer.
     * @exceptsafe Provides strong exception guarantee.
     */
    explicit Point2DList();

    /**
     * Construct a list of points whose coordinates are all zero.
     *
     * @param n the initial list size.
     *
     * @throws LengthError Thrown if the list implementation cannot handle
     *                     `n` elements even if sufficient memory is
     *                     available.
     * @throws MemoryError Thrown if the list could not allocate space for
     *                     the points.
     * @exceptsafe Provides strong exception guarantee.
     */
    explicit Point2DList(size_type n);

    /**
     * Construct a list of identical points.
     *
     * @param n the initial list size.
     * @param point the value for each element
     *
     * @throws LengthError Thrown if the list implementation cannot handle
     *                     `n` elements even if sufficient memory is
     *                     available.
     * @throws MemoryError Thrown if the list could not allocate space for
     *                     the points.
     * @exceptsafe Provides strong exception guarantee.
     */
    Point2DList(size_type n, value_type const &point);

    /**
     * Copy a list from another container.
     *
     * The list shall contain Points equivalent to the data in
     * `[first, last)`, in the same order. If the iterator range
     * `[first, last)` is not valid, this constructor shall have
     * **undefined** behavior.
     *
     * @tparam InputIterator an input iterator pointing to any type from
     *                       which a Point can be constructed.
     *
     * @param first an iterator pointing to the first element to be copied
     *              to this list
     * @param last an iterator pointing immediately after the last element
     *             to be copied to this list
     *
     * @throws LengthError Thrown if the list implementation cannot handle
     *                     `std::distance(first, last)` elements even if
     *                     sufficient memory is available.
     * @throws MemoryError Thrown if the list could not allocate space for
     *                     the points.
     * @exceptsafe Provides strong exception guarantee.
     */
    template <class InputIterator>
    Point2DList(InputIterator first, InputIterator last);

    /**
     * Copy a list from another.
     *
     * @param other the list to be copied
     *
     * @throws MemoryError Thrown if the list could not allocate space for
     *                     the points.
     * @exceptsafe Provides strong exception guarantee.
     */
    Point2DList(Point2DList const &other);

    /**
     * Move-construct a list.
     *
     * The list from which elements are being moved shall be left in an
     * unspecified but valid state. All iterators, pointers, and references
     * related to the old list shall be invalidated.
     *
     * @param other the list to be moved.
     *
     * @exceptsafe Provides strong exception guarantee.
     */
    Point2DList(Point2DList &&other);

    /**
     * Copy a list from an initializer list.
     *
     * @param values a list of values to include in the list
     *
     * @throws LengthError Thrown if the list implementation cannot handle
     *                     `values.size()` elements even if sufficient
     *                     memory is available.
     * @throws MemoryError Thrown if the list could not allocate space for
     *                     the points.
     * @exceptsafe Provides strong exception guarantee.
     */
    Point2DList(std::initializer_list<value_type> values);

    /**
     * Destroy the list and all its elements.
     *
     * @exceptsafe Shall not throw exceptions.
     */
    ~Point2DList();

    /*
     * List properties
     */

    /**
     * `true` if the list is empty.
     *
     * @return `true` if the list has no elements, `false` otherwise.
     *
     * @exceptsafe Shall not throw exceptions.
     */
    bool empty() const noexcept;

    /**
     * Number of elements in the list
     *
     * This is the number of actual points in the list, not its
     * capacity(), which may be larger.
     *
     * @return the number of points currently stored in the list
     *
     * @exceptsafe Shall not throw exceptions.
     */
    size_type size() const noexcept;

    /**
     * Maximum size allowed for a `Point2DList`.
     *
     * This is the maximum size a list can reach because of implementation
     * limitations; it may be compiler- or OS-dependent.
     *
     * @return the maximum number of elements that may be stored in the list,
     *         assuming unlimited memory
     *
     * @exceptsafe Shall not throw exceptions.
     */
    size_type max_size() const noexcept;

    /*
     * Iteration
     */

    /**
     * Iterator pointing to the first element in the list.
     *
     * If the list is empty, the returned iterator must not be
     * dereferenced.
     *
     * @return An iterator to the beginning of the list.
     *
     * @exceptsafe This method shall not throw exceptions.
     *             Copy-construction or assignment of the returned
     *             iterator shall also not throw exceptions.
     *
     * @see front()
     */
    iterator begin() noexcept;

    /// @copydoc begin()
    const_iterator begin() const noexcept;

    /**
     * Iterator pointing to past the last element in the list.
     *
     * Since this iterator does not point to any element, it must not be
     * dereferenced.
     *
     * @return An iterator to the past-the-end element of the list.
     *
     * @exceptsafe This method shall not throw exceptions.
     *             Copy-construction or assignment of the returned
     *             iterator shall also not throw exceptions.
     */
    iterator end() noexcept;

    /// @copydoc end()
    const_iterator end() const noexcept;

    /**
     * Reverse iterator pointing to the last element in the list.
     *
     * If the list is empty, the returned iterator must not be
     * dereferenced.
     *
     * @return A reverse iterator to the reverse beginning of the list.
     *
     * @exceptsafe This method shall not throw exceptions.
     *             Copy-construction or assignment of the returned
     *             iterator shall also not throw exceptions.
     *
     * @see back()
     */
    reverse_iterator rbegin() noexcept;

    /// @copydoc rbegin()
    const_reverse_iterator rbegin() const noexcept;

    /**
     * Reverse iterator pointing to past the first element in the list.
     *
     * Since this iterator does not point to any element, it must not be
     * dereferenced.
     *
     * @return An iterator to the past-the-beginning element of the list.
     *
     * @exceptsafe This method shall not throw exceptions.
     *             Copy-construction or assignment of the returned
     *             iterator shall also not throw exceptions.
     */
    reverse_iterator rend() noexcept;

    /// @copydoc rend()
    const_reverse_iterator rend() const noexcept;

    /**
     * @copybrief begin()
     *
     * Like begin(), but always returns a `const_iterator`, even if
     * the list is not `const`.
     */
    const_iterator cbegin() const noexcept;

    /**
     * @copybrief end()
     *
     * Like end(), but always returns a `const_iterator`, even if
     * the list is not `const`.
     */
    const_iterator cend() const noexcept;

    /**
     * @copybrief rbegin()
     *
     * Like rbegin(), but always returns a `const_iterator`, even if
     * the list is not `const`.
     */
    const_reverse_iterator crbegin() const noexcept;

    /**
     * @copybrief rend()
     *
     * Like rend(), but always returns a `const_iterator`, even if
     * the list is not `const`.
     */
    const_reverse_iterator crend() const noexcept;

    /*
     * Element access
     */

    /**
     * Reference to the first element in the list.
     *
     * @return a reference to the first element
     *
     * @throws DomainError Thrown if this list is empty.
     * @exceptsafe Provides strong exception guarantee.
     *
     * @see begin()
     */
    reference front();

    /// @copydoc front()
    const_reference front() const;

    /**
     * Reference to the last element in the list.
     *
     * @return a reference to the last element
     *
     * @throws DomainError Thrown if this list is empty.
     * @exceptsafe Provides strong exception guarantee.
     *
     * @see rbegin()
     */
    reference back();

    /// @copydoc back()
    const_reference back() const;

    /**
     * Reference to a specific element.
     *
     * Behavior is **undefined** if the index is out of range. Use
     * at(size_type) if you need bounds checking.
     *
     * @param index the position of the desired element. Must be in the
     *          interval `[0, size())`.
     * @return a reference to the `index`th element of the list
     *
     * @exceptsafe Provides strong exception guarantee.
     */
    reference operator[](size_type index);

    /// @copydoc operator[](size_type)
    const_reference operator[](size_type index) const;

    /**
     * Reference to a specific element.
     *
     * Unlike the [] operator, this method checks that the index refers to
     * a real element.
     *
     * @param index the position of the desired element. Must be in the
     *          interval `[0, size())`.
     * @return a reference to the `index`th element of the list
     *
     * @throws OutOfRangeError Thrown if `index` &ge; `size()`.
     * @exceptsafe Provides strong exception guarantee.
     */
    reference at(size_type index);

    /// @copydoc at(size_type)
    const_reference at(size_type index) const;

    /*
     * Element-by-element modification
     */

    /**
     * Insert a new value at a specific position.
     *
     * The list shall be extended by inserting a copy of the new point
     * before the element at `position`. Inserting a point is not
     * guaranteed to be efficient except at the end of the list (see
     * push_back(value_type const &)).
     *
     * If the list's size grows beyond capacity(), then all iterators,
     * pointers, and references related to the list shall be invalidated.
     * Otherwise, all iterators, pointers, and references referring to
     * elements before `position` shall point to the same elements as
     * before.
     *
     * @param position Position in the list where the new point will be
     *                 inserted. The point previously at this position, if
     *                 any, shall be moved to the position after the new
     *                 point. If `position` is invalid, the result is
     *                 **undefined** behavior.
     * @param value The point to be copied into the list.
     * @return an iterator pointing to the newly inserted point.
     *
     * @throws LengthError Thrown if the list would grow larger than
     *                     max_size()
     * @throws MemoryError Thrown if the list could not allocate space for
     *                     the new point.
     * @exceptsafe Provides strong exception guarantee if the new value is
     *             being inserted at the end() of the list, and the basic
     *             guarantee otherwise.
     */
    iterator insert(const_iterator position, value_type const &value);

    /**
     * Move-construct a value at a specific position.
     *
     * Like insert(const_iterator, value_type const &), except that
     * the new value shall be move-constructed, not copy-constructed.
     */
    iterator insert(const_iterator position, value_type &&value);

    /**
     * Insert multiple copies of a value at a specific position.
     *
     * Like insert(const_iterator, value_type const &), except that
     * the list shall be extended by `n` copies of the new point. Despite
     * inserting multiple points, this method provides the atomic exception
     * guarantee for insertions to end().
     *
     * @param position Position in the list where the new points will be
     *                 inserted. The point previously at this position
     *                 shall be moved forward `n` spaces. If `position` is
     *                 invalid, the result is **undefined** behavior.
     * @param n The number of copies to make of `value`.
     * @param value The point to be copied into the list.
     * @return an iterator pointing to the first of the newly inserted
     *         points.
     */
    iterator insert(const_iterator position, size_type n, value_type const &value);

    /**
     * Insert values from another container at a specific position.
     *
     * Like insert(const_iterator, value_type const &), except that
     * the list shall be extended by inserting copies of the points in
     * `[first, last)` before the element at `position`. Despite inserting
     * multiple points, this method provides the atomic exception guarantee
     * for insertions to end().
     *
     * @tparam InputIterator an input iterator pointing to any type from
     *                       which a Point can be constructed.
     *
     * @param position Position in the list where the new points will be
     *                 inserted. The point previously at this position
     *                 shall be moved forward `std::distance(first, last)`
     *                 spaces. If `position` is invalid, the result is
     *                 **undefined** behavior.
     * @param first an iterator pointing to the first element to be copied
     *              to this list
     * @param last an iterator pointing immediately after the last element
     *             to be copied to this list
     * @return an iterator pointing to the first of the newly inserted
     *         points.
     */
    template <class InputIterator>
    iterator insert(const_iterator position, InputIterator first, InputIterator last);

    /**
     * Insert values from an initializer list at a specific position.
     *
     * Like insert(const_iterator, value_type const &), except that
     * the list shall be extended by inserting copies of the points in
     * `values`. Despite inserting multiple points, this method provides
     * the atomic exception guarantee for insertions to end().
     *
     * @param position Position in the list where the new point will be
     *                 inserted. The point previously at this position, if
     *                 any, shall be moved forward `values.size()` spaces.
     *                 If `position` is invalid, the result is **undefined**
     *                 behavior.
     * @param values a list of points to copy into the list
     * @return an iterator pointing to the first of the newly inserted
     *         points.
     */
    iterator insert(const_iterator position, std::initializer_list<value_type> values);

    /**
     * Construct and insert a new Point at a specific position.
     *
     * Like insert(const_iterator, value_type const &), except that
     * the new point shall be constructed in place using `args` as the
     * arguments for its construction.
     *
     * @param position Position in the list where the new point will be
     *                 inserted. The point previously at this position
     *                 shall be moved forward one space. If `position`
     *                 is invalid, the result is **undefined** behavior.
     * @param args the arguments to a Point constructor
     * @return an iterator pointing to the newly inserted point.
     *
     * @exceptsafe As for insert(const_iterator, value_type const &), except
     *             that Point's constructors must provide at least basic
     *             exception safety for this method to provide any guarantee.
     *
     * @see emplace_back()
     */
    template <class... Args>
    iterator emplace(const_iterator position, Args &&... args);

    /**
     * Remove a specific element.
     *
     * Removing a point is not guaranteed to be efficient except at the end
     * of the list (see pop_back()).
     *
     * All iterators, pointers, and references referring to the deleted
     * element or later elements shall be invalidated.
     *
     * @param position An iterator pointing to the element to be removed.
     *                 If `position` is invalid, the result is
     *                 **undefined** behavior.
     * @return an iterator pointing to the element that previously followed
     *         `position`.
     *
     * @exceptsafe If the last element in the list is being removed, this
     *             method shall not throw exceptions. Otherwise, provides
     *             basic exception safety.
     */
    iterator erase(const_iterator position);

    /**
     * Remove a specific range of elements.
     *
     * As erase(const_iterator), except that all elements in the range
     * `[first, last)` shall be removed. Behavior if `[first, last)` is not
     * a valid range is **undefined**.
     *
     * @param first an iterator pointing to the first element to be removed
     * @param last an iterator pointing immediately after the last element
     *             to be removed
     * @return an iterator pointing to the element that previously followed
     *         the last element to be removed
     */
    iterator erase(const_iterator first, const_iterator last);

    /**
     * Append a new value to the list.
     *
     * This method is equivalent to
     * insert(const_iterator, value_type const &) "insert(end(), value)",
     * except that it does not return an iterator to the new element.
     */
    void push_back(value_type const &value);

    /**
     * Move-construct a value at the end of the list.
     *
     * This method is equivalent to
     * insert(const_iterator, value_type &&) "insert(end(), value)",
     * except that it does not return an iterator to the new element.
     */
    void push_back(value_type &&value);

    /**
     * Construct and append a new Point to the end of the list.
     *
     * This method is equivalent to
     * emplace() "emplace(end(), args)", except that it does not
     * return an iterator to the new element.
     */
    template <class... Args>
    void emplace_back(Args &&... args);

    /**
     * Remove the last element.
     *
     * All iterators, pointers, and references referring to the last
     * element or end() shall be invalidated.
     *
     * @throws DomainError Thrown if this list is empty.
     * @exceptsafe Provides strong exception guarantee.
     */
    void pop_back();

    /*
     * Bulk modification
     */

    /**
     * Exchange the content of this list with another.
     *
     * All elements previously in `other` shall be in this list, and vice
     * versa. All iterators, references, and pointers referring to either
     * list (except end()) shall remain valid, but shall switch list
     * associations.
     *
     * @param other a list whose elements are swapped with those in
     *              this object
     *
     * @exceptsafe Shall not throw exceptions.
     */
    void swap(Point2DList &other) noexcept;

    /**
     * Copy a list from another.
     *
     * The current content of this list shall be replaced with copies of
     * the elements of `other`. All iterators, references, and pointers
     * relating to this list shall be invalidated. Points previously in
     * this list may be assigned to or destroyed.
     *
     * @param other the list to be copied over this one
     * @return a reference to this list
     *
     * @throws MemoryError Thrown if the list could not allocate space for
     *                     the incoming points.
     * @exceptsafe Provides basic exception guarantee.
     */
    Point2DList &operator=(Point2DList const &other);

    /**
     * Move-assign a list from another.
     *
     * As operator=(Point2DList const &), except that the list elements shall
     * be move-assigned. `other` shall be left in an undefined but valid state.
     */
    Point2DList &operator=(Point2DList &&other);

    /**
     * Overwrite a list with new values
     *
     * As operator=(Point2DList const &), but taking an initializer list.
     *
     * @param values the points to copy into this list
     *
     * @throws LengthError Thrown if `values.size()` > max_size()
     * @throws MemoryError Thrown if the list could not allocate space for
     *                     the incoming points.
     */
    Point2DList &operator=(std::initializer_list<value_type> values);

    /**
     * Copy a list from another container.
     *
     * The current content of this list shall be replaced with copies of
     * the elements in `[first, last)`. All iterators, references, and pointers
     * relating to this list shall be invalidated. Points previously in
     * this list shall be destroyed. If the iterator range `[first, last)`
     * is not valid, this method shall have **undefined** behavior.
     *
     * @tparam InputIterator an input iterator pointing to any type from
     *                       which a Point can be constructed.
     *
     * @param first an iterator pointing to the first element to be copied
     *              to this list
     * @param last an iterator pointing immediately after the last element
     *             to be copied to this list
     *
     * @throws LengthError Thrown if the operation would create a list
     *                     longer than max_size().
     * @throws MemoryError Thrown if the list could not allocate space for
     *                     the points.
     * @exceptsafe Provides basic exception guarantee.
     */
    template <class InputIterator>
    void assign(InputIterator first, InputIterator last);

    /**
     * Overwrite this list with copies of a point.
     *
     * As assign(InputIterator, InputIterator), except that the list
     * contents shall be replaced with `n` copies of `point`.
     *
     * @param n the number of copies to leave
     * @param point the value for each element
     */
    void assign(size_type n, value_type const &point);

    /**
     * Overwrite a list with new values
     *
     * As assign(InputIterator, InputIterator), except that the list
     * contents shall be replaced with copies of the listed points.
     */
    void assign(std::initializer_list<value_type> values);

    /**
     * Force the list to contain an exact number of points.
     *
     * If `n` is smaller than the current size(), the list shall be
     * truncated to its first `n` elements. If `n` is larger than the
     * current size, the list shall be padded with points whose coordinates
     * are all 0.
     *
     * If the list is truncated, all iterators, pointers, and references to
     * the first `n` elements shall point to the same elements as before,
     * while all others shall be invalidated. If it is extended to a size
     * exceeding capacity(), all iterators, pointers, and references
     * shall be invalidated. Otherwise, only end() shall be
     * invalidated.
     *
     * @param n the number of elements desired
     *
     * @throws LengthError Thrown if `n` > max_size()
     * @throws MemoryError Thrown if the list could not allocate space for
     *                     the extra points.
     * @exceptsafe Shall not throw exceptions if this list is being truncated.
     *             Otherwise, provides strong exception safety.
     */
    void resize(size_type n);

    /**
     * @copybrief resize(size_type)
     *
     * As resize(size_type), except that the list shall be padded with
     * copies of `value` if `n` is larger than size().
     *
     * @param n the number of elements desired
     * @param value the value for each new element
     */
    void resize(size_type n, value_type const &value);

    /**
     * Remove all elements from this list.
     *
     * @exceptsafe Shall not throw exceptions.
     */
    void clear() noexcept;

    /*
     * Memory management
     */

    /**
     * Amount of space currently allocated for new elements.
     *
     * The capacity is not the same as the size(), and shall always be
     * greater than or equal to the size. Attempts to grow the list above
     * its current capacity will force it to reallocate memory.
     *
     * @return the number of elements for which memory is currently
     *         allocated
     *
     * @exceptsafe Shall not throw exceptions.
     */
    size_type capacity() const noexcept;

    /**
     * Ensure the list can store a specific number of elements.
     *
     * The list shall reallocate memory if and only if it is necessary to
     * get storage for `n` elements. It shall take no action if
     * `n` &le; capacity().
     *
     * If a reallocation happens, then all iterators, pointers, and
     * references related to the list shall be invalidated.
     *
     * @param n The minimum capacity for the list. The list implementation
     *          may allocate memory for more than `n` elements.
     *
     * @throws LengthError Thrown if `n` > max_size()
     * @throws MemoryError Thrown if the list could not allocate space for
     *                     `n` points.
     * @exceptsafe Provides strong exception guarantee.
     */
    void reserve(size_type n);

    /**
     * Request that the list reduce its memory usage.
     *
     * While calls to this method will typically reallocate memory and
     * reduce the list's capacity(), implementations are not required
     * to set the capacity to the list's size().
     *
     * If a reallocation happens, then all iterators, pointers, and
     * references related to the list shall be invalidated.
     *
     * @throws MemoryError Thrown if the list could not allocate the
     *                     reduced buffer.
     * @exceptsafe Provides strong exception guarantee.
     */
    void shrink_to_fit();

    /**
     * View of this list as a multidimensional array of coordinates.
     *
     * A slice of the form `list.data()[view(i)()]` shall represent the
     * `i`th point in this list, with coordinates given in the same order
     * as by Point. For example, `list.data()[i][j]` shall be
     * equivalent to `list.at(i)[j]` for valid indices `i` and `j`.
     *
     * The array returned by this method shall allow element modification,
     * which shall be propagated back to this list, but shall not allow
     * changes to its dimensions. In particular, it is not possible to add
     * or remove points from this list using the array view.
     *
     * @return an `ndarray` with dimensions of size() &times; 2.
     *
     * @exceptsafe Provides strong exception guarantee.
     *
     * @warning Any list operation that inserts elements, removes elements,
     *          or reallocates memory shall invalidate all views previously
     *          returned by this method. It is recommended that this method
     *          only be called on effectively immutable lists.
     */
    const ndarray::Array<double, 2, 2> data();

    /**
     * @copybrief data()
     *
     * As data(), except that the array view does not allow any
     * changes to this list.
     */
    const ndarray::Array<const double, 2, 2> data() const;

private:
};

/*
 * List comparisons
 */

/**
 * `true` if two lists are equal
 *
 * Two lists are considered equal if and only if they have the same length,
 * and each element is equal to the corresponding element in the other
 * list.
 *
 * @param lhs, rhs The two lists to compare.
 * @return `true` if the lists have the same elements in the same order,
 *         `false` otherwise
 *
 * @exceptsafe Shall not throw exceptions.
 *
 * @relatesalso Point2DList
 */
bool operator==(Point2DList const &lhs, Point2DList const &rhs) noexcept;

/**
 * `false` if two lists are equal
 *
 * This operator shall always return the logical negation of `==`; see
 * its documentation for a detailed specification.
 *
 * @exceptsafe Shall not throw exceptions.
 *
 * @relatesalso Point2DList
 */
bool operator!=(Point2DList const &lhs, Point2DList const &rhs) noexcept;

/*
 * Swap
 */

/**
 * Exchange the content of two lists.
 *
 * All elements previously in `list1` shall be in `list2`, and vice
 * versa. All iterators, references, and pointers referring to either
 * list (except lsst::afw::geom::Point2DList::end()) shall
 * remain valid, but shall be associated with the other list.
 *
 * This is a specialization of `std::swap` that offers improved performance
 * and better consistency and stability guarantees than the generic
 * algorithm.
 *
 * @param list1, list2 the two lists whose elements shall be exchanged
 *
 * @exceptsafe Shall not throw exceptions.
 *
 * @relatesalso Point2DList
 */
void swap(Point2DList &list1, Point2DList &list2) noexcept { list1.swap(list2); }

/*
 * Iterator arithmetic
 */

/**
 * Offset a random access iterator for Point2DList by an arbitrary amount.
 *
 * @param amount The number of elements by which to move away from `base`.
 * @param base The iterator from which to offset.
 * @return An iterator `it` such that `it - base` = `amount`. Behavior is
 *         **undefined** if no such iterator exists.
 *
 * @relatesalso Point2DList
 */
Point2DList::const_iterator operator+(Point2DList::const_iterator::difference_type amount,
                                      const Point2DList::const_iterator &base);

/**
 * Offset a random access iterator for Point2DList by an arbitrary amount.
 *
 * @param amount The number of elements by which to move away from `base`.
 * @param base The iterator from which to offset.
 * @return An iterator `it` such that `it - base` = `-amount`. Behavior is
 *         **undefined** if no such iterator exists.
 *
 * @relatesalso Point2DList
 */
Point2DList::const_iterator operator-(Point2DList::const_iterator::difference_type amount,
                                      const Point2DList::const_iterator &base);

template <class InputIterator>
Point2DList::Point2DList(InputIterator first, InputIterator last) {
    throw std::runtime_error("Not yet implemented");
}

template <class InputIterator>
inline Point2DList::iterator Point2DList::insert(const_iterator position, InputIterator first,
                                                 InputIterator last) {
    throw std::runtime_error("Not yet implemented");
}

template <class... Args>
inline Point2DList::iterator Point2DList::emplace(const_iterator position, Args &&... args) {
    throw std::runtime_error("Not yet implemented");
}

template <class... Args>
inline void Point2DList::emplace_back(Args &&... args) {
    throw std::runtime_error("Not yet implemented");
}

template <class InputIterator>
inline void lsst::afw::geom::Point2DList::assign(InputIterator first, InputIterator last) {}
}
}
} /* namespace lsst::afw::geom */

namespace std {
/// @copydoc lsst::afw::geom::swap(Point2DList &, Point2DList &)
template <>
void swap<lsst::afw::geom::Point2DList>(lsst::afw::geom::Point2DList &list1,
                                        lsst::afw::geom::Point2DList &list2) noexcept {
    list1.swap(list2);
}
} /* namespace std */

#endif /* LSST_AFW_GEOM_POINT2DLIST_H_ */
