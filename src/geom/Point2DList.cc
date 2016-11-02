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

#include "ndarray.h"

#include "lsst/afw/geom/Point2DList.h"

namespace lsst {
namespace afw {
namespace geom {

Point2DList::Point2DList() { throw std::runtime_error("Not yet implemented"); }

Point2DList::Point2DList(size_type n) { throw std::runtime_error("Not yet implemented"); }

Point2DList::Point2DList(size_type n, const value_type& point) {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::Point2DList(const Point2DList& other) { throw std::runtime_error("Not yet implemented"); }

Point2DList::Point2DList(Point2DList&& other) { throw std::runtime_error("Not yet implemented"); }

Point2DList::Point2DList(std::initializer_list<value_type> values) {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::~Point2DList() { throw std::runtime_error("Not yet implemented"); }

bool Point2DList::empty() const noexcept { throw std::runtime_error("Not yet implemented"); }

Point2DList::size_type Point2DList::size() const noexcept { throw std::runtime_error("Not yet implemented"); }

Point2DList::size_type Point2DList::max_size() const noexcept {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::iterator Point2DList::begin() noexcept { throw std::runtime_error("Not yet implemented"); }

Point2DList::const_iterator Point2DList::begin() const noexcept {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::iterator Point2DList::end() noexcept { throw std::runtime_error("Not yet implemented"); }

Point2DList::const_iterator Point2DList::end() const noexcept {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::reverse_iterator Point2DList::rbegin() noexcept {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::const_reverse_iterator Point2DList::rbegin() const noexcept {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::reverse_iterator Point2DList::rend() noexcept {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::const_reverse_iterator Point2DList::rend() const noexcept {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::const_iterator Point2DList::cbegin() const noexcept {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::const_iterator Point2DList::cend() const noexcept {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::const_reverse_iterator Point2DList::crbegin() const noexcept {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::const_reverse_iterator Point2DList::crend() const noexcept {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::reference Point2DList::front() { throw std::runtime_error("Not yet implemented"); }

Point2DList::const_reference Point2DList::front() const { throw std::runtime_error("Not yet implemented"); }

Point2DList::reference Point2DList::back() { throw std::runtime_error("Not yet implemented"); }

Point2DList::const_reference Point2DList::back() const { throw std::runtime_error("Not yet implemented"); }

Point2DList::reference Point2DList::operator[](size_type index) {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::const_reference Point2DList::operator[](size_type index) const {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::reference Point2DList::at(size_type index) { throw std::runtime_error("Not yet implemented"); }

Point2DList::const_reference Point2DList::at(size_type index) const {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::iterator Point2DList::insert(const_iterator position, const value_type& value) {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::iterator Point2DList::insert(const_iterator position, value_type&& value) {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::iterator Point2DList::insert(const_iterator position, size_type n, const value_type& value) {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::iterator Point2DList::insert(const_iterator position, std::initializer_list<value_type> values) {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::iterator Point2DList::erase(const_iterator position) {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::iterator Point2DList::erase(const_iterator first, const_iterator last) {
    throw std::runtime_error("Not yet implemented");
}

void Point2DList::push_back(const value_type& value) { throw std::runtime_error("Not yet implemented"); }

void Point2DList::push_back(value_type&& value) { throw std::runtime_error("Not yet implemented"); }

void Point2DList::pop_back() { throw std::runtime_error("Not yet implemented"); }

void Point2DList::swap(Point2DList& other) noexcept { throw std::runtime_error("Not yet implemented"); }

Point2DList& Point2DList::operator=(const Point2DList& other) {
    throw std::runtime_error("Not yet implemented");
}

Point2DList& Point2DList::operator=(Point2DList&& other) { throw std::runtime_error("Not yet implemented"); }

Point2DList& Point2DList::operator=(std::initializer_list<value_type> values) {
    throw std::runtime_error("Not yet implemented");
}

void Point2DList::assign(size_type n, const value_type& point) {
    throw std::runtime_error("Not yet implemented");
}

void Point2DList::assign(std::initializer_list<value_type> values) {
    throw std::runtime_error("Not yet implemented");
}

void Point2DList::resize(size_type n) { throw std::runtime_error("Not yet implemented"); }

void Point2DList::resize(size_type n, const value_type& value) {
    throw std::runtime_error("Not yet implemented");
}

void Point2DList::clear() noexcept { throw std::runtime_error("Not yet implemented"); }

Point2DList::size_type Point2DList::capacity() const noexcept {
    throw std::runtime_error("Not yet implemented");
}

void Point2DList::reserve(size_type n) { throw std::runtime_error("Not yet implemented"); }

void Point2DList::shrink_to_fit() { throw std::runtime_error("Not yet implemented"); }

const ndarray::Array<double, 2, 2> Point2DList::data() { throw std::runtime_error("Not yet implemented"); }

const ndarray::Array<const double, 2, 2> Point2DList::data() const {
    throw std::runtime_error("Not yet implemented");
}

bool operator==(const Point2DList& lhs, const Point2DList& rhs) noexcept {
    throw std::runtime_error("Not yet implemented");
}

bool operator!=(const Point2DList& lhs, const Point2DList& rhs) noexcept {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::const_iterator operator+(Point2DList::const_iterator::difference_type amount,
                                      const Point2DList::const_iterator& base) {
    throw std::runtime_error("Not yet implemented");
}

Point2DList::const_iterator operator-(Point2DList::const_iterator::difference_type amount,
                                      const Point2DList::const_iterator& base) {
    throw std::runtime_error("Not yet implemented");
}
}
}
} /* namespace lsst::afw::geom */
