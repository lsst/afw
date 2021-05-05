// -*- LSST-C++ -*-
/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef LSST_AFW_TYPEHANDLING_DETAIL_REFWRAP_UTILS_H
#define LSST_AFW_TYPEHANDLING_DETAIL_REFWRAP_UTILS_H

#include <functional>
#include <variant>

namespace lsst {
namespace afw {
namespace typehandling {
namespace detail {

// An equality comparison function object for std::variant visitors that
// contain only std::reference_wrapper types.  Because reference_wrapper doesn't
// define operator==, std::variant::operator== can't do this for us.
struct refwrap_equals {

    template <typename T>
    bool operator()(std::reference_wrapper<T> a, std::reference_wrapper<T> b) {
        return a.get() == b.get();
    }

    template <typename T, typename U>
    bool operator()(std::reference_wrapper<T> a, std::reference_wrapper<U> b) {
        static_assert(!std::is_same_v<T, U>);
        return false;
    }

    template <typename ...E>
    bool operator()(
        std::variant<std::reference_wrapper<E>...> const & a,
        std::variant<std::reference_wrapper<E>...> const & b
    ) {
        return std::visit(*this, a, b);
    }

};

// An equivalent to const_cast for std::reference_wrapper.
template <typename T>
std::reference_wrapper<T> refwrap_const_cast(std::reference_wrapper<T const> const & r) {
    return std::ref(const_cast<T &>(r.get()));
}

// A visitor function wrapper for callables with two arguments, where the
// second is the target of a std::reference_wrapper (as is the case for
// GenericMap's visitation interface).
//
// Construct with the make_refwrap_visitor function.
template <typename F>
class refwrap_visitor {
private:
    F _func;
public:

    explicit refwrap_visitor(F && func) : _func(std::forward(func)) {}

    template <typename K, typename V>
    auto operator()(K const & key, std::reference_wrapper<V> const & value) const {
        return _func(key, value.get());
    }

};

// Specialization of refwrap_visitor for functions passed by lvalue reference.
//
// This is necessary because you can't perfectly forward an argument that you
// need to capture in order to use it multiple times; if it's an rvalue
// reference, it needs to be moved into a value (not just passed along into
// another function as an rvalue reference).  Without a specialization, lvalue
// references end up being coped, so modifications to the callable would not be
// reflected to the caller.
template <typename F>
class refwrap_visitor<F&> {
private:
    std::reference_wrapper<F> _func;
public:

    explicit refwrap_visitor(F & func) : _func(std::ref(func)) {}

    template <typename K, typename V>
    auto operator()(K const & key, std::reference_wrapper<V> const & value) const {
        return _func(key, value.get());
    }

};

// Helper function to create a refwrap_visitor instance without explicitly
// specifying the type of the wrapped callable.
template <typename F>
refwrap_visitor<F> make_refwrap_visitor(F&& func) {
    return refwrap_visitor<F>(std::forward<F>(func));
}

}  // namespace detail
}  // namespace typehandling
}  // namespace afw
}  // namespace lsst

#endif
