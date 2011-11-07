// -*- c++ -*-
#ifndef CATALOG_Key_h_INCLUDED
#define CATALOG_Key_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"

#include "boost/shared_ptr.hpp"

#include "lsst/catalog/detail/KeyData.h"

#define KEY_STANDARD_PUBLIC_INTERFACE(TYPE)                             \
    template <typename OtherT> bool operator==(Key<OtherT> const & other) const { return false; } \
    template <typename OtherT> bool operator!=(Key<OtherT> const & other) const { return true; } \
    bool operator==(Key const & other) const { return _data == other._data; } \
    bool operator!=(Key const & other) const { return _data == other._data; } \
    Field< TYPE > const & getField() const { return _data->field; }     \
    friend class detail::KeyAccess

#define KEY_STANDARD_PRIVATE_INTERFACE(TYPE)                    \
    typedef detail::KeyData< TYPE > Data;                       \
    Key(boost::shared_ptr<Data> const & data) : _data(data) {}  \
    boost::shared_ptr<Data> _data

namespace lsst { namespace catalog {

template <typename T>
class Key {
public:
    KEY_STANDARD_PUBLIC_INTERFACE(T);
private:
    KEY_STANDARD_PRIVATE_INTERFACE(T);
};

template <typename U>
class Key< Point<U> > {
public:
    KEY_STANDARD_PUBLIC_INTERFACE(Point<U>);

    Key<U> getX() const;
    Key<U> getY() const;

private:
    KEY_STANDARD_PRIVATE_INTERFACE(Point<U>);
};

template <typename U>
class Key< Shape<U> > {
public:
    KEY_STANDARD_PUBLIC_INTERFACE(Shape<U>);

    Key<U> getIxx() const;
    Key<U> getIyy() const;
    Key<U> getIxy() const;

private:
    KEY_STANDARD_PRIVATE_INTERFACE(Shape<U>);
};

template <typename U>
class Key< Array<U> > {
public:
    KEY_STANDARD_PUBLIC_INTERFACE(Array<U>);

    Key<U> at(int i) const;

private:
    KEY_STANDARD_PRIVATE_INTERFACE(Array<U>);
};

template <typename U>
class Key< Covariance<U> > {
public:
    KEY_STANDARD_PUBLIC_INTERFACE(Covariance<U>);

    Key<U> at(int i, int j) const;

private:
    KEY_STANDARD_PRIVATE_INTERFACE(Covariance<U>);
};

template <typename U>
class Key< Covariance< Point<U> > > {
public:
    KEY_STANDARD_PUBLIC_INTERFACE(Covariance< Point<U> >);

    Key<U> at(int i, int j) const;

private:
    KEY_STANDARD_PRIVATE_INTERFACE(Covariance< Point<U> >);
};

template <typename U>
class Key< Covariance< Shape<U> > > {
public:
    KEY_STANDARD_PUBLIC_INTERFACE(Covariance< Shape<U> >);

    Key<U> at(int i, int j) const;

private:
    KEY_STANDARD_PRIVATE_INTERFACE(Covariance< Shape<U> >);
};

}} // namespace lsst::catalog

#endif // !CATALOG_Key_h_INCLUDED
