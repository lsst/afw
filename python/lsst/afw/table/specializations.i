// This file contains workarounds for SWIG's lack of complete support
// for partial specialization of templates in C++, reported as SWIG
// bug #3465431.  If/when that bug is fixed, this should be simplifed
// by replacing many of the get/set implementations with %template lines.

%define %specializeScalar(U)
%extend lsst::afw::table::RecordBase {
    %template(get) get< U >;
    %template(set) set< U, U >;
    U __getitem__(lsst::afw::table::Key< U > const & key) const { return (*self)[key]; }
    void __setitem__(lsst::afw::table::Key< U > const & key, U value) { (*self)[key] = value; }
}
%extend lsst::afw::table::ColumnView {
    lsst::ndarray::Array<U const,1> __getitem__(Key<U> const & key) const { return (*self)[key]; }
}
%enddef

%define %specializePoint(U, VALUE...)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Point< U > > {
    lsst::afw::table::Key<U> getX() const { return self->getX(); }
    lsst::afw::table::Key<U> getY() const { return self->getY(); }
}
%extend lsst::afw::table::RecordBase {
    VALUE get(lsst::afw::table::Key< Point< U > > const & key) const { return self->get(key); }
    void set(lsst::afw::table::Key< Point< U > > const & key, VALUE const & v) { self->set(key, v); }
}
%enddef

%define %specializeMoments(U, VALUE...)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Moments< U > > {
    lsst::afw::table::Key<U> getIXX() const { return self->getIXX(); }
    lsst::afw::table::Key<U> getIYY() const { return self->getIYY(); }
    lsst::afw::table::Key<U> getIXY() const { return self->getIXY(); }
}
%extend lsst::afw::table::RecordBase {
    VALUE get(lsst::afw::table::Key< Moments< U > > const & key) const { return self->get(key); }
    void set(lsst::afw::table::Key< Moments< U > > const & key, VALUE const & v) { self->set(key, v); }
}
%enddef

%define %specializeArray(U)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Array< U > > {
    lsst::afw::table::Key<U> __getitem__(int n) const {
        return (*self)[n];
    }
}
%extend lsst::afw::table::FieldBase< lsst::afw::table::Array< U > > {
    int getSize() const { return self->getSize(); }
}
%extend lsst::afw::table::RecordBase {
    lsst::ndarray::Array<U const,1,1> get(lsst::afw::table::Key< Array< U > > const & key) const {
        return self->get(key);
    }
    void set(
        lsst::afw::table::Key< Array< U > > const & key,
        lsst::ndarray::Array<U const,1> const & v
    ) {
        self->set(key, v);
    }
    lsst::ndarray::Array<U,1,1> __getitem__(lsst::afw::table::Key< Array< U > > const & key) {
        return (*self)[key];
    }
}
%extend lsst::afw::table::ColumnView {
    lsst::ndarray::Array<U const,2> __getitem__(Key< lsst::afw::table::Array<U> > const & key) const {
        return (*self)[key];
    }
}
%enddef

%define %specializeCovariance(U)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Covariance< U > > {
    lsst::afw::table::Key<U> _getitem_impl(int i, int j) const { return (*self)(i, j); }
    %pythoncode %{
        def __getitem__(self, args): return self._getitem_impl(*args)
    %}
}
%extend lsst::afw::table::FieldBase< lsst::afw::table::Covariance< U > > {
    int getSize() const { return self->getSize(); }
    int getPackedSize() const { return self->getPackedSize(); }
}
%extend lsst::afw::table::RecordBase {
    Eigen::Matrix<U,Eigen::Dynamic,Eigen::Dynamic>
    get(lsst::afw::table::Key< Covariance< U > > const & key) const {
        return self->get(key);
    }
    void set(
        lsst::afw::table::Key< Covariance< U > > const & key,
        Eigen::Matrix<U,Eigen::Dynamic,Eigen::Dynamic> const & v
    ) {
        self->set(key, v);
    }
}
%extend lsst::afw::table::KeyBase< lsst::afw::table::Covariance< lsst::afw::table::Point< U > > > {
    lsst::afw::table::Key<U> _getitem_impl(int i, int j) const { return (*self)(i, j); }
    %pythoncode %{
        def __getitem__(self, args): return self._getitem_impl(*args)
    %}
}
%extend lsst::afw::table::FieldBase< lsst::afw::table::Covariance< lsst::afw::table::Point< U > > > {
    int getSize() const { return self->getSize(); }
    int getPackedSize() const { return self->getPackedSize(); }
}
%extend lsst::afw::table::RecordBase {
    Eigen::Matrix<U,2,2> get(lsst::afw::table::Key< Covariance< Point< U > > > const & key) const {
        return self->get(key);
    }
    void set(
        lsst::afw::table::Key< Covariance< Point< U > > > const & key,
        Eigen::Matrix<U,2,2> const & v
    ) {
        self->set(key, v);
    }
}
%extend lsst::afw::table::KeyBase< lsst::afw::table::Covariance< lsst::afw::table::Moments< U > > > {
    lsst::afw::table::Key<U> _getitem_impl(int i, int j) const { return (*self)(i, j); }
    %pythoncode %{
        def __getitem__(self, args): return self._getitem_impl(*args)
    %}
}
%extend lsst::afw::table::FieldBase< lsst::afw::table::Covariance< lsst::afw::table::Moments< U > > > {
    int getSize() const { return self->getSize(); }
    int getPackedSize() const { return self->getPackedSize(); }
}
%extend lsst::afw::table::RecordBase {
    Eigen::Matrix<U,3,3> get(lsst::afw::table::Key< Covariance< Moments< U > > > const & key) const {
        return self->get(key);
    }
    void set(
        lsst::afw::table::Key< Covariance< Moments< U > > > const & key,
        Eigen::Matrix<U,3,3> const & v
    ) {
        self->set(key, v);
    }
}
%enddef

%extend lsst::afw::table::RecordBase {
    bool get(lsst::afw::table::Key< Flag > const & key) const {
        return self->get(key);
    }
    void set(lsst::afw::table::Key< Flag > const & key, bool value) {
        self->set(key, value);
    }
}
%extend lsst::afw::table::ColumnView {
    lsst::ndarray::Array<bool const,1> __getitem__(
        lsst::afw::table::Key< lsst::afw::table::Flag > const & key
    ) const {
        return lsst::ndarray::copy((*self)[key]);
    }
}

%extend lsst::afw::table::KeyBase< lsst::afw::coord::Coord > {
    lsst::afw::table::Key<lsst::afw::geom::Angle> getRa() const { return self->getRa(); }
    lsst::afw::table::Key<lsst::afw::geom::Angle> getDec() const { return self->getDec(); }
}
%extend lsst::afw::table::RecordBase {
    lsst::afw::coord::IcrsCoord get(lsst::afw::table::Key< lsst::afw::coord::Coord > const & key) const {
        return self->get(key);
    }
    void set(lsst::afw::table::Key< lsst::afw::coord::Coord > const & key,
             lsst::afw::coord::Coord const & v) {
        self->set(key, v);
    }
}

%specializeScalar(boost::int32_t)
%specializeScalar(boost::int64_t)
%specializeScalar(float)
%specializeScalar(double)
%specializeScalar(lsst::afw::geom::Angle)

%specializePoint(boost::int32_t, lsst::afw::geom::Point<int,2>)
%specializePoint(float, lsst::afw::geom::Point<double,2>)
%specializePoint(double, lsst::afw::geom::Point<double,2>)
%specializeMoments(float, lsst::afw::geom::ellipses::Quadrupole)
%specializeMoments(double, lsst::afw::geom::ellipses::Quadrupole)

%specializeArray(float)
%specializeArray(double)
%specializeCovariance(float)
%specializeCovariance(double)
