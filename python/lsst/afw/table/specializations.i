// This file contains workarounds for SWIG's lack of complete support
// for partial specialization of templates in C++, reported as SWIG
// bug #3465431.  If/when that bug is fixed, this should be simplifed
// quite a bit.

%define %specializeScalar(U)
%extend lsst::afw::table::RecordBase {
    %template(get) get< U >;
    %template(set) set< U, U >;
    U __getitem__(lsst::afw::table::Key< U > const & key) const { return (*self)[key]; }
    void __setitem__(lsst::afw::table::Key< U > const & key, U value) { (*self)[key] = value; }
}
%enddef

%define %specializePoint(U, VALUE...)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Point< U > > {
    lsst::afw::table::Key<U> getX() const { return self->getX(); }
    lsst::afw::table::Key<U> getY() const { return self->getY(); }
}
%extend lsst::afw::table::RecordBase {
    VALUE get(lsst::afw::table::Key< Point< U > > const & key) const { return self->get(key); }
    void set(lsst::afw::table::Key< Point< U > > const & key, VALUE const & v) const { self->set(key, v); }
}
%enddef

%define %specializeShape(U, VALUE...)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Shape< U > > {
    lsst::afw::table::Key<U> getIXX() const { return self->getIXX(); }
    lsst::afw::table::Key<U> getIYY() const { return self->getIYY(); }
    lsst::afw::table::Key<U> getIXY() const { return self->getIXY(); }
}
%extend lsst::afw::table::RecordBase {
    VALUE get(lsst::afw::table::Key< Shape< U > > const & key) const { return self->get(key); }
    void set(lsst::afw::table::Key< Shape< U > > const & key, VALUE const & v) const { self->set(key, v); }
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
%extend lsst::afw::table::KeyBase< lsst::afw::table::Covariance< lsst::afw::table::Shape< U > > > {
    lsst::afw::table::Key<U> _getitem_impl(int i, int j) const { return (*self)(i, j); }
    %pythoncode %{
        def __getitem__(self, args): return self._getitem_impl(*args)
    %}
}
%extend lsst::afw::table::FieldBase< lsst::afw::table::Covariance< lsst::afw::table::Shape< U > > > {
    int getSize() const { return self->getSize(); }
    int getPackedSize() const { return self->getPackedSize(); }
}
%enddef

%specializeScalar(boost::int32_t)
%specializeScalar(boost::int64_t)
%specializeScalar(float)
%specializeScalar(double)

%specializePoint(boost::int32_t, lsst::afw::geom::Point<int,2>)
%specializePoint(float, lsst::afw::geom::Point<double,2>)
%specializePoint(double, lsst::afw::geom::Point<double,2>)
%specializeShape(float, lsst::afw::geom::ellipses::Quadrupole)
%specializeShape(double, lsst::afw::geom::ellipses::Quadrupole)
%specializeCovariance(float)
%specializeCovariance(double)

%specializeArray(float)
%specializeArray(double)
