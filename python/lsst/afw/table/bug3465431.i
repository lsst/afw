// This file contains workarounds for SWIG's lack of complete support
// for partial specialization of templates in C++, reported as SWIG
// bug #3465431.  If/when that bug is fixed, this file need not be
// included (and may actually cause problems if it is included).

%define %specializePoint(U)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Point< U > > {
    Key<U> getX() const { return self->getX(); }
    Key<U> getY() const { return self->getY(); }
}
%enddef

%define %specializeShape(U)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Shape< U > > {
    Key<U> getIXX() const { return self->getIXX(); }
    Key<U> getIYY() const { return self->getIYY(); }
    Key<U> getIXY() const { return self->getIXY(); }
}
%enddef

%define %specializeArray(U)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Array< U > > {
    Key<U> __getitem__(int n) const {
        return (*self)[n];
    }
}
%extend lsst::afw::table::FieldBase< lsst::afw::table::Array< U > > {
    int getSize() const { return self->getSize(); }
}
%enddef

%define %specializeCovariance(U)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Covariance< U > > {
    Key<U> _getitem_impl(int i, int j) const { return (*self)(i, j); }
    %pythoncode %{
        def __getitem__(self, args): return self._getitem_impl(*args)
    %}
}
%extend lsst::afw::table::FieldBase< lsst::afw::table::Covariance< U > > {
    int getSize() const { return self->getSize(); }
    int getPackedSize() const { return self->getPackedSize(); }
}
%extend lsst::afw::table::KeyBase< lsst::afw::table::Covariance< lsst::afw::table::Point< U > > > {
    Key<U> _getitem_impl(int i, int j) const { return (*self)(i, j); }
    %pythoncode %{
        def __getitem__(self, args): return self._getitem_impl(*args)
    %}
}
%extend lsst::afw::table::FieldBase< lsst::afw::table::Covariance< lsst::afw::table::Point< U > > > {
    int getSize() const { return self->getSize(); }
    int getPackedSize() const { return self->getPackedSize(); }
}
%extend lsst::afw::table::KeyBase< lsst::afw::table::Covariance< lsst::afw::table::Shape< U > > > {
    Key<U> _getitem_impl(int i, int j) const { return (*self)(i, j); }
    %pythoncode %{
        def __getitem__(self, args): return self._getitem_impl(*args)
    %}
}
%extend lsst::afw::table::FieldBase< lsst::afw::table::Covariance< lsst::afw::table::Shape< U > > > {
    int getSize() const { return self->getSize(); }
    int getPackedSize() const { return self->getPackedSize(); }
}
%enddef

%specializePoint(int)
%specializePoint(float)
%specializePoint(double)
%specializeShape(float)
%specializeShape(double)
%specializeCovariance(float)
%specializeCovariance(double)

%specializeArray(float)
%specializeArray(double)
