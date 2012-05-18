// This file contains workarounds for SWIG's lack of complete support
// for partial specialization of templates in C++, reported as SWIG
// bug #3465431.  If/when that bug is fixed, this should be simplifed
// by replacing many of the get/set implementations with %template lines.

%define %specializeScalar(U, PYNAME)
%extend lsst::afw::table::BaseRecord {
    %template(get) get< U >;
    %template(get##PYNAME) get< U >;
    %template(set) set< U, U >;
    %template(set##PYNAME) set< U, U >;
    U __getitem__(lsst::afw::table::Key< U > const & key) const { return (*self)[key]; }
    void __setitem__(lsst::afw::table::Key< U > const & key, U value) { (*self)[key] = value; }
}
%extend lsst::afw::table::BaseColumnView {
    ndarray::Array<U const,1> __getitem__(Key<U> const & key) const { return (*self)[key]; }
}
%enddef

%define %specializePoint(U, PYNAME, VALUE...)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Point< U > > {
    lsst::afw::table::Key<U> getX() const { return self->getX(); }
    lsst::afw::table::Key<U> getY() const { return self->getY(); }
}
%extend lsst::afw::table::BaseRecord {

    VALUE get(lsst::afw::table::Key< Point< U > > const & key) const
    { return self->get(key); }

    VALUE getPoint##PYNAME(lsst::afw::table::Key< Point< U > > const & key) const
    { return self->get(key); }

    void set(lsst::afw::table::Key< Point< U > > const & key, VALUE const & v)
    { self->set(key, v); }

    void setPoint##PYNAME(lsst::afw::table::Key< Point< U > > const & key, VALUE const & v)
    { self->set(key, v); }

}
%enddef

%define %specializeMoments(U, PYNAME, VALUE...)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Moments< U > > {
    lsst::afw::table::Key<U> getIxx() const { return self->getIxx(); }
    lsst::afw::table::Key<U> getIyy() const { return self->getIyy(); }
    lsst::afw::table::Key<U> getIxy() const { return self->getIxy(); }
}
%extend lsst::afw::table::BaseRecord {

    VALUE get(lsst::afw::table::Key< Moments<U> > const & key) const
    { return self->get(key); }

    VALUE getMoments##PYNAME(lsst::afw::table::Key< Moments<U> > const & key) const
    { return self->get(key); }

    void set(lsst::afw::table::Key< Moments<U> > const & key, VALUE const & v)
    { self->set(key, v); }

    void setMoments##PYNAME(lsst::afw::table::Key< Moments<U> > const & key, VALUE const & v)
    { self->set(key, v); }

}
%enddef

%define %specializeArray(U, PYNAME)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Array< U > > {
    lsst::afw::table::Key<U> __getitem__(int n) const {
        return (*self)[n];
    }
}
%extend lsst::afw::table::FieldBase< lsst::afw::table::Array< U > > {
    int getSize() const { return self->getSize(); }
}
%extend lsst::afw::table::BaseRecord {

    ndarray::Array<U const,1,1> get(lsst::afw::table::Key< Array< U > > const & key) const
    { return self->get(key); }

    ndarray::Array<U const,1,1> getArray##PYNAME(lsst::afw::table::Key< Array< U > > const & key) const
    { return self->get(key); }
    
    void set(
        lsst::afw::table::Key< Array< U > > const & key,
        ndarray::Array<U const,1> const & v
    ) {
        self->set(key, v);
    }
    void setArray##PYNAME(
        lsst::afw::table::Key< Array< U > > const & key,
        ndarray::Array<U const,1> const & v
    ) {
        self->set(key, v);
    }

    ndarray::Array<U,1,1> __getitem__(lsst::afw::table::Key< Array< U > > const & key) {
        return (*self)[key];
    }

    void __setitem__(
        lsst::afw::table::Key< Array< U > > const & key,
        ndarray::Array<U const,1> const & v
    ) {
        (*self)[key] = v;
    }

}
%extend lsst::afw::table::BaseColumnView {
    ndarray::Array<U const,2> __getitem__(Key< lsst::afw::table::Array<U> > const & key) const {
        return (*self)[key];
    }
}
%enddef

     %define %specializeCovariance(U, PYNAME)
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
%extend lsst::afw::table::BaseRecord {

    Eigen::Matrix<U,Eigen::Dynamic,Eigen::Dynamic> get(
        lsst::afw::table::Key< Covariance< U > > const & key
    ) const {
        return self->get(key);
    }

    Eigen::Matrix<U,Eigen::Dynamic,Eigen::Dynamic> getCov##PYNAME(
        lsst::afw::table::Key< Covariance< U > > const & key
    ) const {
        return self->get(key);
    }

    void set(
        lsst::afw::table::Key< Covariance< U > > const & key,
        Eigen::Matrix<U,Eigen::Dynamic,Eigen::Dynamic> const & v
    ) {
        self->set(key, v);
    }

    void setCov##PYNAME(
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
%extend lsst::afw::table::BaseRecord {
    Eigen::Matrix<U,2,2> get(lsst::afw::table::Key< Covariance< Point< U > > > const & key) const {
        return self->get(key);
    }

    Eigen::Matrix<U,2,2> getCovPoint##PYNAME(
        lsst::afw::table::Key< Covariance< Point< U > > > const & key
    ) const {
        return self->get(key);
    }

    void set(
        lsst::afw::table::Key< Covariance< Point< U > > > const & key,
        Eigen::Matrix<U,2,2> const & v
    ) {
        self->set(key, v);
    }

    void setCovPoint##PYNAME(
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
%extend lsst::afw::table::BaseRecord {
    Eigen::Matrix<U,3,3> get(
        lsst::afw::table::Key< Covariance< Moments< U > > > const & key
    ) const {
        return self->get(key);
    }

    Eigen::Matrix<U,3,3> getCovMoments##PYNAME(
        lsst::afw::table::Key< Covariance< Moments< U > > > const & key
    ) const {
        return self->get(key);
    }

    void set(
        lsst::afw::table::Key< Covariance< Moments< U > > > const & key,
        Eigen::Matrix<U,3,3> const & v
    ) {
        self->set(key, v);
    }

    void setCovMoments##PYNAME(
        lsst::afw::table::Key< Covariance< Moments< U > > > const & key,
        Eigen::Matrix<U,3,3> const & v
    ) {
        self->set(key, v);
    }
}
%enddef

%extend lsst::afw::table::BaseRecord {

    bool get(lsst::afw::table::Key< Flag > const & key) const {
        return self->get(key);
    }

    bool getFlag(lsst::afw::table::Key< Flag > const & key) const {
        return self->get(key);
    }

    void set(lsst::afw::table::Key< Flag > const & key, bool value) {
        self->set(key, value);
    }

    void set_Flag(lsst::afw::table::Key< Flag > const & key, bool value) {
        self->set(key, value);
    }

}
%extend lsst::afw::table::BaseColumnView {
    ndarray::Array<bool const,1> __getitem__(
        lsst::afw::table::Key< lsst::afw::table::Flag > const & key
    ) const {
        return ndarray::copy((*self)[key]);
    }
}

%extend lsst::afw::table::KeyBase< lsst::afw::coord::Coord > {
    lsst::afw::table::Key<lsst::afw::geom::Angle> getRa() const { return self->getRa(); }
    lsst::afw::table::Key<lsst::afw::geom::Angle> getDec() const { return self->getDec(); }
}
%extend lsst::afw::table::BaseRecord {

    lsst::afw::coord::IcrsCoord get(
        lsst::afw::table::Key< lsst::afw::coord::Coord > const & key
    ) const {
        return self->get(key);
    }

    lsst::afw::coord::IcrsCoord getCoord(
        lsst::afw::table::Key< lsst::afw::coord::Coord > const & key
    ) const {
        return self->get(key);
    }

    void set(
        lsst::afw::table::Key< lsst::afw::coord::Coord > const & key,
        lsst::afw::coord::Coord const & v
    ) {
        self->set(key, v);
    }

    void set_Coord(
        lsst::afw::table::Key< lsst::afw::coord::Coord > const & key,
        lsst::afw::coord::Coord const & v
    ) {
        self->set(key, v);
    }
}

%specializeScalar(boost::int32_t, I4)
%specializeScalar(boost::int64_t, I8)
%specializeScalar(float, F4)
%specializeScalar(double, F8)
%specializeScalar(lsst::afw::geom::Angle, Angle)

%specializePoint(boost::int32_t, I4, lsst::afw::geom::Point<int,2>)
%specializePoint(float, F4, lsst::afw::geom::Point<double,2>)
%specializePoint(double, F8, lsst::afw::geom::Point<double,2>)
%specializeMoments(float, F4, lsst::afw::geom::ellipses::Quadrupole)
%specializeMoments(double, F8, lsst::afw::geom::ellipses::Quadrupole)

%specializeArray(float, F4)
%specializeArray(double, F8)
%specializeCovariance(float, F4)
%specializeCovariance(double, F8)
