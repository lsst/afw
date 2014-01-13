// This file contains workarounds for SWIG's lack of complete support
// for partial specialization of templates in C++, reported as SWIG
// bug #3465431.  If/when that bug is fixed, this should be simplifed
// by replacing many of the get/set implementations with %template lines.

%define %specializeScalar(U, PYNAME)
%extend lsst::afw::table::KeyBase< U > {
    %pythoncode %{
        subfields = None
        subkeys = None
        HAS_NAMED_SUBFIELDS = False
    %}
}
%extend lsst::afw::table::BaseRecord {
    %template(get) get< U >;
    %template(get##PYNAME) get< U >;
    %template(set) set< U, U >;
    %template(set##PYNAME) set< U, U >;
    U __getitem__(lsst::afw::table::Key< U > const & key) const { return (*self)[key]; }
    void __setitem__(lsst::afw::table::Key< U > const & key, U value) { (*self)[key] = value; }
}
%extend lsst::afw::table::BaseColumnView {
    ndarray::Array<U,1> __getitem__(Key<U> const & key) const { return (*self)[key]; }
    void __setitem__(Key<U> const & key, ndarray::Array<U const,1> const & v) const { (*self)[key] = v; }
}
%enddef

%define %specializePoint(U, PYNAME, VALUE...)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Point< U > > {
    lsst::afw::table::Key<U> getX() const { return self->getX(); }
    lsst::afw::table::Key<U> getY() const { return self->getY(); }
    %pythoncode %{
        subfields = ("x", "y")
        subkeys = property(lambda self: (self.getX(), self.getY()))
        HAS_NAMED_SUBFIELDS = True
    %}
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
%extend lsst::afw::table::BaseColumnView {
    void __getitem__(Key< Point<U> > const & key) const {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Cannot get column view to Point field."
        );
    }
}
%enddef

%define %specializeMoments(U, PYNAME, VALUE...)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Moments< U > > {
    lsst::afw::table::Key<U> getIxx() const { return self->getIxx(); }
    lsst::afw::table::Key<U> getIyy() const { return self->getIyy(); }
    lsst::afw::table::Key<U> getIxy() const { return self->getIxy(); }
    %pythoncode %{
        subfields = ("xx", "yy", "xy")
        subkeys = property(lambda self: (self.getIxx(), self.getIyy(), self.getIxy()))
        HAS_NAMED_SUBFIELDS = True
    %}
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
%extend lsst::afw::table::BaseColumnView {
    void __getitem__(Key< Moments<U> > const & key) const {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Cannot get column view to Moments field."
        );
    }
}
%enddef

%define %specializeArray(U, PYNAME)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Array< U > > {
    lsst::afw::table::Key<U> get(int n) const {
        return (*self)[n];
    }
    lsst::afw::table::Key< lsst::afw::table::Array< U > > slice(int begin, int end) const {
        return self->slice(begin, end);
    }
    %pythoncode %{
        subfields = property(lambda self: tuple(range(self.getSize())))
        subkeys = property(lambda self: tuple(self[i] for i in range(self.getSize())))
        HAS_NAMED_SUBFIELDS = False

        def __getitem__(self, k):
            if isinstance(k, slice):
                return self.slice(k.start, k.stop)
            else:
                return self.get(k)
    %}
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
        ndarray::Array<U,1> const & v
    ) {
        (*self)[key] = v;
    }

}
%extend lsst::afw::table::BaseColumnView {
    ndarray::Array<U,2> __getitem__(Key< lsst::afw::table::Array<U> > const & key) const {
        return (*self)[key];
    }
    void __setitem__(
        Key< lsst::afw::table::Array<U> > const & key,
        ndarray::Array<U const,2> const & v
    ) const {
        (*self)[key] = v;
    }
}
%enddef

%define %specializeCovariance(U, PYNAME)
%extend lsst::afw::table::KeyBase< lsst::afw::table::Covariance< U > > {
    lsst::afw::table::Key<U> _getitem_impl(int i, int j) const { return (*self)(i, j); }
    %pythoncode %{
        def __getitem__(self, args): return self._getitem_impl(*args)
        subfields = property(_syntax.KeyBaseCov_subfields)
        subkeys = property(_syntax.KeyBaseCov_subkeys)
        HAS_NAMED_SUBFIELDS = False
    %}
}
%extend lsst::afw::table::FieldBase< lsst::afw::table::Covariance< U > > {
    int getSize() const { return self->getSize(); }
    int getPackedSize() const { return self->getPackedSize(); }
}
%extend lsst::afw::table::BaseRecord {

    Eigen::Matrix<U,Eigen::Dynamic,Eigen::Dynamic> get(
        lsst::afw::table::Key< lsst::afw::table::Covariance< U > > const & key
    ) const {
        return self->get(key);
    }

    Eigen::Matrix<U,Eigen::Dynamic,Eigen::Dynamic> getCov##PYNAME(
        lsst::afw::table::Key< lsst::afw::table::Covariance< U > > const & key
    ) const {
        return self->get(key);
    }

    void set(
        lsst::afw::table::Key< lsst::afw::table::Covariance< U > > const & key,
        Eigen::Matrix<U,Eigen::Dynamic,Eigen::Dynamic> const & v
    ) {
        self->set(key, v);
    }

    void setCov##PYNAME(
        lsst::afw::table::Key< lsst::afw::table::Covariance< U > > const & key,
        Eigen::Matrix<U,Eigen::Dynamic,Eigen::Dynamic> const & v
    ) {
        self->set(key, v);
    }
}
%extend lsst::afw::table::BaseColumnView {
    void __getitem__(Key< lsst::afw::table::Covariance<U> > const & key) const {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Cannot get column view to Covariance field."
        );
    }
}
%extend lsst::afw::table::KeyBase< lsst::afw::table::Covariance< lsst::afw::table::Point< U > > > {
    lsst::afw::table::Key<U> _getitem_impl(int i, int j) const { return (*self)(i, j); }
    %pythoncode %{
        def __getitem__(self, args): return self._getitem_impl(*args)
        subfields = property(_syntax.KeyBaseCov_subfields)
        subkeys = property(_syntax.KeyBaseCov_subkeys)
        HAS_NAMED_SUBFIELDS = False
    %}
}
%extend lsst::afw::table::FieldBase< lsst::afw::table::Covariance< lsst::afw::table::Point< U > > > {
    int getSize() const { return self->getSize(); }
    int getPackedSize() const { return self->getPackedSize(); }
}
%extend lsst::afw::table::BaseRecord {
    Eigen::Matrix<U,2,2> get(
        lsst::afw::table::Key< lsst::afw::table::Covariance< lsst::afw::table::Point<U> > > const & key
    ) const {
        return self->get(key);
    }

    Eigen::Matrix<U,2,2> getCovPoint##PYNAME(
        lsst::afw::table::Key< lsst::afw::table::Covariance< lsst::afw::table::Point<U> > > const & key
    ) const {
        return self->get(key);
    }

    void set(
        lsst::afw::table::Key< lsst::afw::table::Covariance< lsst::afw::table::Point<U> > > const & key,
        Eigen::Matrix<U,2,2> const & v
    ) {
        self->set(key, v);
    }

    void setCovPoint##PYNAME(
        lsst::afw::table::Key< lsst::afw::table::Covariance< lsst::afw::table::Point<U> > > const & key,
        Eigen::Matrix<U,2,2> const & v
    ) {
        self->set(key, v);
    }
}
%extend lsst::afw::table::BaseColumnView {
    void __getitem__(Key< lsst::afw::table::Covariance< lsst::afw::table::Point<U> > > const & key) const {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Cannot get column view to Covariance field."
        );
    }
}
%extend lsst::afw::table::KeyBase< lsst::afw::table::Covariance< lsst::afw::table::Moments< U > > > {
    lsst::afw::table::Key<U> _getitem_impl(int i, int j) const { return (*self)(i, j); }
    %pythoncode %{
        def __getitem__(self, args): return self._getitem_impl(*args)
        subfields = property(_syntax.KeyBaseCov_subfields)
        subkeys = property(_syntax.KeyBaseCov_subkeys)
        HAS_NAMED_SUBFIELDS = False
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
%extend lsst::afw::table::BaseColumnView {
    void __getitem__(Key< lsst::afw::table::Covariance< lsst::afw::table::Moments<U> > > const & key) const {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Cannot get column view to Covariance field."
        );
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

    void setFlag(lsst::afw::table::Key< Flag > const & key, bool value) {
        self->set(key, value);
    }

}

%extend lsst::afw::table::KeyBase< Flag > {
    %pythoncode %{
        subfields = None
        subkeys = None
        HAS_NAMED_SUBFIELDS = False
    %}
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
    %pythoncode %{
        subfields = ("ra", "dec")
        subkeys = property(lambda self: (self.getRa(), self.getDec()))
        HAS_NAMED_SUBFIELDS = True
    %}
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

    void setCoord(
        lsst::afw::table::Key< lsst::afw::coord::Coord > const & key,
        lsst::afw::coord::Coord const & v
    ) {
        self->set(key, v);
    }
}
%extend lsst::afw::table::BaseColumnView {
    void __getitem__(Key< lsst::afw::coord::Coord > const & key) const {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Cannot get column view to Coord field."
        );
    }
}

%extend lsst::afw::table::BaseRecord {

    std::string get(lsst::afw::table::Key< std::string > const & key) const {
        return self->get(key);
    }

    std::string getString(lsst::afw::table::Key< std::string > const & key) const {
        return self->get(key);
    }

    std::string __getitem__(lsst::afw::table::Key< std::string > const & key) const {
        return self->get(key);
    }

    void set(lsst::afw::table::Key< std::string > const & key, std::string const & v) {
        self->set(key, v);
    }

    void setString(lsst::afw::table::Key< std::string > const & key, std::string const & v) {
        self->set(key, v);
    }

    void __setitem__(lsst::afw::table::Key< std::string > const & key, std::string const & v) {
        self->set(key, v);
    }
}
%extend lsst::afw::table::KeyBase< std::string > {
    %pythoncode %{
        subfields = None
        subkeys = None
        HAS_NAMED_SUBFIELDS = False
    %}
}

%specializeScalar(boost::int32_t, I)
%specializeScalar(boost::int64_t, L)
%specializeScalar(float, F)
%specializeScalar(double, D)
%specializeScalar(lsst::afw::geom::Angle, Angle)

%specializePoint(boost::int32_t, I, lsst::afw::geom::Point<int,2>)
%specializePoint(float, F, lsst::afw::geom::Point<double,2>)
%specializePoint(double, D, lsst::afw::geom::Point<double,2>)
%specializeMoments(float, F, lsst::afw::geom::ellipses::Quadrupole)
%specializeMoments(double, D, lsst::afw::geom::ellipses::Quadrupole)

%specializeArray(float, F)
%specializeArray(double, D)
%specializeCovariance(float, F)
