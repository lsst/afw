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
    bool isVariableLength() const { return self->isVariableLength(); }
}
%extend lsst::afw::table::BaseRecord {

    ndarray::Array<U const,1,1> get(lsst::afw::table::Key< Array< U > > const & key) const
    { return self->get(key); }

    ndarray::Array<U const,1,1> getArray##PYNAME(lsst::afw::table::Key< Array< U > > const & key) const
    { return self->get(key); }

    void set(
        lsst::afw::table::Key< Array< U > > const & key,
        ndarray::Array<U,1,1> const & v
    ) {
        self->set(key, v);
    }
    void setArray##PYNAME(
        lsst::afw::table::Key< Array< U > > const & key,
        ndarray::Array<U,1,1> const & v
    ) {
        self->set(key, v);
    }

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

%specializeScalar(boost::uint16_t, U)
%specializeScalar(boost::int32_t, I)
%specializeScalar(boost::int64_t, L)
%specializeScalar(float, F)
%specializeScalar(double, D)
%specializeScalar(lsst::afw::geom::Angle, Angle)

%specializeArray(boost::uint16_t, U)
%specializeArray(int, I)
%specializeArray(float, F)
%specializeArray(double, D)
