%define %fits_reduce(cls...)
%extend cls {
    %pythoncode %{
        def __reduce__(self):
            return lsst.afw.fits.reduceToFits(self)
    %}
}
%enddef
