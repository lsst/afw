// -*- lsst-c++  -*-
//
// This should be in lsst/p_lsstSwig.i
//
//
// We'd like to just say
//  def __iter__(self):
//      return next()
// but this crashes, at least with swig 1.3.36
//
%define %definePythonIterator(TYPE...)
%extend TYPE {
    %pythoncode %{
        def __iter__(self):
            ptr = self.begin()
            end = self.end()
            while True:
                if ptr == end:
                    raise StopIteration

                yield ptr.value()
                ptr.incr()

        def __getitem__(self, i):
            return [e for e in self][i]
    %}
}
%enddef
