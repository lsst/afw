try:
    import sys
    sys.meta_path[0].LSSTImporter()     # Is it already loaded?
except (NameError, IndexError):
    import imp, os, sys

    class LSSTImporter:
        """An importer to go on sys.meta_path that enables you to
        find a module somewhere on sys.path, even if not found at
        the top level (requires python 2.5; cf PEP 302).

        More precisely, look down one level, which permits a sys.path
        of ["foo", "goo", "hoo"] each of which contains
        an lsst directory (foo/lsst, goo/lsst, ...) and still allow the
        user to say "import hoo.lsst.pkg"
        """

        def __init__(self):
            _fd = None
            _filename = None
            _desc = None

        def LSSTImporter(self):
            """Is this the LSSTImporter?"""
            return True

        def find_module(self, fullname, path = None):
            """Find a module somewhere on the sys.path, even if not found
            at the top level."""

            name = fullname.split(".")[-1]

            try:
                (self._fd, self._filename, self._desc) = imp.find_module(name, path)
            except:
                for d in sys.path:
                    dirname = os.path.join(d, apply(os.path.join, fullname.split(".")))
                    if os.path.isabs(dirname) and os.path.isdir(dirname):
                        (self._fd, self._filename, self._desc) = \
                                   imp.find_module(name, [os.path.dirname(dirname)])
                        return self

            return None

        def load_module(self, fullname):
            """Load a module, using the information from find_module"""
            fd = self._fd;             self._fd = None
            filename = self._filename; self._filename = None
            desc = self._desc;         self._desc = None
            
            return imp.load_module(fullname, fd, filename, desc)
        
    sys.meta_path += [LSSTImporter()]
