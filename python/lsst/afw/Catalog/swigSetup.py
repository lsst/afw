#setup.py
import distutils
from distutils.core import setup, Extension

setup (name = "simple example",
       version = "0.1",
       ext_modules = [Extension ("_wcstools",
                        ["swigCtgread.i"], 
                        include_dirs = ["/home/mmiller/michelle/tools/wcstools-3.6.2/libwcs","/usr/include/python2.4"], 
                        library_dirs = ["/home/mmiller/michelle/tools/wcstools-3.6.2/libwcs"], 
                        libraries=['c','wcs']
                     )
               ]
    )
