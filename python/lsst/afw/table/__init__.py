from .tableLib import *

for name in globals().keys():
    if "_" in name: # clean up namespace; these are all private or unfortunate swig droppings
        del globals()[name]
