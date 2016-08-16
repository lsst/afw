"""
Code that works with gdb 7.1's python pretty printing.  When gdb >= 7.2 is widely available this
file should be deleted (it's only used after importing gdb.printing fails)
"""
from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from builtins import object
import gdb
import re

class CitizenPrinter(object):
    "Print a Citizen"

    def __init__(self, typename, val):
        self.val = val

    def to_string(self):
        return "{0x%x %d %s 0x%x}" % (self.val, self.val["_CitizenId"],
                                    self.val["_typeName"], self.val["_sentinel"])

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# afw

class BaseSourceAttributesPrinter(object):
    "Print a BaseSourceAttributes"

    def __init__(self, typename, val):
        self.val = val

    def to_string(self):
        return "Base: {id=%d astrom=(%.3f, %.3f)}" % (self.val["_id"], self.val["_xAstrom"], self.val["_yAstrom"])

class SourcePrinter(object):
    "Print a Source"

    def __init__(self, typename, val):
        self.val = val

    def to_string(self):
        return "{id=%d astrom=(%.3f, %.3f)}" % (self.val["_id"], self.val["_xAstrom"], self.val["_yAstrom"])

class FootprintPrinter(object):
    "Print a Footprint"

    def __init__(self, typename, val):
        self.val = val

    def to_string(self):
        return "RHL Footprint (fixme when gdb 7.3 arrives)"

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class CoordinateBasePrinter(object):
    "Print a CoordinateBase"

    def __init__(self, typename, val):
        self.val = val

    def to_string(self):
        # Make sure &foo works, too.
        type = self.val.type
        if type.code == gdb.TYPE_CODE_REF:
            type = type.target ()

        return self.val["_vector"]["m_storage"]["m_data"]["array"]

    def display_hint (self):
        return "array"

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ImagePrinter(object):
    "Print an ImageBase or derived class"

    def dimenStr(self, val=None):
        if val is None:
            val = self.val

        # Make sure &foo works, too.
        type = val.type
        if type.code == gdb.TYPE_CODE_REF:
            type = type.target ()

        gilView = val["_gilView"]
        arr = val["_origin"]["_vector"]["m_storage"]["m_data"]["array"]

        return "%dx%d+%d+%d" % (
            #val["getWidth"](), val["getHeight"](),
            gilView["_dimensions"]["x"], gilView["_dimensions"]["y"],
            arr[0], arr[1])

    def typeName(self):
        return self.typename.split(":")[-1]

    def __init__(self, typename, val):
        self.typename = typename
        self.val = val

    def to_string(self):
        return "%s(%s)" % (self.typeName(), self.dimenStr())

class MaskedImagePrinter(ImagePrinter):
    "Print a MaskedImage"

    def to_string(self):
        return "%s(%s)" % (self.typeName(), self.dimenStr(self.val["_image"]["px"].dereference()))

class ExposurePrinter(ImagePrinter):
    "Print an Exposure"

    def to_string(self):
        return "%s(%s)" % (self.typeName(),
                           self.dimenStr(self.val["_maskedImage"]["_image"]["px"].dereference()))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class PrintImageCommand(gdb.Command):
    """Print an Image
Usage: image x0 y0 [nx [ny] [centerPatch] [obeyXY0]]
"""

    def __init__ (self):
        super (PrintImageCommand, self).__init__ ("show image",
                                                  gdb.COMMAND_DATA,
                                                  gdb.COMPLETE_SYMBOL)

    def get(self, var, x, y):
        if False:
            return var["operator()(int, int, bool)"](x, y, True)
        else:
            dimensions = var["_gilView"]["_dimensions"]
            if x < 0 or x >= dimensions["x"] or y < 0 or y >= dimensions["y"]:
                raise gdb.GdbError("Pixel (%d, %d) is out of range 0:%d, 0:%d" %
                                   (x, y, dimensions["x"] - 1, dimensions["y"] - 1))

            pixels = var["_gilView"]["_pixels"]["_p"]
            step = pixels["_step_fn"]["_step"]/var.type.template_argument(0).sizeof

            return pixels["m_iterator"][x + y*step]["_v0"]

    def invoke (self, args, fromTty):
        self.dont_repeat()

        args = gdb.string_to_argv(args)
        if len(args) < 1:
            raise gdb.GdbError("Please specify an image")
        imgName = args.pop(0)
        var = gdb.parse_and_eval(imgName)

        if re.search(r"MaskedImage", str(var.type)):
            print("N.b. %s is a MaskedImage; showing image" % (imgName))
            var = var["_image"]

        if re.search(r"shared_ptr<", str(var.type)):
            var = var["px"].dereference()

        if var.type.code == gdb.TYPE_CODE_PTR:
            var = var.dereference()     # be nice

        pixelTypeName = str(var.type.template_argument(0))

        if len(args) < 2:
            raise gdb.GdbError("Please specify a pixel's x and y indexes")

        x0 = gdb.parse_and_eval(args.pop(0))
        y0 = gdb.parse_and_eval(args.pop(0))

        if len(args) == 0:
            print("%g" % self.get(var, x0, y0))
            return

        nx = int(args.pop(0))
        if args:
            ny = int(args.pop(0))
        else:
            ny = 1

        if args:
            centerPatch = gdb.parse_and_eval(args.pop(0))
            if centerPatch:
                x0 -= nx//2
                y0 -= ny//2

        if args:
            obeyXY0 = gdb.parse_and_eval(args.pop(0))

            if obeyXY0:
                arr = var["_origin"]["_vector"]["m_storage"]["m_data"]["array"]

                x0 -= arr[0]
                y0 -= arr[1]

        if args:
            raise gdb.GdbError('Unexpected trailing arguments: "%s"' % '", "'.join(args))
        #
        # OK, finally time to print
        #
        if pixelTypeName in ["short", "unsigned short"]:
            dataFmt = "0x%x"
        elif pixelTypeName in ["int", "unsigned int"]:
            dataFmt = "%d"
        else:
            dataFmt = "%.2f"

        print("%-4s" % "", end=' ')
        for x in range(x0, x0 + nx):
            print("%8d" % x, end=' ')
        print("")

        for y in reversed(list(range(y0, y0 + ny))):
            print("%-4d" % y, end=' ')
            for x in range(x0, x0 + nx):
                print("%8s" % (dataFmt % self.get(var, x, y)), end=' ')
            print("")


PrintImageCommand()

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# These two classes (RxPrinter and Printer) come directly from
# python/libstdcxx/v6/printers.py and the GPL license therein applies
#
# A "regular expression" printer which conforms to the
# "SubPrettyPrinter" protocol from gdb.printing.
class RxPrinter(object):
    def __init__(self, name, function):
        super(RxPrinter, self).__init__()
        self.name = name
        self.function = function
        self.enabled = True

    def invoke(self, value):
        if not self.enabled:
            return None
        return self.function(self.name, value)

# A pretty-printer that conforms to the "PrettyPrinter" protocol from
# gdb.printing.  It can also be used directly as an old-style printer.
#
class Printer(object):
    def __init__(self, name):
        super(Printer, self).__init__()
        self.name = name
        self.subprinters = []
        self.lookup = {}
        self.enabled = True
        self.compiled_rx = re.compile('^([a-zA-Z0-9_:]+)<.*>$')

    def add(self, name, function):
        # A small sanity check.
        # FIXME
        if not self.compiled_rx.match(name + '<>'):
            raise ValueError('libstdc++ programming error: "%s" does not match' % name)
        printer = RxPrinter(name, function)
        self.subprinters.append(printer)
        self.lookup[name] = printer

    @staticmethod
    def get_basic_type(type):
        # If it points to a reference, get the reference.
        if type.code == gdb.TYPE_CODE_REF:
            type = type.target ()

        # Get the unqualified type, stripped of typedefs.
        type = type.unqualified ().strip_typedefs ()

        return type.tag

    def __call__(self, val):
        typename = self.get_basic_type(val.type)
        if not typename:
            return None

        # All the types we match are template types, so we can use a
        # dictionary.
        match = self.compiled_rx.match(typename)
        if not match:
            return None

        basename = match.group(1)
        if basename in self.lookup:
            return self.lookup[basename].invoke(val)

        # Cannot find a pretty printer.  Return None.
        return None

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

printers = []

def register(obj):
    "Register my pretty-printers with objfile Obj."

    if obj is None:
        obj = gdb

    for p in printers:
        obj.pretty_printers.insert(0, p)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def build_afw_dictionary():
    printer = Printer("afw")

    printer.add('lsst::afw::detection::Footprint', FootprintPrinter)
    printer.add('lsst::afw::detection::Source', SourcePrinter)
    printer.add('lsst::afw::detection::BaseSourceAttributes', BaseSourceAttributesPrinter)

    printer.add('lsst::afw::geom::Point', CoordinateBasePrinter)
    printer.add('lsst::afw::geom::Extent', CoordinateBasePrinter)

    printer.add('lsst::afw::image::ImageBase', ImagePrinter)
    printer.add('lsst::afw::image::Image', ImagePrinter)
    printer.add('lsst::afw::image::Mask', ImagePrinter)
    printer.add('lsst::afw::image::MaskedImage', MaskedImagePrinter)
    printer.add('lsst::afw::image::Exposure', ExposurePrinter)

    return printer

printers.append(build_afw_dictionary())

def build_daf_base_dictionary():
    printer = Printer("daf::base")

    printer.add('lsst::daf::base::Citizen', CitizenPrinter)

    return printer

printers.append(build_daf_base_dictionary())
