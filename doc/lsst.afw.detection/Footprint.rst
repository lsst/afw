###############################################################
Using lsst.afw.detection.Footprint to represent detection areas
###############################################################

The fundamental unit of a detection in the LSST pipeline is an instance of the
``Footprint`` class. This class contains the x, y locations for pixels that are
part of a detection, as well as the x, y location and intensity of pixels which
are considered local peaks within the detection area.

Spans
=====

Internally the pixel locations of a detection are stored in an instance of a
``SpanSet`` which can be retrieved through the ``getSpans`` function (or the
Footprint.spans property in Python). The ``SpanSet`` provides methods for
working with the detected area as a mathematical set. Of important note is that
``SpanSet``\s are immutable once created. If the area of a ``Footprint`` needs
to be updated after initialization a new ``SpanSet`` must be created which will
then replace the original ``SpanSet`` held by the ``Footprint`` instance. This
can be done by calling the ``setSpan``\s method (or assigning to the
Footprint.spans property in Python). This does not modify the ``PeakCatalog``,
and there may be peaks which no longer fall inside the footprint. Call
``removeOrphanPeaks`` to remove these peaks. Below modifying a ``SpanSet`` is
demonstrated.

.. code-block:: python

    from lsst.afw.geom import SpanSet
    from lsst.afw.detection import Footprint

    # Create a SpanSet and use it to construct a Footprint
    radius = 5
    ss = SpanSet.fromShape(radius, offset=(10, 10))
    foot = Footprint(ss)

    # Demo the SpanSet is the same
    assert(ss == foot.spans)

    # Create a new SpanSet and assign it to the Footprint
    newRadius = 4
    ss2 = SpanSet.fromShape(newRadius, offset=(7, 7))
    foot.spans = ss2

    # Show that the SpanSet held by the Footprint is no longer the original
    assert(ss != foot.spans)

Some common operations which involve calling ``SpanSet`` methods, and
possibly setting the results back to a ``Footprint``, have convenience methods
defined in the ``Footprint`` class. Unlike the ``SpanSet`` class a
``Footprint`` is mutable, such that calling the some convenience methods (e.g.
``dilate``, ``erode``) modifies the ``Footprint``, as shown below.

.. code-block:: python

    from lsst.afw.geom import SpanSet
    from lsst.afw.detection import Footprint

    # Create a SpanSet and use it to construct a Footprint
    radius = 5
    ss = SpanSet.fromShape(radius, offset=(10, 10))
    foot = Footprint(ss)

    # Grab the SpanSet back from the Footprint and show it is the same
    ssFromFoot = foot.spans
    assert(ss is ssFromFoot)

    # Modify the Footprint in place (automatically update the internal reference
    # to the SpanSet), show that they are now different
    kernelRad = 2
    foot.erode(kernelRad)
    newSsFromFoot = foot.spans
    assert(ss is not newSsFromFoot)

In cases where the location information is all that is needed, it is strongly
suggested to use a ``SpanSet`` directly and avoid the overhead of carrying
around an empty ``PeakCatalog``. The constructors for a ``Footprint`` have been
structured to remind users of this by first requiring a SpanSet to be created.

Peaks
=====

Information on the location and intensity of the detected peaks within a
``Footprint`` are kept in a ``PeakCatalog`` which can be retrieved with the
``getPeaks`` method (or the .peaks property within Python). Like with a
``Footprint``'s ``SpanSet`` it is possible to set a ``Footprint's``
``PeakCatalog``. Unlike with the ``SpanSet`` member it is only possible to set a
new ``PeakCatalog`` if the existing catalog is empty. It is also possible to
change the schema that defines the ``PeakCatalog`` but the existing catalog must
be empty for this operation as well. Another asymmetry between the ``SpanSet``
and ``PeakCatalog`` members is in the behavior of the Python property accessor.
In Python the .spans property is both readable and writable, while the .peaks
property is read only. This behavior is intended to make a programmer cognisant
of the fact that the ``PeakCatalog`` can only be set if it is currently empty.
The ``PeakCatalog`` can be populated by calling methods on the catalog itself,
or with the convenience method ``addPeak`` supplied with the ``Footprint``
class. An additional convince method ``sortPeaks`` is also provided to increase
the ease of sorting the catalog. The following is an example of using a
``Footprint``'s ``PeakCatalog``.

.. code-block:: python

    from lsst.afw.geom import SpanSet
    from lsst.afw.detection import Footprint

    # Create a Footprint from a SpanSet
    radius = 5
    ss = SpanSet.fromShape(radius, offset=(10, 10))
    foot = Footprint(ss)

    # Add a few peaks to the PeakCatalog (x, y, intensity)
    foot.addPeak(7, 7, 95)
    foot.addPeak(8, 8, 103)
    foot.addPeak(9, 9, 100)

    # Sort the peaks according to the intensity
    foot.sortPeaks()

    # Print the peaks in the Footprint
    for peak in foot.peaks:
        print(peak.getPeakValue())

    # Output:
    # 103.0
    # 100.0
    # 95.0

Regions
=======

The ``Footprint`` class also contains a few miscellaneous data members and
methods unrelated to the main data containers mentioned above. One such data
member is the``region`` which defines the boundary of the image in which the
detection was made. This property can be retrieved or set with ``getRegion``
and ``setRegion`` respectively.

Transformations
===============

A method named ``transform`` is provided which operates on both the ``SpanSet``
and ``PeakCatalog`` transforming the x, y values into a new coordinate system.
The transform method returns a newly
created Footprint.

Handling discontinuous Footprints
=================================

To split apart a footprint which may have a discontinuous area into continuous
regions which contain only peaks which fall in the region use the ``split``
method. Occasionally, as mentioned above, operations on the ``SpanSet`` may
create an area which no longer contains the x, y locations of peaks in the
``PeakCatalog``. When this occurs the ``removeOrphanPeaks`` may be used to trim
peaks which fall outside the new area.
