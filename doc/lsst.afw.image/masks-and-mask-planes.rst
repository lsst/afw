######################
MaskDict functionality
######################

Exposure masks consist of a bitmask array, with a mapping (`~lsst.afw.image.MaskDict`) of plane name to bit id and docstring.
In this document, the term "plane" refers both to the status of the slice of the array at that bit id, and to the bit id itself.
Users of Masks should never assume a given bit always corresponds to a particular named plane; new Tasks may add planes in a different order or remove planes.
Code needing to access particular plane(s) should only ever get those bit(s) by their name(s) from an in-memory Mask, e.g. ``bits = mask.getPlanes(["BAD", "EDGE"])``.

In typical usage of masks, the planes will be defined either during the initialization/running of a Task, or when an Exposure is read from disk.
In the first case, all of the Masks operated on in that Task will be the same.
In the second case, the mask planes in that Exposure may be different from other mask planes currently in memory.
The read-in mask planes will not affect the plane definitions of any other planes.
New Masks would be created using the `~lsst.afw.image.MaskDict` containing the desired plane and docstring definitions, or built up from either an empty `~lsst.afw.image.MaskDict` or the default one.
All mask plane addition and removal can be done on a `~lsst.afw.image.Mask` or on the `~lsst.afw.image.MaskDict` itself.
**NOTE**: Removing a plane (``maskDict.remove("NAME")``) that has bits set on the mask is not recommend; use ``mask.removeAndClearMaskPlane("NAME")`` instead to clear those bits.

All `~lsst.afw.image.MaskDict` are held by value, but use an internal shared pointer so that Mask subsets (i.e. cutouts) share the same `~lsst.afw.image.MaskDict` as the parent.
Calling `~lsst.afw.image.MaskDict.add` or `~lsst.afw.image.MaskDict.remove` to modify an existing plane has a few caveats:

  * When a supplied docstring is empty, the existing docstring is used (allowing old files to be loaded while maintaining newly-defined docstrings).
  * When the existing docstring is empty, the supplied docstring is used (allowing tasks to update old Masks with new docstrings in-place).
  * When both the existing and supplied docstrings are non-empty, raise `RuntimeError`: changing an existing mask plane docstring requires removing and re-adding (to prevent accidental modification).
  * If any other Masks share that MaskDict, `~lsst.afw.image.MaskDict.remove` will raise.

For example:

::

    mask1 = lsst.afw.image.Mask(100, 100)
    maskDict2 = lsst.afw.image.MaskDict(lsst.afw.image.Mask.getNumPlanesMax())
    mask1.maskDict == maskDict2

    # Adding a new plane:
    # Planes are now different.
    bit = mask1.addPlane("New", "docs for new")
    bit == 9
    mask1.maskDict != maskDict2

    # Add the same plane to the bare MaskDict to get them to match.
    maskDict2.add("New", "docs for new")
    mask1.maskDict == maskDict2

    # Adding a new plane with an empty docstring and then re-adding it with a docstring:
    # The formerly empty docstring is changed.
    bit = mask1.addPlane("NewEmpty", "")
    bit == 10
    cloned = mask1.maskDict.clone()
    bit = mask1.addPlane("NewEmpty", "docs for newEmpty")
    bit == 10
    mask1.maskDict != cloned
    mask1.maskDict.docs["NewEmpty"] == "docs for newEmpty"

    # Calling add on an existing plane with an empty docstring:
    # The docstring is unchanged.
    cloned = mask1.maskDict.clone()
    mask1.addPlane("New", "")
    mask1.maskDict == cloned
    mask1.maskDict.docs["New"] == "docs for new"

    # Adding an existing plane with a different docstring:
    # An exception is raised.
    mask1.addPlane("New", "different docs")  # raises RuntimeError

    # Removing from the Mask works, but not from its MaskDict;
    # this prevents a mismatch between the pixels and MaskDict definiton.
    box = lsst.geom.Box2I(lsst.geom.Point2I(10, 10), lsst.geom.Extent2I(10, 20))
    mask1.removeAndClearMaskPlane("SAT")
    "SAT" not in mask1.maskDict
    mask1.maskDict.remove("EDGE")  # raises RuntimeError

    # Trying to remove from a mask with a shared MaskDict also raises.
    cutout = mask1.subset(box)
    mask1.removeAndClearMaskPlane("EDGE")  # raises RuntimeError

Newly created Masks start with the current default MaskDict if one is not specified:

::

    default = lsst.afw.image.MaskDict(lsst.afw.image.Mask.getNumPlanesMax())
    empty = lsst.afw.image.MaskDict(lsst.afw.image.Mask.getNumPlanesMax(), default=False)
    mask1 = lsst.afw.image.Mask(100, 100)
    mask2 = lsst.afw.image.Mask(100, 100, maskDict=default)
    mask3 = lsst.afw.image.Mask(100, 100, maskDict=empty)
    mask1.maskDict == mask2.maskDict
    mask1.maskDict != mask3.maskDict


TODO: Should we mention anything about the old static/global behavior?

Use cases
=========

The expectation is that the mask planes used by a given task will not change, no matter what other tasks are running in the same process.

Task creating a brand new mask, e.g. ISR
    Task will start with either an empty MaskDict, or, more likely, a copy of the default MaskDict.
    ``Task.run`` will make a mask with the default planes, add desired new planes and their docs, and will proceed.
    There is no need to copy the MaskDict in this case.
    Possible future TODO: During ``Task.__init__``, add desired new planes and their docs.
    This MaskDict will then be used by ``Task.run`` to create an Exposure and work with its Mask.

Task starting with existing planes from one Exposure, e.g. SourceInjection, CalibrateImage
    Mask planes will be loaded from disk as part of the Exposure.
    New planes may be added during ``Task.run``; this should not need to copy the MaskDict.

Task starting with multiple identical sets of planes, e.g. AssembleCoadd
    The Task will read in many files, which should have identical MaskDict contents, but different instances, as each Exposure is loaded from disk.
    TOOD: Do we assert that they are identical, or just assume it?
    The Task will create a new set of planes, possibly starting from the read-in MaskDict; this will probably result in a copy of that read-in MaskDict.

Task starting with multiple different planes, e.g. SubtractImages
    As above, but the Task will have to determine how to merge the different planes.
    This is where a "conform" or "merge" function will be necessary.

User reading a file from disk
    File has whatever mask planes were loaded.
    User can add to the read-in MaskDict, but it does not affect any other Masks in memory.
