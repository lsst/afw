#
# LSST Data Management System
# Copyright 2016 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
from __future__ import absolute_import, division, print_function

import lsst.utils.tests
import difflib
from .schema import Schema

__all__ = ["assertSchemasEqual", "diffSchemas", "joinWords"]


def joinWords(items):
    """Join a sequence of words into a comma-separated, 'and'-finalized
    string with correct English syntax.
    """
    if len(items) == 1:
        result = items[0]
    elif len(items) == 2:
        result = "%s and %s" % tuple(items)
    else:
        result = "%s, and %s" % (", ".join(items[:-1]), items[-1])
    return result


def diffSchemas(schema1, schema2, flags=Schema.IDENTICAL):
    """Return a string diff of two schemas.

    Parameters
    ----------
    schema1 : :py:class:`lsst.afw.table.Schema`
        First schema to diff.  Items appearing only in this schema
        will be prefixed with "-" in the diff.
    schema2 : :py:class:`lsst.afw.table.Schema`
        Second schema to diff.  Items appearing only in this schema
        will be prefixed with "-" in the diff.
    flags : `int`
        A bitwise OR of :py:class:`lsst.afw.table.Schema.ComparisonFlags`
        indicating which features of schema items to compare.  The returned
        diff will always show all differences, but no diff will be shown if
        the only differences are not included in the flags.  Default is
        `lsst.afw.table.Schema.IDENTICAL`, which checks everything.

    Returns
    -------
    diff : `str`
        A "unified diff" string representation of the difference between the
        schemas, or an empty string if there is no difference.
    """
    result = schema1.compare(schema2, flags)
    if result == flags:
        return ""
    components = []
    if not result & Schema.EQUAL_KEYS:
        components.append("keys")
    if not result & Schema.EQUAL_NAMES:
        components.append("names")
    if not result & Schema.EQUAL_DOCS:
        components.append("docs")
    if not result & Schema.EQUAL_UNITS:
        components.append("units")
    if not result & Schema.EQUAL_ALIASES:
        components.append("aliases")
    diff = "\n".join(difflib.unified_diff(str(schema1).split("\n"), str(schema2).split("\n")))
    return "%s differ:\n%s" % (joinWords(components).capitalize(), diff)


@lsst.utils.tests.inTestCase
def assertSchemasEqual(testCase, schema1, schema2, flags=Schema.IDENTICAL):
    """Assert that two Schemas are equal.

    Generates a message from the difference between the schemas; see
    :py:func:`diffSchemas` for more information.
    """
    msg = diffSchemas(schema1, schema2, flags=flags)
    if msg:
        testCase.fail(msg)
