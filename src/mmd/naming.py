"""Names for MuJoCo objects built from a source rig.

Every emitter uses the source's own names - a bone is called ``左腕`` in the MJCF because that is what
the model calls it. Nothing downstream needs a translation table, which is why the mapping files are
gone. This module exists so the PMX and glTF emitters share one rule.
"""
from __future__ import annotations


def unique_names(raw_names):
    """Make names unique (MuJoCo rejects duplicates) and non-empty."""
    seen = {}
    names = []
    for index, raw in enumerate(raw_names):
        name = (raw or '').strip() or 'bone_%d' % index
        count = seen.get(name, 0) + 1
        seen[name] = count
        names.append(name if count == 1 else '%s_%d' % (name, count))
    return names


def xml_attr(value):
    """Escape a name for use inside a double-quoted XML attribute."""
    return (value.replace('&', '&amp;').replace('<', '&lt;')
                 .replace('>', '&gt;').replace('"', '&quot;'))
