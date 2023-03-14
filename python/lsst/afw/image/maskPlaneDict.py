import dataclasses


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class MaskPlaneDict:
    """An immutable copy of the image bit masks, relating their names, bit
    numbers, and docs.

    Instantiate with the C++ `MaskPlaneDict`, a
    `std::map<std::string, std::pair<int, std::string>>`
    """
    planes: dict
    """Map of plane name to mask bit."""

    docs: dict = None
    """Map of plane name to docstring."""

    bits: dict = None
    """Map of mask bit to plane name. The inverse map of `planes`, populated
    on init.
    """

    def __post_init__(self) -> None:
        # Set the dictionaries, even though this class is frozen.
        object.__setattr__(self, "_maskDict", self.planes.copy())
        object.__setattr__(self, "docs", {k: v[1] for k, v in self._maskDict.items()})
        object.__setattr__(self, "bits", {v[0]: k for k, v in self._maskDict.items()})
        object.__setattr__(self, "planes", {k: v[0] for k, v in self._maskDict.items()})

    def __contains__(self, name):
        return name in self.planes

    def __getitem__(self, name):
        return self.planes[name]

    def __iter__(self):
        for name in self.planes:
            yield name

    def __len__(self):
        return len(self.planes)

    def __str__(self):
        return "\n".join(f'bit {bit}: {name}, "{self.docs[name]}"'
                         for bit, name in sorted(self.bits.items()))

    def keys(self):
        return self.planes.keys()
