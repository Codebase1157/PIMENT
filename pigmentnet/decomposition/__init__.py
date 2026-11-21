"""Module A: training-free photometric decomposition.

The public entry points reproduce Algorithm 1 of the paper:

    I_s -> linearise_srgb -> dichromatic_split with gloss_white_point
        -> optical_density -> anchored_sparse_factorise
        -> leaf_relative_deviation / pigment_composition
"""

from .linearise import linearise_srgb, encode_srgb
from .dichromatic import dichromatic_split, glare_mask
from .white_point import gloss_white_point, grey_world, von_kries_correct
from .optical_density import optical_density, from_optical_density
from .factorisation import anchored_sparse_factorise, leaf_relative_deviation, pigment_composition
from .pipeline import photometric_decompose, Decomposition
from .segmentation import segment_blade

__all__ = [
    "linearise_srgb",
    "encode_srgb",
    "dichromatic_split",
    "glare_mask",
    "gloss_white_point",
    "grey_world",
    "von_kries_correct",
    "optical_density",
    "from_optical_density",
    "anchored_sparse_factorise",
    "leaf_relative_deviation",
    "pigment_composition",
    "photometric_decompose",
    "Decomposition",
    "segment_blade",
]
