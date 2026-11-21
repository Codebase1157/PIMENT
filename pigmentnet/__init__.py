"""PigmentNet: gloss-referenced pigment deconvolution and symptom-concept bottleneck
for interpretable sugarcane leaf disease classification.

Rani & Garg, 2026.
"""

from .decomposition import (
    linearise_srgb,
    dichromatic_split,
    gloss_white_point,
    optical_density,
    anchored_sparse_factorise,
    leaf_relative_deviation,
    pigment_composition,
    photometric_decompose,
    Decomposition,
)
from .physaug import PhysAug, PhysAugConfig

try:
    from .models.pigmentnet import PigmentNet
    from .concepts.layer import SymptomConceptLayer
    from .head import ConceptFirstHead
except ImportError:
    # torch is optional for the decomposition and PhysAug modules.
    PigmentNet = None             # type: ignore
    SymptomConceptLayer = None    # type: ignore
    ConceptFirstHead = None       # type: ignore

__version__ = "1.0.0"
__all__ = [
    "PigmentNet",
    "PhysAug",
    "PhysAugConfig",
    "linearise_srgb",
    "dichromatic_split",
    "gloss_white_point",
    "optical_density",
    "anchored_sparse_factorise",
    "leaf_relative_deviation",
    "pigment_composition",
    "photometric_decompose",
    "Decomposition",
    "SymptomConceptLayer",
    "ConceptFirstHead",
]
