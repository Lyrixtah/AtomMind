import pytest

def test_torch_import():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

def test_models_import():
    from models.den import DomainExpertNetwork
    from models.gkb import GeneralKnowledgeBackbone
    from models.srm import SymbolicReasoningModule
    from models.oam import OptimizationAlgorithmModule

def test_config_import():
    from config import (
        HIDDEN_SIZE,
        NUM_ATTENTION_HEADS,
        DEN_LAYERS,
        GKB_LAYERS,
        SRM_LAYERS,
        OAM_LAYERS,
        DOMAINS
    )
