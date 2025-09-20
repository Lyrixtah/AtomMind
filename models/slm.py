import torch
import torch.nn as nn
from models.den import DomainExpertNetwork
from models.gkb import GeneralKnowledgeBackbone
from models.srm import SymbolicReasoningModule
from models.oam import OptimizationAlgorithmModule
from models.cen import ChatExpertNetwork
from config import HIDDEN_SIZE, NUM_ATTENTION_HEADS, DEN_LAYERS, GKB_LAYERS, SRM_LAYERS, OAM_LAYERS, DOMAINS

class SmallScientificLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.dens = nn.ModuleDict({
            domain: DomainExpertNetwork(HIDDEN_SIZE, DEN_LAYERS, NUM_ATTENTION_HEADS)
            for domain in DOMAINS
        })
        self.gkb = GeneralKnowledgeBackbone(HIDDEN_SIZE, GKB_LAYERS, NUM_ATTENTION_HEADS)
        self.srm = SymbolicReasoningModule(HIDDEN_SIZE, SRM_LAYERS, NUM_ATTENTION_HEADS)
        self.oam = OptimizationAlgorithmModule(HIDDEN_SIZE, OAM_LAYERS, NUM_ATTENTION_HEADS)
        self.cen = ChatExpertNetwork()
        self.output_layer = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, x_dict, chat_tensor=None):
        den_outputs = [self.dens[d](x_dict[d]) for d in DOMAINS]
        gkb_out = self.gkb(den_outputs)
        srm_out = self.srm(gkb_out)
        oam_out = self.oam(srm_out)

        if chat_tensor is not None:
            chat_out = self.cen(chat_tensor)
            final_out = torch.cat([oam_out, chat_out], dim=1)
        else:
            final_out = oam_out

        return self.output_layer(final_out)
