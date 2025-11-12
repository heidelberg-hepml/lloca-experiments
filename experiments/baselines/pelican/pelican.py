# agnostic PELICAN tagger
# quick and dirty approach, Jonas wrote this

import torch
from torch import nn
from .perm_equiv_models import Net2to2, Eq2to0
from .generic_layers import InputEncoder, GInvariants, MyLinear, MessageNet, BasicMLP


class PELICANOfficial(nn.Module):
    def __init__(
        self,
        num_scalars,
        out_channels,
        num_channels_1,
        num_channels_2,
        mlp_out=True,
    ):
        assert num_scalars == 0, "Scalar inputs for PELICAN not supported yet (non-trivial)"

        # use the typical model setup in the pelican repo
        num_channels_m = [[num_channels_1]] * 5
        num_channels_2to2 = [num_channels_2] * 5
        num_channels_m_out = [num_channels_1, num_channels_2]
        num_channels_out = [num_channels_1]

        super().__init__()
        embedding_dim = num_channels_m[0][
            0
        ]  # set this based on num_channels_m or num_channels_net2to2?

        # stabilizer='so13' means that no spurions are added
        # formally we should use stabilizer'so2' here (default in pelican tagger)
        # but we already add spurions before so we can use 'so13' here
        self.ginvariants = GInvariants(stabilizer="so13", irc_safe=False)
        rank1_dim = self.ginvariants.rank1_dim
        rank2_dim = self.ginvariants.rank2_dim

        self.input_encoder = InputEncoder(
            rank1_dim_multiplier=1,
            rank2_dim_multiplier=embedding_dim,
            rank1_in_dim=rank1_dim,
            rank2_in_dim=rank2_dim,
            mode="slog",
        )

        weight = torch.ones((embedding_dim, rank2_dim))  # why?
        self.linear = MyLinear(rank2_dim, embedding_dim, weight=weight)  # why???

        # TODO: explore args later
        self.net2to2 = Net2to2(
            num_channels_2to2 + [num_channels_m_out[0]],
            num_channels_m,
            factorize=True,
        )
        self.msg_2to0 = MessageNet(num_channels_m_out)
        self.agg_2to0 = Eq2to0(
            num_channels_m_out[-1],
            num_channels_out[0] if mlp_out else out_channels,
            factorize=True,
        )

        if mlp_out:
            self.mlp_out = BasicMLP(num_channels_out + [out_channels])

    def forward(self, scalars, fourmomenta, mask):
        edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        rank1_inputs, rank2_inputs, _ = self.ginvariants(fourmomenta)

        nobj = mask.sum(dim=-1, keepdim=True)

        rank2_inputs = self.linear(rank2_inputs)
        rank1_inputs, rank2_inputs = self.input_encoder(
            rank1_inputs,
            rank2_inputs,
            rank1_mask=mask,
            rank2_mask=edge_mask,
        )

        # TODO: include extra scalar inputs into rank2_inputs
        act1 = self.net2to2(rank2_inputs, mask=edge_mask, nobj=nobj)
        act2 = self.msg_2to0(act1, mask=edge_mask)
        act3 = self.agg_2to0(act2, nobj=nobj)

        prediction = self.mlp_out(act3) if self.mlp_out else act3
        return prediction
