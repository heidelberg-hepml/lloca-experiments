"""ParticleNet with local frame transformations."""

"""
Paper: "ParticleNet: Jet Tagging via Particle Clouds" - https://arxiv.org/abs/1902.08570
Code: https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleNet.py

We have to do three things to build LLoCa-ParticleNet
- Specify the hidden_reps_list for hidden representations in each message-passing layer
- Pass the frames through the network to have them ready in the message-passing steps
- During the edge convolution, transform the features from the local to the global frames
  using tools from the lloca code (TensorRepsTransform, IndexSelectFrames, ChangeOfFrames)

You can use 'git diff --no-index' to compare this file with the original particlenet.py file.
"""

import torch
import torch.nn as nn

from ..reps.tensorreps import TensorReps
from ..reps.tensorreps_transform import TensorRepsTransform
from ..framesnet.frames import IndexSelectFrames, ChangeOfFrames


def change_local_frame(x_j_framej, idx, frames, trafo):
    """Transform features x_j from frame 'j' ('x_j_framej') to frame 'i' ('x_j_framei').

    Parameters
    ----------
    x_j_framej : torch.Tensor
        Input features in local frame 'j' of shape (batch_size, num_dims, num_points, k).
    idx : torch.Tensor
        Indices of the nearest neighbors in the batch of shape (batch_size*num_points*k).
    frames : Frames
        Local frames of reference for the particles, shape (num_points, 4, 4).
    trafo : TensorRepsTransform
        Transformation function to apply to the features.

    Returns
    -------
    torch.Tensor
    """
    # we use batch_size*num_points with repeats of k for idx_i, e.g. for 2 points with 3 batch and k=2,
    # idx_i becomes (0,1,2,3,4,5) -> (0,0,1,1,2,2,3,3,4,4,5,5).
    idx_i = torch.arange(
        x_j_framej.shape[2] * x_j_framej.shape[0], device=x_j_framej.device
    ).repeat_interleave(
        x_j_framej.shape[-1]
    )  # identity (batch, num_points*k)
    idx_j = idx  # indices from knn (batch, num_points*k)

    frames_i = IndexSelectFrames(frames, idx_i)
    frames_j = IndexSelectFrames(frames, idx_j)
    trafo_j_to_i = ChangeOfFrames(
        frames_j, frames_i
    )  # convention: (frames_start, frames_end)

    # reshape and apply trafo
    x_j_framej_2 = x_j_framej.permute(
        0, 2, 3, 1
    )  # (batch_size, num_points, k, num_dims)
    pre = x_j_framej_2.reshape(
        -1, x_j_framej_2.shape[-1]
    )  # (batch_size*num_points*k, num_dims)
    x_j_framei = trafo(pre, trafo_j_to_i)
    x_j_framei = x_j_framei.view(x_j_framej_2.shape).permute(
        0, 3, 1, 2
    )  # (batch_size, num_dims, num_points, k)
    return x_j_framei


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][
        :, :, 1:
    ]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx, frames, trafo):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(
        -1, num_dims
    )  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(
        batch_size, num_points, k, num_dims
    )  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = change_local_frame(fts, idx, frames, trafo)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx, frames, trafo):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(
        num_dims, -1
    )  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(
        num_dims, batch_size, num_points, k
    )  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)
    fts = change_local_frame(fts, idx, frames, trafo)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(
        self,
        k,
        in_reps,
        out_feats,
        batch_norm=True,
        activation=True,
        cpu_mode=False,
    ):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = (
            get_graph_feature_v2 if cpu_mode else get_graph_feature_v1
        )
        in_feat = in_reps.dim
        self.trafo = TensorRepsTransform(TensorReps(in_reps))

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(
                nn.Conv2d(
                    2 * in_feat if i == 0 else out_feats[i - 1],
                    out_feats[i],
                    kernel_size=1,
                    bias=False if self.batch_norm else True,
                )
            )

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features, frames):

        topk_indices = knn(points, self.k)
        x = self.get_graph_feature(features, self.k, topk_indices, frames, self.trafo)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNet(nn.Module):
    """ParticleNet with local frame transformations."""

    def __init__(
        self,
        input_dims,
        hidden_reps_list,
        num_classes,
        conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
        fc_params=[(128, 0.1)],
        use_fusion=True,
        use_fts_bn=True,
        use_counts=True,
        for_inference=False,
        for_segmentation=False,
        **kwargs
    ):
        # hidden_reps_list: hidden representation for message-passing at beginning of each layer
        super(ParticleNet, self).__init__(**kwargs)
        hidden_reps_list = [TensorReps(x) for x in hidden_reps_list]
        assert input_dims == hidden_reps_list[0].dim
        assert len(hidden_reps_list) == len(conv_params)

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(hidden_reps_list[0].dim)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_reps = hidden_reps_list[idx]
            assert (
                in_reps.dim == conv_params[idx - 1][1][-1]
                if idx > 0
                else hidden_reps_list[0].dim
            )
            self.edge_convs.append(
                EdgeConvBlock(
                    k=k, in_reps=in_reps, out_feats=channels, cpu_mode=for_inference
                )
            )

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = max(128, min((in_chn // 128) * 128, 1024))
            self.fusion_block = nn.Sequential(
                nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_chn),
                nn.ReLU(),
            )

        self.for_segmentation = for_segmentation

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            if self.for_segmentation:
                fcs.append(
                    nn.Sequential(
                        nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                        nn.BatchNorm1d(channels),
                        nn.ReLU(),
                        nn.Dropout(drop_rate),
                    )
                )
            else:
                fcs.append(
                    nn.Sequential(
                        nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)
                    )
                )
        if self.for_segmentation:
            fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        else:
            fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, points, features, frames, mask=None):
        #         print('points:\n', points)
        #         print('features:\n', features)
        if mask is None:
            mask = features.abs().sum(dim=1, keepdim=True) != 0  # (N, 1, P)
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts, frames) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

        #         assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)

        if self.for_segmentation:
            x = fts
        else:
            if self.use_counts:
                x = fts.sum(dim=-1) / counts  # divide by the real counts
            else:
                x = fts.mean(dim=-1)

        output = self.fc(x)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        # print('output:\n', output)
        return output
