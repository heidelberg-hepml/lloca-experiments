import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.utils import scatter
from torch_geometric.utils import to_dense_batch
from lgatr import embed_vector, extract_scalar

from experiments.tagging.embedding import get_tagging_features
from lloca.lframes.lframes import LFrames
from lloca.utils.utils import (
    get_ptr_from_batch,
    get_batch_from_ptr,
    get_edge_index_from_ptr,
    get_edge_attr,
)
from lloca.nn.attention import get_xformers_attention_mask
from lloca.utils.lorentz import lorentz_eye
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.lframes.nonequi_lframes import IdentityLFrames


class TaggerWrapper(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lframesnet,
        add_fourmomenta_backbone: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_fourmomenta_backbone = add_fourmomenta_backbone
        self.lframesnet = lframesnet
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

    def forward(self, embedding):
        # extract embedding
        fourmomenta_withspurions = embedding["fourmomenta"]
        scalars_withspurions = embedding["scalars"]
        global_tagging_features_withspurions = embedding["global_tagging_features"]
        batch_withspurions = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        ptr_withspurions = embedding["ptr"]

        # remove spurions from the data again and recompute attributes
        fourmomenta_nospurions = fourmomenta_withspurions[~is_spurion]
        scalars_nospurions = scalars_withspurions[~is_spurion]

        batch_nospurions = batch_withspurions[~is_spurion]
        ptr_nospurions = get_ptr_from_batch(batch_nospurions)

        scalars_withspurions = torch.cat(
            [scalars_withspurions, global_tagging_features_withspurions], dim=-1
        )
        lframes_spurions, tracker = self.lframesnet(
            fourmomenta_withspurions,
            scalars_withspurions,
            ptr=ptr_withspurions,
            return_tracker=True,
        )
        lframes_nospurions = LFrames(
            lframes_spurions.matrices[~is_spurion],
            is_global=lframes_spurions.is_global,
            det=lframes_spurions.det[~is_spurion],
            inv=lframes_spurions.inv[~is_spurion],
            is_identity=lframes_spurions.is_identity,
            device=lframes_spurions.device,
            dtype=lframes_spurions.dtype,
            shape=lframes_spurions.matrices[~is_spurion].shape,
        )

        # transform features into local frames
        fourmomenta_local_nospurions = self.trafo_fourmomenta(
            fourmomenta_nospurions, lframes_nospurions
        )
        jet_nospurions = scatter(
            fourmomenta_nospurions, index=batch_nospurions, dim=0, reduce="sum"
        ).index_select(0, batch_nospurions)
        jet_local_nospurions = self.trafo_fourmomenta(
            jet_nospurions, lframes_nospurions
        )
        local_tagging_features_nospurions = get_tagging_features(
            fourmomenta_local_nospurions,
            jet_local_nospurions,
        )

        features_local_nospurions = torch.cat(
            [scalars_nospurions, local_tagging_features_nospurions], dim=-1
        )
        if self.add_fourmomenta_backbone:
            features_local_nospurions = torch.cat(
                [features_local_nospurions, fourmomenta_local_nospurions], dim=-1
            )

        # change dtype (see embedding.py fourmomenta_float64 option)
        features_local_nospurions = features_local_nospurions.to(
            scalars_nospurions.dtype
        )
        lframes_nospurions.to(scalars_nospurions.dtype)

        return (
            features_local_nospurions,
            fourmomenta_local_nospurions,
            lframes_nospurions,
            ptr_nospurions,
            batch_nospurions,
            tracker,
        )


class AggregatedTaggerWrapper(TaggerWrapper):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aggregator = MeanAggregation()

    def extract_score(self, features, ptr):
        score = self.aggregator(features, ptr=ptr)
        return score


class BaselineTransformerWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(in_channels=self.in_channels, num_classes=self.out_channels)
        assert (
            self.lframesnet.is_global
        ), "Non-equivariant model can only handle global lframes"

    def forward(self, embedding):
        (
            features_local,
            _,
            lframes,
            ptr,
            batch,
            tracker,
        ) = super().forward(embedding)

        mask = get_xformers_attention_mask(
            batch,
            materialize=features_local.device == torch.device("cpu"),
            dtype=features_local.dtype,
        )

        # network
        outputs = self.net(
            inputs=features_local,
            attention_mask=mask,
        )

        # aggregation
        score = self.extract_score(outputs, ptr)
        return score, tracker, lframes


class BaselineGraphNetWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(in_channels=self.in_channels, num_classes=self.out_channels)
        assert (
            self.lframesnet.is_global
        ), "Non-equivariant model can only handle global lframes"

    def forward(self, embedding):
        (
            features_local,
            _,
            lframes,
            ptr,
            batch,
            tracker,
        ) = super().forward(embedding)

        edge_index = get_edge_index_from_ptr(ptr)
        # network
        outputs = self.net(x=features_local, edge_index=edge_index)

        # aggregation
        score = self.extract_score(outputs, ptr)
        return score, tracker, lframes


class BaselineParticleNetWrapper(TaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert (
            self.lframesnet.is_global
        ), "Non-equivariant model can only handle global lframes"
        self.net = net(input_dims=self.in_channels, num_classes=self.out_channels)

    def forward(self, embedding):
        (
            features_local,
            _,
            lframes,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        # ParticleNet uses L2 norm in (phi, eta) for kNN
        phieta_local = features_local[..., [4, 5]]

        phieta_local, mask = to_dense_batch(phieta_local, batch)
        features_local, _ = to_dense_batch(features_local, batch)
        phieta_local = phieta_local.transpose(1, 2)
        features_local = features_local.transpose(1, 2)
        mask = mask.unsqueeze(1)

        # network
        score = self.net(
            points=phieta_local,
            features=features_local,
            mask=mask,
        )
        return score, tracker, lframes


class BaselineParTWrapper(TaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert (
            self.lframesnet.is_global
        ), "Non-equivariant model can only handle global lframes"
        self.net = net(input_dim=self.in_channels, num_classes=self.out_channels)

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            lframes,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        features_local = features_local.transpose(1, 2)
        fourmomenta_local = fourmomenta_local.transpose(1, 2)
        mask = mask.unsqueeze(1).float()

        # network
        score = self.net(
            x=features_local,
            v=fourmomenta_local,
            mask=mask,
        )
        return score, tracker, lframes


class GraphNetWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        include_edges,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.include_edges = include_edges
        self.net = net(in_channels=self.in_channels, out_channels=self.out_channels)
        if self.include_edges:
            self.register_buffer("edge_inited", torch.tensor(False))
            self.register_buffer("edge_mean", torch.tensor(0.0))
            self.register_buffer("edge_std", torch.tensor(1.0))

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            lframes,
            ptr,
            batch,
            tracker,
        ) = super().forward(embedding)

        edge_index = get_edge_index_from_ptr(ptr)
        if self.include_edges:
            edge_attr = self.get_edge_attr(fourmomenta_local, edge_index).to(
                features_local.dtype
            )
        else:
            edge_attr = None
        # network
        outputs = self.net(
            inputs=features_local,
            lframes=lframes,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # aggregation
        score = self.extract_score(outputs, ptr)
        return score, tracker, lframes

    def get_edge_attr(self, fourmomenta, edge_index):
        edge_attr = get_edge_attr(fourmomenta, edge_index)
        if not self.edge_inited:
            self.edge_mean = edge_attr.mean().detach()
            self.edge_std = edge_attr.std().clamp(min=1e-5).detach()
            self.edge_inited = torch.tensor(True, device=edge_attr.device)
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        return edge_attr.unsqueeze(-1)


class TransformerWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        use_amp=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.net = net(in_channels=self.in_channels, out_channels=self.out_channels)

    def forward(self, embedding):
        (
            features_local,
            _,
            lframes,
            ptr,
            batch,
            tracker,
        ) = super().forward(embedding)

        mask = get_xformers_attention_mask(
            batch,
            materialize=features_local.device == torch.device("cpu"),
            dtype=features_local.dtype,
        )

        # add artificial batch dimension
        features_local = features_local.unsqueeze(0)
        lframes = lframes.reshape(1, *lframes.shape)

        # network
        with torch.autocast("cuda", enabled=self.use_amp):
            outputs = self.net(
                inputs=features_local, lframes=lframes, attention_mask=mask
            )

        # aggregation
        outputs = outputs[0, ...]
        score = self.extract_score(outputs, ptr)
        return score, tracker, lframes


class ParticleNetWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(input_dims=self.in_channels, num_classes=self.out_channels)

    def forward(self, embedding):
        (
            features_local,
            _,
            lframes,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        # ParticleNet uses L2 norm in (phi, eta) for kNN
        phieta_local = features_local[..., [4, 5]]
        phieta_local, mask = to_dense_batch(phieta_local, batch)
        features_local, _ = to_dense_batch(features_local, batch)
        phieta_local = phieta_local.transpose(1, 2)
        features_local = features_local.transpose(1, 2)
        dense_lframes, _ = to_dense_batch(lframes.matrices, batch)
        dense_lframes[~mask] = (
            torch.eye(4, device=dense_lframes.device, dtype=dense_lframes.dtype)
            .unsqueeze(0)
            .expand((~mask).sum(), -1, -1)
        )

        lframes = LFrames(
            dense_lframes.view(-1, 4, 4),
            is_global=lframes.is_global,
            is_identity=lframes.is_identity,
            device=lframes.device,
            dtype=lframes.dtype,
            shape=lframes.matrices.shape,
        )
        mask = mask.unsqueeze(1)

        # network
        score = self.net(
            points=phieta_local,
            features=features_local,
            lframes=lframes,
            mask=mask,
        )
        return score, tracker, lframes


class LGATrWrapper(nn.Module):
    def __init__(
        self,
        net,
        lframesnet,
        out_channels,
        mean_aggregation=False,
        use_amp=False,
    ):
        super().__init__()
        self.use_amp = use_amp
        self.net = net(out_mv_channels=out_channels)
        self.aggregator = MeanAggregation() if mean_aggregation else None

        self.lframesnet = lframesnet  # not actually used
        assert isinstance(lframesnet, IdentityLFrames)

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = embedding["scalars"]
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20

        # handle global token
        if self.aggregator is None:
            batchsize = len(ptr) - 1
            global_idxs = ptr[:-1] + torch.arange(batchsize, device=batch.device)
            is_global = torch.zeros(
                fourmomenta.shape[0] + batchsize,
                dtype=torch.bool,
                device=ptr.device,
            )
            is_global[global_idxs] = True
            fourmomenta_buffer = fourmomenta.clone()
            fourmomenta = torch.zeros(
                is_global.shape[0],
                *fourmomenta.shape[1:],
                dtype=fourmomenta.dtype,
                device=fourmomenta.device,
            )
            fourmomenta[~is_global] = fourmomenta_buffer
            scalars_buffer = scalars.clone()
            scalars = torch.zeros(
                fourmomenta.shape[0],
                scalars.shape[1] + 1,
                dtype=scalars.dtype,
                device=scalars.device,
            )
            token_idx = torch.nn.functional.one_hot(
                torch.arange(1, device=scalars.device)
            )
            token_idx = token_idx.repeat(batchsize, 1)
            scalars[~is_global] = torch.cat(
                (
                    scalars_buffer,
                    torch.zeros(
                        scalars_buffer.shape[0],
                        token_idx.shape[1],
                        dtype=scalars.dtype,
                        device=scalars.device,
                    ),
                ),
                dim=-1,
            )
            scalars[is_global] = torch.cat(
                (
                    torch.zeros(
                        token_idx.shape[0],
                        scalars_buffer.shape[1],
                        dtype=scalars.dtype,
                        device=scalars.device,
                    ),
                    token_idx,
                ),
                dim=-1,
            )
            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)
        else:
            is_global = None

        fourmomenta = fourmomenta.unsqueeze(0).to(scalars.dtype)
        scalars = scalars.unsqueeze(0)

        mask = get_xformers_attention_mask(
            batch,
            materialize=fourmomenta.device == torch.device("cpu"),
            dtype=fourmomenta.dtype,
        )
        kwargs = {
            "attn_mask"
            if fourmomenta.device == torch.device("cpu")
            else "attn_bias": mask
        }

        mv = embed_vector(fourmomenta).unsqueeze(-2)
        s = scalars if scalars.shape[-1] > 0 else None

        with torch.autocast("cuda", enabled=self.use_amp):
            mv_outputs, _ = self.net(mv, s, **kwargs)
        out = extract_scalar(mv_outputs)[0, :, :, 0]

        if self.aggregator is not None:
            logits = self.aggregator(out, index=batch)
        else:
            logits = out[is_global]
        return logits, {}, None


class ParTWrapper(TaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        use_amp=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(
            input_dim=self.in_channels, num_classes=self.out_channels, use_amp=use_amp
        )

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            lframes,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        features_local = features_local.transpose(1, 2)
        fourmomenta_local = fourmomenta_local.transpose(1, 2)

        lframes_matrices, _ = to_dense_batch(lframes.matrices, batch)
        det, _ = to_dense_batch(lframes.det, batch)
        inv, _ = to_dense_batch(lframes.inv, batch)
        lframes_matrices[~mask] = lorentz_eye(
            lframes_matrices[~mask].shape[:-2],
            device=lframes.device,
            dtype=lframes.dtype,
        )
        lframes = LFrames(
            matrices=lframes_matrices,
            is_global=lframes.is_global,
            det=det,
            inv=inv,
            is_identity=lframes.is_identity,
            device=lframes.device,
            dtype=lframes.dtype,
            shape=lframes.matrices.shape,
        )

        mask = mask.unsqueeze(1).float()

        # network
        score = self.net(
            x=features_local,
            lframes=lframes,
            v=fourmomenta_local,
            mask=mask,
        )
        return score, tracker, lframes


class MIParTWrapper(ParTWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.lframesnet, IdentityLFrames)

    def forward(self, embedding):
        (features_local, fourmomenta_local, lframes, _, batch, tracker,) = super(
            ParTWrapper, self
        ).forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        features_local = features_local.transpose(1, 2)
        fourmomenta_local = fourmomenta_local.transpose(1, 2)
        mask = mask.unsqueeze(1).float()

        # network
        score = self.net(
            x=features_local,
            v=fourmomenta_local,
            mask=mask,
        )
        return score, tracker, lframes


class LorentzNetWrapper(nn.Module):
    def __init__(
        self,
        net,
        lframesnet,
        out_channels,
    ):
        super().__init__()
        self.net = net(n_class=out_channels)

        self.lframesnet = lframesnet  # not actually used
        assert isinstance(lframesnet, IdentityLFrames)

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = embedding["scalars"]
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20

        edge_index = get_edge_index_from_ptr(ptr)
        fourmomenta = fourmomenta.to(scalars.dtype)
        output = self.net(scalars, fourmomenta, edges=edge_index, batch=batch)
        return output, {}, None


class PELICANWrapper(nn.Module):
    def __init__(self, net, lframesnet, out_channels):
        super().__init__()
        self.net = net(out_channels=out_channels)
        self.lframesnet = lframesnet
        assert isinstance(lframesnet, IdentityLFrames)

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = embedding["scalars"]
        batch = embedding["batch"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20
        fourmomenta = fourmomenta.to(scalars.dtype)
        fourmomenta, mask = to_dense_batch(fourmomenta, batch)
        scalars, _ = to_dense_batch(scalars, batch)
        mask = mask.unsqueeze(-1)

        output = self.net(scalars, fourmomenta, mask=mask)
        return output, {}, None


class CGENNWrapper(nn.Module):
    def __init__(self, net, lframesnet, out_channels):
        super().__init__()
        self.net = net(n_outputs=out_channels)
        self.lframesnet = lframesnet
        assert isinstance(lframesnet, IdentityLFrames)

    def forward(self, embedding):
        # we mimic the CGENN wrapper of
        # https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/blob/master/models/lorentz_cggnn.py

        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = embedding["scalars"]
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]
        edge_index = get_edge_index_from_ptr(ptr)

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20
        fourmomenta = fourmomenta.to(scalars.dtype)
        zeros = torch.zeros(
            scalars.shape[0], 1, device=scalars.device, dtype=scalars.dtype
        )
        scalars = torch.cat((scalars, zeros), dim=-1)

        # pad to dense tensors
        fourmomenta, mask = to_dense_batch(fourmomenta, batch)
        scalars, _ = to_dense_batch(scalars, batch)
        batch_size, n_nodes, _ = fourmomenta.shape
        fourmomenta = fourmomenta.view(batch_size * n_nodes, -1)
        scalars = scalars.view(batch_size * n_nodes, -1)
        mask = mask.view(batch_size * n_nodes, -1)

        x = fourmomenta.unsqueeze(-2)
        i, j = edge_index
        edge_attr_x = torch.cat(
            [
                x[i],
                x[j],
                x[i] - x[j],
            ],
            dim=-2,
        )
        node_attr_x = x
        x = embed_vector(x)
        edge_attr_x = embed_vector(edge_attr_x)
        node_attr_x = embed_vector(node_attr_x)

        h = scalars
        edge_attr_h = None
        node_attr_h = h

        out = self.net(
            h=h,
            x=x,
            edge_attr_x=edge_attr_x,
            node_attr_x=node_attr_x,
            edge_attr_h=edge_attr_h,
            node_attr_h=node_attr_h,
            edges=edge_index,
            n_nodes=n_nodes,
            node_mask=mask,
        )

        return out, {}, None
