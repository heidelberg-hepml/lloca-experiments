import torch
from lgatr import embed_vector, extract_scalar
from lloca.backbone.attention_backends.xformers_attention import (
    get_xformers_attention_mask,
)
from lloca.framesnet.frames import Frames
from lloca.framesnet.nonequi_frames import IdentityFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.lorentz import lorentz_eye
from lloca.utils.utils import (
    get_batch_from_ptr,
    get_edge_attr,
    get_edge_index_from_ptr,
    get_ptr_from_batch,
)
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.utils import scatter, to_dense_batch

from experiments.tagging.embedding import get_tagging_features


class TaggerWrapper(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        framesnet,
        add_fourmomenta_backbone: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_fourmomenta_backbone = add_fourmomenta_backbone
        self.framesnet = framesnet
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

    def init_standardization(self, fourmomenta, ptr, reduce_size=None):
        # framesnet equivectors edge_attr standardization (if applicable)
        if hasattr(self.framesnet, "equivectors") and hasattr(
            self.framesnet.equivectors, "init_standardization"
        ):
            fourmomenta_reduced = (
                fourmomenta[:reduce_size] if reduce_size is not None else fourmomenta
            )
            self.framesnet.equivectors.init_standardization(fourmomenta_reduced, ptr)

    def forward(self, embedding):
        # extract embedding
        fourmomenta_withspurions = embedding["fourmomenta"]
        scalars_withspurions = embedding["scalars"]
        global_tagging_features_withspurions = embedding["global_tagging_features"]
        batch_withspurions = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        ptr_withspurions = embedding["ptr"]
        num_graphs = embedding["num_graphs"]
        nospurion_idxs = (~is_spurion).nonzero(as_tuple=False).squeeze(-1)

        # remove spurions from the data again and recompute attributes
        fourmomenta_nospurions = fourmomenta_withspurions.index_select(0, nospurion_idxs)
        scalars_nospurions = scalars_withspurions.index_select(0, nospurion_idxs)

        batch_nospurions = batch_withspurions.index_select(0, nospurion_idxs)
        ptr_nospurions = get_ptr_from_batch(batch_nospurions)
        B = ptr_nospurions.numel() - 1

        scalars_withspurions = torch.cat(
            [scalars_withspurions, global_tagging_features_withspurions], dim=-1
        )
        frames_spurions, tracker = self.framesnet(
            fourmomenta_withspurions,
            scalars_withspurions,
            ptr=ptr_withspurions,
            return_tracker=True,
            num_graphs=num_graphs,
        )
        matrices = frames_spurions.matrices.index_select(0, nospurion_idxs)
        frames_nospurions = Frames(
            matrices,
            is_global=frames_spurions.is_global,
            det=frames_spurions.det.index_select(0, nospurion_idxs),
            inv=frames_spurions.inv.index_select(0, nospurion_idxs),
            is_identity=frames_spurions.is_identity,
            device=frames_spurions.device,
            dtype=frames_spurions.dtype,
            shape=matrices.shape,
        )

        # transform features into local frames
        fourmomenta_local_nospurions = self.trafo_fourmomenta(
            fourmomenta_nospurions, frames_nospurions
        )
        jet_nospurions = scatter(
            fourmomenta_nospurions,
            index=batch_nospurions,
            dim=0,
            reduce="sum",
            dim_size=B,
        ).index_select(0, batch_nospurions)
        jet_local_nospurions = self.trafo_fourmomenta(jet_nospurions, frames_nospurions)
        local_tagging_features_nospurions = get_tagging_features(
            fourmomenta_local_nospurions,
            jet_local_nospurions,
            tagging_features="all",
        )

        features_local_nospurions = torch.cat(
            [scalars_nospurions, local_tagging_features_nospurions], dim=-1
        )
        if self.add_fourmomenta_backbone:
            features_local_nospurions = torch.cat(
                [features_local_nospurions, fourmomenta_local_nospurions], dim=-1
            )

        # change dtype (see embedding.py fourmomenta_float64 option)
        features_local_nospurions = features_local_nospurions.to(scalars_nospurions.dtype)
        frames_nospurions.to(scalars_nospurions.dtype)

        return (
            features_local_nospurions,
            fourmomenta_local_nospurions,
            frames_nospurions,
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
        B = ptr.numel() - 1
        score = self.aggregator(features, ptr=ptr, dim_size=B)
        return score


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
            frames,
            ptr,
            batch,
            tracker,
        ) = super().forward(embedding)

        edge_index = get_edge_index_from_ptr(ptr, features_local.shape, remove_self_loops=True)
        if self.include_edges:
            edge_attr = self.get_edge_attr(fourmomenta_local, edge_index).to(features_local.dtype)
        else:
            edge_attr = None
        # network
        outputs = self.net(
            inputs=features_local,
            frames=frames,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # aggregation
        score = self.extract_score(outputs, ptr)
        return score, tracker, frames

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
        mean_aggregation=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.mean_aggregation = mean_aggregation
        self.net = net(in_channels=self.in_channels, out_channels=self.out_channels)

    def forward(self, embedding):
        # precompute attention mask to avoid cudaStreamSynchronize
        # from .tolist() in get_xformers_attention_mask
        dtype = embedding["scalars"].dtype
        batch_withspurions = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        nospurion_idxs = (~is_spurion).nonzero(as_tuple=False).squeeze(-1)
        batch_nospurions = batch_withspurions.index_select(0, nospurion_idxs)
        ptr_nospurions = get_ptr_from_batch(batch_nospurions)
        ptr, batch = ptr_nospurions, batch_nospurions
        if not self.mean_aggregation:
            batchsize = len(ptr) - 1
            ptr = ptr.clone()
            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)
        mask = get_xformers_attention_mask(
            batch,
            materialize=batch.device == torch.device("cpu"),
            dtype=dtype,
        )

        (
            features_local,
            _,
            frames,
            ptr,
            batch,
            tracker,
        ) = super().forward(embedding)

        # handle global token
        if self.mean_aggregation:
            is_global = None
        else:
            # append global tokens to batch, ptr, features_local and frames
            # and keep a is_global mask for later extraction
            batchsize = len(ptr) - 1
            global_idxs = ptr[:-1] + torch.arange(batchsize, device=batch.device)
            is_global = torch.zeros(
                features_local.shape[0] + batchsize,
                dtype=torch.bool,
                device=ptr.device,
            )
            is_global[global_idxs] = True
            features_local_buffer = features_local.clone()
            features_local = torch.zeros(
                is_global.shape[0],
                *features_local.shape[1:],
                dtype=features_local.dtype,
                device=features_local.device,
            )
            features_local[~is_global] = features_local_buffer
            is_global_channel = torch.zeros(
                features_local.shape[0],
                1,
                dtype=features_local.dtype,
                device=features_local.device,
            )
            is_global_channel[is_global] = 1
            features_local = torch.cat((features_local, is_global_channel), dim=-1)

            # global token frames are identity
            matrices_new = (
                torch.eye(4, device=frames.device, dtype=frames.dtype)
                .unsqueeze(0)
                .expand(is_global.shape[0], -1, -1)
            ).clone()
            matrices_new[~is_global] = frames.matrices
            det_new = torch.ones(
                is_global.shape[0], device=frames.device, dtype=frames.dtype
            ).clone()
            det_new[~is_global] = frames.det
            inv_new = (
                torch.eye(4, device=frames.device, dtype=frames.dtype)
                .unsqueeze(0)
                .expand(is_global.shape[0], -1, -1)
            ).clone()
            inv_new[~is_global] = frames.inv
            frames = Frames(
                matrices_new,
                is_global=frames.is_global,
                det=det_new,
                inv=inv_new,
                is_identity=frames.is_identity,
                device=frames.device,
                dtype=frames.dtype,
                shape=matrices_new.shape,
            )

            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)

        # add artificial batch dimension
        features_local = features_local.unsqueeze(0)
        frames = frames.reshape(1, *frames.shape)

        # network
        kwargs = {
            "attn_mask" if features_local.device == torch.device("cpu") else "attn_bias": mask
        }
        with torch.autocast("cuda", enabled=self.use_amp):
            outputs = self.net(inputs=features_local, frames=frames, **kwargs)

        # aggregation
        outputs = outputs[0, ...]
        if self.mean_aggregation:
            score = self.extract_score(outputs, ptr)
        else:
            score = outputs[is_global]
        return score, tracker, frames


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
            frames,
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
        dense_frames, _ = to_dense_batch(frames.matrices, batch)
        dense_frames[~mask] = (
            torch.eye(4, device=dense_frames.device, dtype=dense_frames.dtype)
            .unsqueeze(0)
            .expand((~mask).sum(), -1, -1)
        )

        frames = Frames(
            dense_frames.view(-1, 4, 4),
            is_global=frames.is_global,
            is_identity=frames.is_identity,
            device=frames.device,
            dtype=frames.dtype,
            shape=frames.matrices.shape,
        )
        mask = mask.unsqueeze(1)

        # network
        score = self.net(
            points=phieta_local,
            features=features_local,
            frames=frames,
            mask=mask,
        )
        return score, tracker, frames


class LGATrWrapper(nn.Module):
    def __init__(
        self,
        net,
        framesnet,
        out_channels,
        mean_aggregation=False,
        use_amp=False,
    ):
        super().__init__()
        self.use_amp = use_amp
        self.net = net(out_mv_channels=out_channels)
        self.aggregator = MeanAggregation() if mean_aggregation else None

        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

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
            token_idx = torch.nn.functional.one_hot(torch.arange(1, device=scalars.device))
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
        kwargs = {"attn_mask" if fourmomenta.device == torch.device("cpu") else "attn_bias": mask}

        mv = embed_vector(fourmomenta).unsqueeze(-2)
        s = scalars if scalars.shape[-1] > 0 else None

        with torch.autocast("cuda", enabled=self.use_amp):
            mv_outputs, _ = self.net(mv, s, **kwargs)
        out = extract_scalar(mv_outputs)[0, :, :, 0]

        if self.aggregator is not None:
            B = ptr.numel() - 1
            logits = self.aggregator(out, index=batch, dim_size=B)
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
        self.net = net(input_dim=self.in_channels, num_classes=self.out_channels, use_amp=use_amp)

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
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

        frames_matrices, _ = to_dense_batch(frames.matrices, batch)
        det, _ = to_dense_batch(frames.det, batch)
        inv, _ = to_dense_batch(frames.inv, batch)
        frames_matrices[~mask] = lorentz_eye(
            frames_matrices[~mask].shape[:-2],
            device=frames.device,
            dtype=frames.dtype,
        )
        frames = Frames(
            matrices=frames_matrices,
            is_global=frames.is_global,
            det=det,
            inv=inv,
            is_identity=frames.is_identity,
            device=frames.device,
            dtype=frames.dtype,
            shape=frames.matrices.shape,
        )

        mask = mask.unsqueeze(1).float()

        # network
        score = self.net(
            x=features_local,
            frames=frames,
            v=fourmomenta_local,
            mask=mask,
        )
        return score, tracker, frames


class MIParTWrapper(ParTWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.framesnet, IdentityFrames)

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            _,
            batch,
            tracker,
        ) = super(
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
        return score, tracker, frames


class LorentzNetWrapper(nn.Module):
    def __init__(
        self,
        net,
        framesnet,
        out_channels,
    ):
        super().__init__()
        self.net = net(n_class=out_channels)

        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = embedding["scalars"]
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20

        edge_index = get_edge_index_from_ptr(ptr, fourmomenta.shape, remove_self_loops=True)
        fourmomenta = fourmomenta.to(scalars.dtype)
        output = self.net(scalars, fourmomenta, edges=edge_index, batch=batch)
        return output, {}, None


class PELICANWrapper(nn.Module):
    def __init__(
        self,
        net,
        framesnet,
        out_channels,
    ):
        super().__init__()
        self.net = net(out_channels=out_channels)

        self.register_buffer("edge_inited", torch.tensor(False))
        self.register_buffer("edge_mean", torch.tensor(0.0))
        self.register_buffer("edge_std", torch.tensor(1.0))

        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = embedding["scalars"]
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]
        num_graphs = embedding["num_graphs"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20

        edge_index = get_edge_index_from_ptr(ptr, fourmomenta.shape, remove_self_loops=False)
        fourmomenta = fourmomenta.to(scalars.dtype)
        edge_attr = self.get_edge_attr(fourmomenta, edge_index).to(scalars.dtype)
        output = self.net(
            in_rank2=edge_attr,
            edge_index=edge_index,
            batch=batch,
            in_rank1=scalars,
            num_graphs=num_graphs,
        )
        return output, {}, None

    def get_edge_attr(self, fourmomenta, edge_index):
        edge_attr = get_edge_attr(fourmomenta, edge_index)
        if not self.edge_inited:
            self.edge_mean = edge_attr.mean().detach()
            self.edge_std = edge_attr.std().clamp(min=1e-5).detach()
            self.edge_inited = torch.tensor(True, device=edge_attr.device)
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        return edge_attr.unsqueeze(-1)


class PELICANWrapperOfficial(nn.Module):
    def __init__(self, net, framesnet, out_channels):
        super().__init__()
        self.net = net(out_channels=out_channels)
        self.framesnet = framesnet
        assert isinstance(framesnet, IdentityFrames)

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
    def __init__(self, net, framesnet, out_channels):
        super().__init__()
        self.net = net(n_outputs=out_channels)
        self.framesnet = framesnet
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        # we mimic the CGENN wrapper of
        # https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/blob/master/models/lorentz_cggnn.py

        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = embedding["scalars"]
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]
        edge_index = get_edge_index_from_ptr(ptr, fourmomenta.shape, remove_self_loops=True)

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20
        fourmomenta = fourmomenta.to(scalars.dtype)
        zeros = torch.zeros(scalars.shape[0], 1, device=scalars.device, dtype=scalars.dtype)
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
