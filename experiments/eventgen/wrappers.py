import torch

from experiments.eventgen.cfm import EventCFM
from experiments.tagging.embedding import get_spurion
from lloca.framesnet.frames import Frames, InverseFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.utils import get_edge_index_from_shape
from lgatr import embed_vector, extract_vector


class CFMWrapper(EventCFM):
    def __init__(
        self,
        framesnet,
        cfm,
        odeint,
        n_particles,
        spurions,
        fourmomenta_velocity=False,
        **kwargs
    ):
        super().__init__(
            cfm,
            odeint,
            **kwargs,
        )
        self.fourmomenta_velocity = fourmomenta_velocity
        self.framesnet = framesnet
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

        self.register_buffer("particle_type", torch.arange(n_particles))
        self.scalar_dims = spurions.scalar_dims

        self.spurions = get_spurion(
            beam_reference=spurions.beam_reference,
            add_time_reference=spurions.add_time_reference,
            two_beams=spurions.two_beams,
            dtype=self.momentum_dtype,
            device="cpu",
        )
        self.n_spurions = self.spurions.shape[0]

    def init_standardization(self, fourmomenta, reduce_size=None):
        # framesnet equivectors edge_attr standardization (if applicable)
        if hasattr(self.framesnet, "equivectors") and hasattr(
            self.framesnet.equivectors, "init_standardization"
        ):
            fourmomenta_reduced = (
                fourmomenta[:reduce_size] if reduce_size is not None else fourmomenta
            )
            self.framesnet.equivectors.init_standardization(fourmomenta_reduced)

    def preprocess_velocity(self, x, t):
        t_embedding = self.t_embedding(t).expand(x.shape[0], x.shape[1], -1)
        particle_type = self.encode_particle_type(x.shape)

        if self.framesnet.is_identity:
            # shortcut
            # local frame = global frame -> nothing to do
            frames, tracker = self.framesnet(x, return_tracker=True)
            x_local = x

        else:
            # long route (also works for identity but is slower)
            fm = self.coordinates.x_to_fourmomenta(x)
            scalars = torch.cat([t_embedding, particle_type], dim=-1)

            spurions = self.spurions.to(x.device).unsqueeze(0).repeat(fm.shape[0], 1, 1)
            fm_withspurions = torch.cat((spurions, fm), dim=-2)
            scalars_zeros = torch.zeros(
                scalars.shape[0],
                self.n_spurions,
                scalars.shape[2],
                device=scalars.device,
                dtype=scalars.dtype,
            )
            scalars_withspurions = torch.cat((scalars_zeros, scalars), dim=-2)
            frames_withspurions, tracker = self.framesnet(
                fm_withspurions,
                scalars=scalars_withspurions,
                ptr=None,
                return_tracker=True,
            )
            frames = Frames(
                matrices=frames_withspurions.matrices[:, self.n_spurions :],
                det=frames_withspurions.det[:, self.n_spurions :],
                inv=frames_withspurions.inv[:, self.n_spurions :],
                is_global=frames_withspurions.is_global,
                is_identity=frames_withspurions.is_identity,
            )

            fm_local = self.trafo_fourmomenta(fm, frames)
            x_local = self.coordinates.fourmomenta_to_x(fm_local)

        # move everything to self.save_dtype
        x_local = x_local.to(self.network_dtype)
        frames.to(self.network_dtype)

        return (
            x_local,
            t_embedding,
            particle_type,
            frames,
            tracker,
        )

    def postprocess_velocity(self, v_mixed_local, x, frames):
        v_mixed_local = v_mixed_local.to(x.dtype)
        frames.to(x.dtype)
        v_fm_local, v_s_local = v_mixed_local[..., 0:4], v_mixed_local[..., 4:]

        if self.framesnet.is_identity and not self.fourmomenta_velocity:
            # network output is velocity in x-coordinates
            v_x = v_fm_local

        else:
            # network output is velocity in fourmomenta-coordinates
            v_fm = self.trafo_fourmomenta(v_fm_local, InverseFrames(frames))
            fm = self.coordinates.x_to_fourmomenta(x)

            v_x, _ = self.coordinates.velocity_fourmomenta_to_x(v_fm, fm)
            v_x[..., self.scalar_dims] = v_s_local

        v_x = v_x.to(self.momentum_dtype)
        return v_x

    def encode_particle_type(self, shape):
        particle_type = torch.nn.functional.one_hot(
            self.particle_type, num_classes=self.particle_type.max() + 1
        )
        particle_type = particle_type.unsqueeze(0).repeat(shape[0], 1, 1)
        return particle_type


class MLPCFM(CFMWrapper):
    # Not properly tested
    def __init__(self, net, **kwargs):
        super().__init__(**kwargs)
        self.net = net

    def get_velocity(self, x, t):
        (
            x_local,
            t_embedding,
            _,
            frames,
            tracker,
        ) = super().preprocess_velocity(x, t)

        x_local = x_local
        fts = torch.cat(
            [
                x_local.reshape(*x_local.shape[:-2], -1),
                t_embedding[..., 0, :],
            ],
            dim=-1,
        )
        v_local = self.net(fts)
        v_local = v_local.reshape(*x_local.shape[:-1], -1)
        v = self.postprocess_velocity(v_local, x, frames)
        return v, tracker


class TransformerCFM(CFMWrapper):
    def __init__(self, net, **kwargs):
        super().__init__(**kwargs)
        self.net = net

    def get_velocity(self, x, t):
        (
            x_local,
            t_embedding,
            particle_type,
            frames,
            tracker,
        ) = super().preprocess_velocity(x, t)

        fts = torch.cat([x_local, particle_type, t_embedding], dim=-1)
        v_local = self.net(fts, frames)
        v = self.postprocess_velocity(v_local, x, frames)
        return v, tracker


class GraphNetCFM(CFMWrapper):
    # Not properly tested
    def __init__(self, net, include_edges, **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.include_edges = include_edges
        assert not include_edges

    def get_velocity(self, x, t):
        (
            x_local,
            t_embedding,
            particle_type,
            frames,
            tracker,
        ) = super().preprocess_velocity(x, t)
        edge_index, batch = get_edge_index_from_shape(x_local.shape, x_local.device)
        fts = torch.cat([x_local, particle_type, t_embedding], dim=-1)

        fts_flat = fts.flatten(0, 1)
        frames_flat = frames.reshape(-1, *frames.shape[2:])
        v_local_flat = self.net(
            fts_flat, frames_flat, edge_index=edge_index, batch=batch
        )
        v_local = v_local_flat.reshape(*fts.shape[:-1], v_local_flat.shape[-1])

        v = self.postprocess_velocity(v_local, x, frames)
        return v, tracker


class LGATrCFM(CFMWrapper):
    def __init__(self, net, **kwargs):
        super().__init__(fourmomenta_velocity=True, **kwargs)
        self.net = net
        assert self.framesnet.is_identity

    def get_velocity(self, x, t):
        (
            _,
            t_embedding,
            particle_type,
            frames,
            tracker,
        ) = super().preprocess_velocity(x, t)

        # embed into geometric algebra
        scalars = torch.cat((particle_type, t_embedding), dim=-1)
        fm = self.coordinates.x_to_fourmomenta(x).unsqueeze(-2)
        spurions = (
            self.spurions.to(x.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(fm.shape[0], fm.shape[1], 1, 1)
        )
        vector = torch.cat((fm, spurions), dim=-2)
        multivectors = embed_vector(vector).to(scalars.dtype)

        v_mv, v_s = self.net(multivectors, scalars)
        v_v = extract_vector(v_mv).squeeze(dim=-2)
        v_pre = torch.cat((v_v, v_s), dim=-1)
        v = self.postprocess_velocity(v_pre, x, frames)
        return v, tracker
