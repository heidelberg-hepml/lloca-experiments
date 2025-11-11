import torch

import experiments.eventgen.transforms as tr


class BaseCoordinates:
    """
    Class that implements transformations
    from fourmomenta to an abstract set of variables
    Heavily uses functionality from Transforms classes
    """

    def __init__(self):
        self.contains_phi = False
        self.contains_mass = False
        self.transforms = []

    def init_fit(self, fourmomenta):
        # only does something for StandardNormal()
        # requires that StandardNormal() comes last in self.transforms
        x = fourmomenta.clone()
        for transform in self.transforms[:-1]:
            x = transform.forward(x)
        self.transforms[-1].init_fit(x)

    def init_unit(self, particles_list):
        self.transforms[-1].init_unit(particles_list)

    def fourmomenta_to_x(self, a):
        assert torch.isfinite(a).all()
        for transform in self.transforms:
            a = transform.forward(a)
        return a

    def x_to_fourmomenta(self, a):
        assert torch.isfinite(a).all()
        for transform in self.transforms[::-1]:
            a = transform.inverse(a)
        return a

    def velocity_fourmomenta_to_x(self, v, a):
        assert torch.isfinite(a).all() and torch.isfinite(v).all()
        for transform in self.transforms:
            b = transform.forward(a)
            v = transform.velocity_forward(v, a, b)
            a = b
        return v, a

    def velocity_x_to_fourmomenta(self, v, a):
        assert torch.isfinite(a).all() and torch.isfinite(v).all()
        for transform in self.transforms[::-1]:
            b = transform.inverse(a)
            v = transform.velocity_inverse(v, a, b)
            a = b
        return v, a

    def logdetjac_fourmomenta_to_x(self, a):
        # logdetjac = log|da/db| = -log|db/da| with a=fourmomenta, b=x
        assert torch.isfinite(a).all()
        b = self.transforms[0].forward(a)
        logdetjac = -self.transforms[0].logdetjac_forward(a, b)
        a = b
        for transform in self.transforms[1:]:
            b = transform.forward(a)
            logdetjac -= transform.logdetjac_forward(a, b)
            a = b
        return logdetjac, a

    def logdetjac_x_to_fourmomenta(self, a):
        # logdetjac = log|da/db| = -log|db/da| with a=x, b=fourmomenta
        assert torch.isfinite(a).all()
        b = self.transforms[-1].inverse(a)
        logdetjac = -self.transforms[-1].logdetjac_inverse(a, b)
        a = b
        for transform in self.transforms[::-1][1:]:
            b = transform.inverse(a)
            logdetjac -= transform.logdetjac_inverse(a, b)
            a = b
        return logdetjac, a


class Fourmomenta(BaseCoordinates):
    # (E, px, py, pz)
    # this class effectively does nothing,
    # because fourmomenta are already the baseline representation
    def __init__(self):
        super().__init__()
        self.transforms = [tr.EmptyTransform()]


class PPPM2(BaseCoordinates):
    def __init__(self):
        super().__init__()
        self.contains_mass = True
        self.transforms = [tr.EPPP_to_PPPM2()]


class EPhiPtPz(BaseCoordinates):
    # (E, phi, pt, pz)
    def __init__(self):
        super().__init__()
        self.contains_phi = True
        self.transforms = [tr.EPPP_to_EPhiPtPz()]


class PtPhiEtaE(BaseCoordinates):
    # (pt, phi, eta, E)
    def __init__(self):
        super().__init__()
        self.contains_phi = True
        self.transforms = [tr.EPPP_to_PtPhiEtaE()]


class PtPhiEtaM2(BaseCoordinates):
    def __init__(self):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
        ]


class PPPLogM2(BaseCoordinates):
    # (px, py, pz, log(m^2))
    def __init__(self):
        super().__init__()
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PPPM2(),
            tr.M2_to_LogM2(),
        ]


class StandardPPPLogM2(BaseCoordinates):
    # fitted (px, py, pz, log(m^2))
    def __init__(self, onshell_list=[]):
        super().__init__()
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PPPM2(),
            tr.M2_to_LogM2(),
            tr.StandardNormal([], onshell_list),
        ]


class LogPtPhiEtaE(BaseCoordinates):
    # (log(pt), phi, eta, E)
    def __init__(self, pt_min, units):
        super().__init__()
        self.contains_phi = True
        self.transforms = [tr.EPPP_to_PtPhiEtaE(), tr.Pt_to_LogPt(pt_min, units)]


class PtPhiEtaLogM2(BaseCoordinates):
    # (pt, phi, eta, log(m^2))
    def __init__(self):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_LogM2(),
        ]


class LogPtPhiEtaM2(BaseCoordinates):
    # (log(pt), phi, eta, m^2)
    def __init__(self, pt_min, units):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min, units),
        ]


class LogPtPhiEtaLogM2(BaseCoordinates):
    # (log(pt), phi, eta, log(m^2)
    def __init__(self, pt_min, units):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min, units),
            tr.M2_to_LogM2(),
        ]


class StandardLogPtPhiEtaLogM2(BaseCoordinates):
    # Fitted (log(pt), phi, eta, log(m^2)
    def __init__(self, pt_min, units, onshell_list=[]):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min, units),
            tr.M2_to_LogM2(),
            tr.StandardNormal([1, 2], onshell_list),
        ]
