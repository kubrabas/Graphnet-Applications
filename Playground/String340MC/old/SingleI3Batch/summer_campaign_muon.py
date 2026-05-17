import os
import numpy as np
import photospline
from icecube import icetray, dataio, LeptonInjector

icetray.I3Logger.global_logger = icetray.I3NullLogger()

raw_file = "/project/6008051/pone_simulation/MC000008-nu_mu-2_6-LeptonInjector_PROPOSAL_clsim-v17.1/Generator/31003917/gen_001.i3.zst"
xs_dir   = os.getenv("PONESRCDIR") + "/CrossSectionModels/csms_differential_v1.0"

# Cross section spline'larini yukle
spline_nu    = photospline.SplineTable(xs_dir + "/sigma_nu_CC_iso.fits")
spline_nubar = photospline.SplineTable(xs_dir + "/sigma_nubar_CC_iso.fits")

def total_xs_cm2(energy_GeV, is_nubar=False):
    """Returns total CC cross section in cm²"""
    spline = spline_nubar if is_nubar else spline_nu
    log10_xs = spline.evaluate_simple([np.log10(energy_GeV)])
    return 10**log10_xs

def one_weight(ep, props):
    E      = ep.totalEnergy          # GeV
    zenith = ep.zenith               # radians
    col_depth = ep.totalColumnDepth  # g/cm²

    gamma    = props.powerlawIndex
    E_min    = props.energyMinimum   # GeV
    E_max    = props.energyMaximum   # GeV
    N_events = props.events
    R_inj    = props.injectionRadius * 1e2   # m -> cm
    L_endcap = props.endcapLength    * 1e2   # m -> cm

    # Enerji faktoru
    energy_factor = (E**gamma) * (E_max**(1-gamma) - E_min**(1-gamma)) / (1 - gamma)

    # Solid angle (full sky: 0 to pi)
    solid_angle = 2 * np.pi * (np.cos(props.zenithMinimum) - np.cos(props.zenithMaximum))

    # Cross section
    is_nubar = (ep.initialType == dataclasses.I3Particle.ParticleType.NuMuBar)
    xs = total_xs_cm2(E, is_nubar=is_nubar)

    # Interaction probability
    N_A = 6.02214076e23
    m_p = 1.67262192e-24  # g
    interaction_prob = xs * col_depth * N_A / m_p

    # Injection area (cylinder projected along neutrino direction)
    A_inj = np.pi * R_inj**2 * abs(np.cos(zenith)) + 2 * R_inj * L_endcap * abs(np.sin(zenith))

    return energy_factor * solid_angle * interaction_prob * A_inj / N_events

# S-frame'den props oku
from icecube import dataclasses
f = dataio.I3File(raw_file, "r")
props = None
for frame in f:
    if frame.Stop == icetray.I3Frame.Simulation:
        props = frame["LeptonInjectorProperties"]
        break
f.close()

# Event'leri iterate et
f = dataio.I3File(raw_file, "r")
for i, frame in enumerate(f):
    if frame.Stop != icetray.I3Frame.DAQ:
        continue
    ep = frame["EventProperties"]
    ow = one_weight(ep, props)
    print(f"Event {i}: E={ep.totalEnergy:.1f} GeV  zenith={np.degrees(ep.zenith):.1f}°  OneWeight={ow:.4e} GeV cm² sr")
    if i > 5:
        break
