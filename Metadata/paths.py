import os

# ============================================================
# GCD Files
# ============================================================

GCD = {
    "Spring2026MC": "/project/6008051/pone_simulation/GCD_Library/PONE_800mGrid_40mSpacing_40OMstring.i3.gz",
    "340StringMC":  "/project/6008051/pone_simulation/GCD_Library/PONE_800mGrid.i3.gz",
}

GCD_TRIMMED = {
    "Spring2026MC": {
        "strings_102_40m": "/project/def-nahee/kbas/Graphnet-Applications/Metadata/GCD/Spring2026MC/strings_102_40m.i3.gz",
        "strings_102_80m": "/project/def-nahee/kbas/Graphnet-Applications/Metadata/GCD/Spring2026MC/strings_102_80m.i3.gz",
    },
    "340StringMC": {
        "102_string": "/project/def-nahee/kbas/Graphnet-Applications/Metadata/GCD/340StringMC/102_string.i3.gz",
    },
}


# ============================================================
# LIC Files
# ============================================================


LIC = {
    "STRING340MC": {
        "Muon":     {"path": "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Generator",   "format": "lic"},
        "Electron": {"path": "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator",   "format": "lic"},
        "Tau":      {"path": "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator",  "format": "lic"},
        "NC":       {"path": "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator","format": "lic"},
    },
    "SPRING2026MC": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    }
}


# ============================================================
# Spring 2026 MC - I3 Folders
# ============================================================

SPRING2026MC_I3 = {
    "full_geometry": {
        "Muon":     {"path": "/project/6008051/pone_simulation/MC000008-nu_mu-2_6-LeptonInjector_PROPOSAL_clsim-v17.1/Generator",   "format": "zst"},
        "Electron": {"path": "/project/6008051/pone_simulation/MC000009-nu_e-2_6-LeptonInjector_PROPOSAL_clsim-v17.1/Generator",   "format": "zst"},
        "Tau":      {"path": "/project/6008051/pone_simulation/MC000010-nu_tau-2_6-LeptonInjector_PROPOSAL_clsim-v17.1/Generator",  "format": "zst"},
        "NC":       {"path": "/project/6008051/pone_simulation/MC000011-nu_NC-2_6-LeptonInjector_PROPOSAL_clsim_NC-v17.1/Generator","format": "zst"},
    },
    "strings_102_40m": {
        "Muon":     {"path": "/scratch/kbas/Spring2026MC/Strings_102_40m/Muon_I3Photons", "format": "gz"},
        "Electron": {"path": "/scratch/kbas/Spring2026MC/Strings_102_40m/Electron_I3Photons", "format": "gz"},
        "Tau":      {"path": "/scratch/kbas/Spring2026MC/Strings_102_40m/Tau_I3Photons", "format": "gz"},
        "NC":       {"path": "/scratch/kbas/Spring2026MC/Strings_102_40m/NC_I3Photons", "format": "gz"},
    },
    "strings_102_80m": {
        "Muon":     {"path": "/scratch/kbas/Spring2026MC/Strings_102_80m/Muon_I3Photons", "format": "gz"},
        "Electron": {"path": "/scratch/kbas/Spring2026MC/Strings_102_80m/Electron_I3Photons", "format": "gz"},
        "Tau":      {"path": "/scratch/kbas/Spring2026MC/Strings_102_80m/Tau_I3Photons", "format": "gz"},
        "NC":       {"path": "/scratch/kbas/Spring2026MC/Strings_102_80m/NC_I3Photons", "format": "gz"},
    },
}


# ============================================================
# Spring 2026 MC - PMT Folders
# ============================================================

SPRING2026MC_PMT = {
    "full_geometry": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
    "strings_102_40m": {
        "Muon":     {"path": "/home/kbas/scratch/Spring2026MC/Strings_102_40m/Muon_PMT_Response", "format": "gz"},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
    "strings_102_80m": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
}


# ============================================================
# 340 String MC - I3 Folders
# ============================================================

STRING340MC_I3 = {
    "full_geometry": {
        "Muon":     {"path": "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon", "format": "i3"},
        "Electron": {"path": "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator", "format": "zst"},
        "Tau":      {"path": "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator", "format": "zst"},
        "NC":       {"path": "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator", "format": "zst"},
    },
    "102_string": {
        "Muon":     {"path": "/home/kbas/scratch/String340MC/102_string/Muon_I3Photons", "format": "gz"},
        "Electron": {"path": "/home/kbas/scratch/String340MC/102_string/Electron_I3Photons", "format": "gz"},
        "Tau":      {"path": "/home/kbas/scratch/String340MC/102_string/Tau_I3Photons", "format": "gz"},
        "NC":       {"path": "/home/kbas/scratch/String340MC/102_string/NC_I3Photons", "format": "gz"},
    },
    "160_string": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
    "compact": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
    "default": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
    "expanded": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
    "large": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
    "modified": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
}


# ============================================================
# 340 String MC - PMT Folders
# ============================================================

STRING340MC_PMT = {
    "full_geometry": {
        "Muon":     {"path": "/home/kbas/scratch/String340MC/Full_Geometry/Muon_PMT_Response", "format": "gz"},
        "Electron": {"path": "/home/kbas/scratch/String340MC/Full_Geometry/Electron_PMT_Response", "format": "gz"},
        "Tau":      {"path": "/home/kbas/scratch/String340MC/Full_Geometry/Tau_PMT_Response", "format": "gz"},
        "NC":       {"path": "/home/kbas/scratch/String340MC/Full_Geometry/NC_PMT_Response", "format": "gz"},
    },
    "102_string": {
        "Muon":     {"path": "/home/kbas/scratch/String340MC/102_String/Muon_PMT_Response", "format": "gz"},
        "Electron": {"path": "/home/kbas/scratch/String340MC/102_String/Electron_PMT_Response", "format": "gz"},
        "Tau":      {"path": "/home/kbas/scratch/String340MC/102_String/Tau_PMT_Response", "format": "gz"},
        "NC":       {"path": "/home/kbas/scratch/String340MC/102_String/NC_PMT_Response", "format": "gz"},
    },
    "160_string": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
    "compact": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
    "default": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
    "expanded": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
    "large": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
    "modified": {
        "Muon":     {"path": None, "format": None},
        "Electron": {"path": None, "format": None},
        "Tau":      {"path": None, "format": None},
        "NC":       {"path": None, "format": None},
    },
}

