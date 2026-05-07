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
# Spring 2026 MC - Parquet (per flavor)
# ============================================================

SPRING2026MC_PARQUET = {
    "strings_102_40m": {
        "Muon":     {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "Electron": {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "Tau":      {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "NC":       {"train": None, "val": None, "test": None, "percentiles_csv": None},
    },
    "strings_102_80m": {
        "Muon":     {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "Electron": {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "Tau":      {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "NC":       {"train": None, "val": None, "test": None, "percentiles_csv": None},
    },
    "full_geometry": {
        "Muon":     {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "Electron": {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "Tau":      {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "NC":       {"train": None, "val": None, "test": None, "percentiles_csv": None},
    },
}


# ============================================================
# Spring 2026 MC - Parquet (mixed / multi-flavor)
# ============================================================

SPRING2026MC_PARQUET_MIXED = {
    "strings_102_40m": {
        "train": None, "val": None, "test": None, "percentiles_csv": None,
        "flavors": ["Muon", "Electron", "Tau", "NC"],
    },
    "strings_102_80m": {
        "train": None, "val": None, "test": None, "percentiles_csv": None,
        "flavors": ["Muon", "Electron", "Tau", "NC"],
    },
    "full_geometry": {
        "train": None, "val": None, "test": None, "percentiles_csv": None,
        "flavors": ["Muon", "Electron", "Tau", "NC"],
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
        "Muon":     {"path": "/home/kbas/scratch/String340MC/102_string/Muon_PMT_Response", "format": "gz"},
        "Electron": {"path": "/home/kbas/scratch/String340MC/102_string/Electron_PMT_Response", "format": "gz"},
        "Tau":      {"path": "/home/kbas/scratch/String340MC/102_string/Tau_PMT_Response", "format": "gz"},
        "NC":       {"path": "/home/kbas/scratch/String340MC/102_string/NC_PMT_Response", "format": "gz"},
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
# 340 String MC - Parquet (per flavor)
# ============================================================
# Each entry: train/val/test -> path to merged/*_reindexed dir
#             percentiles_csv -> RobustScaler percentiles CSV

STRING340MC_PARQUET = {
    "102_string": {
        "Muon":     {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "Electron": {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "Tau":      {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "NC":       {"train": None, "val": None, "test": None, "percentiles_csv": None},
    },
    "160_string": {
        "Muon":     {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "Electron": {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "Tau":      {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "NC":       {"train": None, "val": None, "test": None, "percentiles_csv": None},
    },
    "full_geometry": {
        "Muon":     {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "Electron": {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "Tau":      {"train": None, "val": None, "test": None, "percentiles_csv": None},
        "NC":       {"train": None, "val": None, "test": None, "percentiles_csv": None},
    },
}


# ============================================================
# 340 String MC - Parquet (mixed / multi-flavor)
# ============================================================
# Use when events from multiple flavors are merged into one dataset.
# "flavors" field documents which flavors were combined.

STRING340MC_PARQUET_MIXED = {
    "102_string": {
        "train": None, "val": None, "test": None, "percentiles_csv": None,
        "flavors": ["Muon", "Electron", "Tau", "NC"],
    },
    "160_string": {
        "train": None, "val": None, "test": None, "percentiles_csv": None,
        "flavors": ["Muon", "Electron", "Tau", "NC"],
    },
    "full_geometry": {
        "train": None, "val": None, "test": None, "percentiles_csv": None,
        "flavors": ["Muon", "Electron", "Tau", "NC"],
    },
}


# ============================================================
# Bad Frame List
# ============================================================




BAD_I3_FILES = {
    "String340MC": {
        "Muon": {
            "available_daq_counts": {
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_019.i3": 110,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_030.i3": 185,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_038.i3": 188,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_084.i3": 63,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_089.i3": 96,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_092.i3": 36,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1023.i3": 183,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1036.i3": 43,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1089.i3": 150,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1149.i3": 150,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_115.i3": 179,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1208.i3": 151,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1243.i3": 26,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_126.i3": 1,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1337.i3": 125,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1347.i3": 4,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1362.i3": 10,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1455.i3": 180,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1575.i3": 158,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1660.i3": 189,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1683.i3": 179,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1836.i3": 49,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1922.i3": 147,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_195.i3": 141,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_198.i3": 9,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_1988.i3": 129,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_2080.i3": 66,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_2085.i3": 49,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_2090.i3": 167,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_2097.i3": 121,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_211.i3": 185,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_224.i3": 189,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_2338.i3": 135,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_2472.i3": 62,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_2481.i3": 88,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_250.i3": 104,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_2567.i3": 80,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_2669.i3": 139,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_270.i3": 151,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_276.i3": 100,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_293.i3": 131,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_2936.i3": 142,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_294.i3": 15,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_3096.i3": 69,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_3158.i3": 147,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_323.i3": 174,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_324.i3": 120,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_3279.i3": 25,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_333.i3": 192,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_3345.i3": 93,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_336.i3": 97,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_338.i3": 100,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_339.i3": 113,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_341.i3": 3,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_3417.i3": 178,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_342.i3": 182,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_3424.i3": 16,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_3493.i3": 116,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_350.i3": 77,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_354.i3": 104,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_355.i3": 72,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_3691.i3": 106,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_3708.i3": 43,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_372.i3": 155,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_378.i3": 106,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_3782.i3": 153,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_3836.i3": 159,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_386.i3": 120,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_3911.i3": 133,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_3972.i3": 59,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_404.i3": 141,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4072.i3": 113,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4082.i3": 186,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_412.i3": 140,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4230.i3": 151,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4231.i3": 49,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_434.i3": 193,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4341.i3": 109,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4346.i3": 63,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4397.i3": 62,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_443.i3": 154,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_446.i3": 177,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_447.i3": 176,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4476.i3": 167,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4485.i3": 127,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4517.i3": 11,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4543.i3": 87,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_457.i3": 174,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4589.i3": 106,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_459.i3": 93,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4630.i3": 75,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4642.i3": 82,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_469.i3": 48,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_475.i3": 46,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_479.i3": 66,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4799.i3": 151,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4803.i3": 40,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_481.i3": 119,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4884.i3": 24,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_490.i3": 42,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_491.i3": 173,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_495.i3": 61,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_497.i3": 16,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4975.i3": 189,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_499.i3": 158,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_4991.i3": 141,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_5033.i3": 73,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_508.i3": 169,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_5099.i3": 179,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_513.i3": 70,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_514.i3": 172,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_516.i3": 194,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_518.i3": 113,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_528.i3": 104,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_529.i3": 91,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_532.i3": 120,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_538.i3": 140,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_539.i3": 128,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_540.i3": 161,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_545.i3": 156,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_552.i3": 187,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_557.i3": 14,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_5588.i3": 166,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_559.i3": 128,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_5599.i3": 97,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_560.i3": 179,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_561.i3": 123,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_562.i3": 53,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_571.i3": 34,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_572.i3": 134,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_574.i3": 73,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_576.i3": 114,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_577.i3": 17,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_5779.i3": 89,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_579.i3": 16,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_584.i3": 123,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_5863.i3": 7,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_588.i3": 43,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_590.i3": 50,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_591.i3": 18,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_5928.i3": 149,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_594.i3": 77,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_596.i3": 134,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_597.i3": 146,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_598.i3": 2,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_600.i3": 5,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_601.i3": 80,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_602.i3": 95,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_603.i3": 163,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_604.i3": 54,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_606.i3": 54,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6079.i3": 15,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6083.i3": 104,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_610.i3": 48,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_613.i3": 63,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_614.i3": 102,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6146.i3": 148,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6148.i3": 101,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6166.i3": 68,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_617.i3": 52,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_618.i3": 22,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_620.i3": 56,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6202.i3": 166,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6209.i3": 102,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_621.i3": 1,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_622.i3": 25,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_623.i3": 62,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_625.i3": 102,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_627.i3": 90,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6458.i3": 5,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6559.i3": 97,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6797.i3": 170,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6818.i3": 39,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6828.i3": 175,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6844.i3": 198,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6855.i3": 126,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6932.i3": 123,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_7114.i3": 142,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_7139.i3": 131,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_7366.i3": 53,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_7420.i3": 117,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_7518.i3": 113,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_7972.i3": 47,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_7978.i3": 59,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8135.i3": 83,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8184.i3": 147,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8235.i3": 122,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8356.i3": 163,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8375.i3": 91,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8386.i3": 166,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8406.i3": 54,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_843.i3": 166,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8486.i3": 108,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8502.i3": 178,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8588.i3": 121,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8589.i3": 163,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8600.i3": 104,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8627.i3": 186,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8657.i3": 106,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8747.i3": 78,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8824.i3": 86,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8844.i3": 76,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8849.i3": 60,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_8886.i3": 167,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9062.i3": 3,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9122.i3": 71,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9143.i3": 33,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_925.i3": 114,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9288.i3": 31,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9296.i3": 15,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9335.i3": 116,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9341.i3": 42,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9345.i3": 20,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9435.i3": 46,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9470.i3": 128,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9507.i3": 42,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9536.i3": 54,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9555.i3": 99,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9656.i3": 38,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9667.i3": 170,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9669.i3": 4,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9728.i3": 14,
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_9981.i3": 21,
            },
            "no_daq_for_some_reason": {  
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_546.i3",
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_616.i3",
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_6282.i3",
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_706.i3",
                "/project/6008051/pone_simulation/MC000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim-v10/Photon/cls_7930.i3",
            },
        },
        "Electron": {
            "available_daq_counts": {
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_1150.i3.zst": 165,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_1492.i3.zst": 122,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_2418.i3.zst": 117,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_2798.i3.zst": 122,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_2959.i3.zst": 158,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_3157.i3.zst": 30,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_3491.i3.zst": 169,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_3882.i3.zst": 174,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_3930.i3.zst": 140,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_4415.i3.zst": 139,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_5257.i3.zst": 127,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_590.i3.zst": 23,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_5971.i3.zst": 30,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_6044.i3.zst": 32,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_703.i3.zst": 8,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_7084.i3.zst": 16,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_7383.i3.zst": 55,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_7504.i3.zst": 76,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_7615.i3.zst": 121,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_771.i3.zst": 49,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_8150.i3.zst": 127,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_8164.i3.zst": 152,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_8222.i3.zst": 81,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_8779.i3.zst": 18,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_880.i3.zst": 119,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_9287.i3.zst": 115,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_9361.i3.zst": 96,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_976.i3.zst": 133,
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_989.i3.zst": 17,
            },

            "no_daq_for_some_reason": { 
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_1080.i3.zst",
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_1641.i3.zst",
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_1687.i3.zst",
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_354.i3.zst",
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_5235.i3.zst",
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_6428.i3.zst",
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_6629.i3.zst",
                "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_9619.i3.zst",
            },
        },
        "Tau": {
            "available_daq_counts": {
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_1014.i3.zst": 107,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_1243.i3.zst": 106,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_1562.i3.zst": 144,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_2320.i3.zst": 76,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_3114.i3.zst": 11,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_3232.i3.zst": 182,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_3656.i3.zst": 129,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_3660.i3.zst": 12,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_3959.i3.zst": 121,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_4229.i3.zst": 137,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_4666.i3.zst": 15,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_5344.i3.zst": 89,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_5448.i3.zst": 41,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_5478.i3.zst": 94,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_5776.i3.zst": 123,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_5779.i3.zst": 122,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_5909.i3.zst": 59,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_6021.i3.zst": 132,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_6566.i3.zst": 87,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_6879.i3.zst": 8,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_7185.i3.zst": 58,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_7203.i3.zst": 76,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_7456.i3.zst": 23,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_7460.i3.zst": 95,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_7910.i3.zst": 115,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_8175.i3.zst": 109,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_8516.i3.zst": 14,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_8917.i3.zst": 55,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_9061.i3.zst": 44,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_9249.i3.zst": 31,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_9369.i3.zst": 144,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_9422.i3.zst": 63,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_9519.i3.zst": 164,
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_9675.i3.zst": 169,
            },
            "no_daq_for_some_reason": {
                "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator/gen_3950.i3.zst"
            },
        },
        "NC": {
            "no_daq_for_some_reason": {
                "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator/gen_2335.i3.zst",
                "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator/gen_9055.i3.zst",
            },
            "available_daq_counts": {
                "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator/gen_2136.i3.zst": 54,
                "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator/gen_2761.i3.zst": 120,
                "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator/gen_3952.i3.zst": 151,
                "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator/gen_4193.i3.zst": 51,
                "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator/gen_4613.i3.zst": 71,
                "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator/gen_6835.i3.zst": 111,
                "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator/gen_7077.i3.zst": 120,
                "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator/gen_7183.i3.zst": 30,
                "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator/gen_7966.i3.zst": 184,
                "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator/gen_8410.i3.zst": 50,
                "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator/gen_8750.i3.zst": 54,
                "/project/6008051/pone_simulation/MC000005-nu_NC-2_7-LeptonInjector_PROPOSAL_clsim_NC-v10/Generator/gen_9656.i3.zst": 73,
            },
        },
    }
}



# aga bu tau icinde event weight hesaplanamamis olanlar var :ooooo 
# onlara bakmam gerek. gerekirse onlari da buraya almaliyim.

