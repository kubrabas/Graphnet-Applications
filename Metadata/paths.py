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
            "available_daq_counts": {},
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
            "available_daq_counts": {},
        },
    }
}



# aga bu tau icinde event weight hesaplanamamis olanlar var :ooooo 
# onlara bakmam gerek. gerekirse onlari da buraya almaliyim.
# bu paths.py'yi duzenlemeliyim. her yerde duzenli yazsin 340str mi str340 mi vsvs
# bi de belli bis asamadan sonra neden bazi frame'ler okunamiyo ona bakmak istiyom.s
