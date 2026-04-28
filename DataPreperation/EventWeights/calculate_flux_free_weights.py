"""
Script to extract event properties from LeptonInjector simulation files,
calculate cross sections, and save as CSV for flux-free weight calculation.

Usage:
    python3 calculate_flux_free_weights_prepare_metadata.py --mc_name SPRING2026MC
    python3 calculate_flux_free_weights_prepare_metadata.py --mc_name STRING340MC

NOTE on cross sections:
    Although nubar-specific cross section files exist in the CrossSectionModels directory
    (sigma_nubar_CC_iso.fits, dsdxdy_nubar_CC_iso.fits, etc.), they are NOT used in this
    simulation. Both neutrinos and antineutrinos use the nu (neutrino) cross section files:
        - sigma_nu_CC_iso.fits
        - dsdxdy_nu_CC_iso.fits
        - sigma_nu_NC_iso.fits
        - dsdxdy_nu_NC_iso.fits

NOTE on units:
    InjectionRadius:  m  → cm  (× 100) applied at weight calculation
    CylinderRadius:   m  → cm  (× 100) applied at weight calculation
    CylinderHeight:   m  → cm  (× 100) applied at weight calculation
    sigma_tot:        m² → cm² (× 1e4) applied at weight calculation
    sigma_diff:       m² → cm² (× 1e4) applied at weight calculation
    TotalColumnDepth: g/cm² (no conversion needed)
    TotalEnergy:      GeV (no conversion needed)
    ImpactParameter:  cm (no conversion needed)
"""

import os
import glob
import argparse
import importlib.util
import numpy as np
import pandas as pd
import photospline
from icecube import icetray, dataio, dataclasses, LeptonInjector

# ============================================================
# Load paths
# ============================================================

spec = importlib.util.spec_from_file_location(
    "paths",
    "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
)
paths_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths_module)

SPRING2026MC_I3 = paths_module.SPRING2026MC_I3
STRING340MC_I3  = paths_module.STRING340MC_I3

# ============================================================
# Config
# ============================================================

OUTPUT_PATHS = {
    'SPRING2026MC': '/project/def-nahee/kbas/Graphnet-Applications/Metadata/EventWeights/Spring2026MC/flux_free_weight_metadata.csv',
    'STRING340MC':  '/project/def-nahee/kbas/Graphnet-Applications/Metadata/EventWeights/String340MC/flux_free_weight_metadata.csv',
}

MC_I3 = {
    'SPRING2026MC': SPRING2026MC_I3,
    'STRING340MC':  STRING340MC_I3,
}

XS_DIR = "/project/6008051/pone_simulation/pone_offline/CrossSectionModels/csms_differential_v1.0"

EVENTS_PER_FILE = 200

# ============================================================
# Load cross section splines
# ============================================================

def load_splines():
    # NOTE: nubar events use the same cross section files as nu — see module docstring.
    splines = {
        'CC': {
            'sigma_tot':  photospline.SplineTable(XS_DIR + "/sigma_nu_CC_iso.fits"),
            'sigma_diff': photospline.SplineTable(XS_DIR + "/dsdxdy_nu_CC_iso.fits"),
        },
        'NC': {
            'sigma_tot':  photospline.SplineTable(XS_DIR + "/sigma_nu_NC_iso.fits"),
            'sigma_diff': photospline.SplineTable(XS_DIR + "/dsdxdy_nu_NC_iso.fits"),
        },
    }
    return splines

def get_spline_key(final_type2):
    """Determine CC or NC based on final state particle."""
    is_NC = 'Nu' in str(final_type2)
    return 'NC' if is_NC else 'CC'

def compute_cross_sections(row, splines):
    """Compute sigma_tot and sigma_diff for a single event."""
    key        = get_spline_key(row['FinalType2'])
    log_E      = np.log10(row['TotalEnergy'])
    log_x      = np.log10(row['FinalStateX'])
    log_y      = np.log10(row['FinalStateY'])

    sigma_tot  = 10 ** splines[key]['sigma_tot'].evaluate_simple([log_E])
    sigma_diff = 10 ** splines[key]['sigma_diff'].evaluate_simple([log_E, log_x, log_y])

    return sigma_tot, sigma_diff

# ============================================================
# Helpers
# ============================================================

def get_all_files(base_path, fmt):
    if fmt == 'zst':
        ext = '*.i3.zst'
    elif fmt == 'gz':
        ext = '*.i3.gz'
    else:
        ext = '*.i3'
    pattern = os.path.join(base_path, '**', ext)
    return sorted(glob.glob(pattern, recursive=True))

def get_relative_path(filepath, base_path):
    return os.path.relpath(filepath, base_path).replace(os.sep, '/')

def compute_energy_integral(e_min, e_max, gamma):
    """
    Compute the power-law energy integral:
        integral from E_min to E_max of E^(-gamma) dE
        = (E_max^(1-gamma) - E_min^(1-gamma)) / (1 - gamma)
    Valid for gamma != 1.
    """
    assert gamma != 1.0, "gamma == 1 not supported (log integral needed)"
    return (e_max**(1 - gamma) - e_min**(1 - gamma)) / (1 - gamma)

def compute_solid_angle(zenith_min, zenith_max, azimuth_min, azimuth_max):
    """
    Compute the solid angle:
        Omega = (cos(zenith_min) - cos(zenith_max)) * (azimuth_max - azimuth_min)
    """
    return (np.cos(zenith_min) - np.cos(zenith_max)) * (azimuth_max - azimuth_min)

def sanity_check_constants(df, flavor):
    """
    Check that generation parameters are constant across all events for a given flavor.
    Prints a warning if any variation is found.
    """
    constant_cols_common = [
        'EnergyMin', 'EnergyMax', 'ZenithMin', 'ZenithMax',
        'AzimuthMin', 'AzimuthMax', 'PowerlawIndex', 'injection_mode'
    ]
    constant_cols_ranged = ['InjectionRadius', 'EndcapLength']
    constant_cols_volume = ['CylinderRadius', 'CylinderHeight']

    df_flavor = df[df['flavor'] == flavor]
    mode      = df_flavor['injection_mode'].iloc[0]

    cols = constant_cols_common
    if mode == 'ranged':
        cols = cols + constant_cols_ranged
    else:
        cols = cols + constant_cols_volume

    all_ok = True
    for col in cols:
        unique_vals = df_flavor[col].dropna().unique()
        if len(unique_vals) > 1:
            print(f"  WARNING: {col} is not constant for {flavor}! Values: {unique_vals}")
            all_ok = False

    if all_ok:
        print(f"  {flavor}: all generation parameters are constant ✓")

# ============================================================
# Extraction
# ============================================================

def extract_events(file_list, base_path, flavor, splines):
    rows = []
    for filepath in file_list:
        relative_path = get_relative_path(filepath, base_path)
        try:
            f = dataio.I3File(filepath)
            frame_idx = 0
            while f.more():
                frame = f.pop_daq()
                if frame is None:
                    continue
                if 'EventProperties' not in frame:
                    continue

                ep    = frame['EventProperties']
                props = frame['LeptonInjectorProperties']

                is_ranged = hasattr(props, 'injectionRadius')

                row = {
                    'flavor':           flavor,
                    'injection_mode':   'ranged' if is_ranged else 'volume',
                    'file':             relative_path,
                    'frame':            frame_idx,
                    'TotalEnergy':      ep.totalEnergy,
                    'Zenith':           ep.zenith,
                    'Azimuth':          ep.azimuth,
                    'FinalStateX':      ep.finalStateX,
                    'FinalStateY':      ep.finalStateY,
                    'FinalType1':       str(ep.finalType1),
                    'FinalType2':       str(ep.finalType2),
                    'InitialType':      str(ep.initialType),
                    'EnergyMin':        props.energyMinimum,
                    'EnergyMax':        props.energyMaximum,
                    'ZenithMin':        props.zenithMinimum,
                    'ZenithMax':        props.zenithMaximum,
                    'AzimuthMin':       props.azimuthMinimum,
                    'AzimuthMax':       props.azimuthMaximum,
                    'PowerlawIndex':    props.powerlawIndex,
                    'NEvents':          props.events,
                    # Ranged-only
                    'InjectionRadius':  props.injectionRadius  if is_ranged else None,
                    'EndcapLength':     props.endcapLength     if is_ranged else None,
                    'TotalColumnDepth': ep.totalColumnDepth    if is_ranged else None,
                    'ImpactParameter':  ep.impactParameter     if is_ranged else None,
                    # Volume-only
                    'CylinderRadius':   props.cylinderRadius   if not is_ranged else None,
                    'CylinderHeight':   props.cylinderHeight   if not is_ranged else None,
                }

                sigma_tot, sigma_diff = compute_cross_sections(row, splines)
                row['sigma_tot']  = sigma_tot
                row['sigma_diff'] = sigma_diff

                rows.append(row)
                frame_idx += 1

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    return pd.DataFrame(rows)

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract LeptonInjector event properties for flux-free weight calculation.',
        epilog="""
Examples:
  python3 calculate_flux_free_weights_prepare_metadata.py --mc_name SPRING2026MC
  python3 calculate_flux_free_weights_prepare_metadata.py --mc_name STRING340MC
        """
    )
    parser.add_argument(
        '--mc_name',
        required=True,
        choices=['SPRING2026MC', 'STRING340MC'],
        help='MC dataset to process'
    )
    args = parser.parse_args()

    mc_dict  = MC_I3[args.mc_name]
    out_path = OUTPUT_PATHS[args.mc_name]
    geometry = mc_dict['full_geometry']

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # --------------------------------------------------------
    # Unit conversion summary
    # --------------------------------------------------------
    print("--- Unit conversions assumed ---")
    print("  InjectionRadius:  m  → cm  (× 100) applied at weight calculation")
    print("  CylinderRadius:   m  → cm  (× 100) applied at weight calculation")
    print("  CylinderHeight:   m  → cm  (× 100) applied at weight calculation")
    print("  sigma_tot:        m² → cm² (× 1e4)  applied at weight calculation")
    print("  sigma_diff:       m² → cm² (× 1e4)  applied at weight calculation")
    print("  TotalColumnDepth: g/cm² (no conversion)")
    print("  TotalEnergy:      GeV   (no conversion)")
    print("  ImpactParameter:  cm    (no conversion)")

    # --------------------------------------------------------
    # Load splines
    # --------------------------------------------------------
    print("\nLoading cross section splines...")
    splines = load_splines()
    print("Cross section splines loaded:")
    for key in splines:
        print(f"  {key}: sigma_tot ✓  sigma_diff ✓")

    # --------------------------------------------------------
    # Count N_gen per flavor
    # --------------------------------------------------------
    print("\n--- N_gen per flavor ---")
    n_gen = {}
    for flavor, info in geometry.items():
        base_path = info['path']
        fmt       = info['format']
        if base_path is None:
            continue
        files         = get_all_files(base_path, fmt)
        n_files       = len(files)
        n_events      = n_files * EVENTS_PER_FILE
        n_gen[flavor] = n_events
        print(f"  {flavor}: {n_files} files × {EVENTS_PER_FILE} = {n_events} events")

    # --------------------------------------------------------
    # Extract events
    # --------------------------------------------------------
    print("\n--- Extracting events ---")
    all_dfs = []
    for flavor, info in geometry.items():
        base_path = info['path']
        fmt       = info['format']

        if base_path is None:
            print(f"Skipping {flavor} — path is None")
            continue

        files = get_all_files(base_path, fmt)
        print(f"{flavor}: processing {len(files)} files...")

        df = extract_events(files, base_path, flavor, splines)
        print(f"{flavor}: {len(df)} events extracted")
        all_dfs.append(df)

    if not all_dfs:
        print("Error: no data extracted.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)

    # --------------------------------------------------------
    # Sanity check constants per flavor
    # --------------------------------------------------------
    print("\n--- Sanity check: generation parameters ---")
    for flavor in final_df['flavor'].unique():
        sanity_check_constants(final_df, flavor)

    # --------------------------------------------------------
    # Compute and print per-flavor constants
    # --------------------------------------------------------
    print("\n--- Per-flavor generation constants ---")
    for flavor in final_df['flavor'].unique():
        df_f  = final_df[final_df['flavor'] == flavor].iloc[0]
        mode  = df_f['injection_mode']

        e_min   = df_f['EnergyMin']
        e_max   = df_f['EnergyMax']
        gamma   = df_f['PowerlawIndex']
        z_min   = df_f['ZenithMin']
        z_max   = df_f['ZenithMax']
        az_min  = df_f['AzimuthMin']
        az_max  = df_f['AzimuthMax']

        energy_integral = compute_energy_integral(e_min, e_max, gamma)
        solid_angle     = compute_solid_angle(z_min, z_max, az_min, az_max)

        print(f"\n  {flavor} ({mode}):")
        print(f"    EnergyMin       = {e_min} GeV")
        print(f"    EnergyMax       = {e_max} GeV")
        print(f"    PowerlawIndex   = {gamma}")
        print(f"    Energy integral = {energy_integral:.6e} GeV^(1-gamma)")
        print(f"    ZenithMin/Max   = {z_min:.4f} / {z_max:.4f} rad")
        print(f"    AzimuthMin/Max  = {az_min:.4f} / {az_max:.4f} rad")
        print(f"    Solid angle     = {solid_angle:.6e} sr")
        print(f"    N_gen           = {n_gen[flavor]}")

        if mode == 'ranged':
            r_inj  = df_f['InjectionRadius'] * 100  # m → cm
            A_gen  = np.pi * r_inj**2
            print(f"    InjectionRadius = {df_f['InjectionRadius']} m = {r_inj} cm")
            print(f"    A_gen           = pi × r² = {A_gen:.6e} cm²")
        else:
            r_cyl  = df_f['CylinderRadius'] * 100   # m → cm
            h_cyl  = df_f['CylinderHeight'] * 100   # m → cm
            V_gen  = np.pi * r_cyl**2 * h_cyl
            print(f"    CylinderRadius  = {df_f['CylinderRadius']} m = {r_cyl} cm")
            print(f"    CylinderHeight  = {df_f['CylinderHeight']} m = {h_cyl} cm")
            print(f"    V_gen           = pi × r² × h = {V_gen:.6e} cm³")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    final_df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")
    print(f"Total shape: {final_df.shape}")

if __name__ == '__main__':
    main()

## check which cross sections are used
## check this script from scratch



# slurm script:
# apptainer shell /cvmfs/software.pacific-neutrino.org/containers/itray_v1.15.3
# sonra 2 secioz. ama slurmda nasi olcak? apptainer/1.4.5 StdEnv/2023 mi deseydik once
# source /usr/local/icetray/build/env-shell.sh
# claude'a tekrar sor iki ayri channel icin. ne eksik de event weight dogru hesaplasin. density cancel oluyo mu olmuyo mu vsvs
# makale sonuclari ile compare mi etsem?
