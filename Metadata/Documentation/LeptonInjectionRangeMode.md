# Lepton Injection Range Mode

This notebook documents how LeptonInjector works in range mode.

## Coordinate System

In range mode, the injection geometry is defined relative to the detector-coordinate origin `(0, 0, 0)`, but the interaction vertex is not sampled directly inside a fixed cylinder. Instead, after a neutrino direction is sampled, LeptonInjector defines a disk centered on the detector-coordinate origin and perpendicular to that direction. The radius of this disk is `InjectionRadius`.

The point sampled on this disk is called `pca` in the source code. It is the closest approach point of the extrapolated neutrino path to the detector-coordinate origin. The actual interaction vertex is selected later along this path using column-depth sampling.

For column-depth calculations, EarthModel converts detector coordinates to Earth-center coordinates internally. The detector-coordinate origin is shifted into the EarthModel coordinate system using `DetectorDepth`, measured downward from the ice surface. In the current runscript, `DetectorDepth = (2600 - 500) m = 2100 m`, so the detector-coordinate origin `(0, 0, 0)` is placed 2100 m below the ice surface in EarthModel.

## Injection Process

In range mode, each event is generated in the following order:

1. The injector samples the total neutrino energy from the configured energy spectrum. In the current runscript this is a power law between `10^2 GeV` and `10^6 GeV`.
2. The injector samples the neutrino direction from the configured zenith and azimuth ranges.
3. The injector samples a `pca` point on a disk of radius `InjectionRadius`, centered on the detector-coordinate origin and perpendicular to the sampled neutrino direction.
4. The injector estimates a maximum possible range for the secondary charged lepton from the sampled neutrino energy. For the current `nu_mu` charged-current run, this is a muon-range allowance; the motivation for "maximum" is described in Sampling Formulas.
5. The allowed sampling region is defined along the extrapolated neutrino path using `EndcapLength` and this maximum range allowance.
6. The total column depth is then computed over this allowed sampling region.
7. A column depth is sampled uniformly between `0` and `totalColumnDepth`. This is the amount of material the incoming neutrino traverses within the allowed sampling region before interacting. This quantity is called the traversed column depth. Note that the sampling is uniform in column depth, not necessarily uniform in physical path length.
8. EarthModel places the interaction vertex along the allowed sampling region such that, within that region, the incoming neutrino has accumulated the sampled traversed column depth at that point. In this step, EarthModel converts the column-depth coordinate into the corresponding physical position along the path.
9. The doubly differential cross-section model samples the final-state variables `x` and `y`. Here `y` sets the energy split between the outgoing muon and the hadronic system.
10. The final-state particles are constructed at the sampled vertex. For the current `nu_mu` charged-current run, the first final-state particle is either `MuMinus` or `MuPlus`, and the second is `Hadrons`.
11. A primary neutrino particle is added to the `I3MCTree`. Its type is inferred from the final-state particles: `MuMinus + Hadrons` corresponds to `NuMu`, and `MuPlus + Hadrons` corresponds to `NuMuBar`.
12. The `I3MCTree` and `EventProperties` object are written to the DAQ frame.

## Sampling Formulas

### Energy

The sampled energy is stored as `EventProperties.totalEnergy`. In the current runscript, it is drawn from a power-law spectrum proportional to `E^(-PowerLawIndex)`.

For the current runscript:

```text
PowerLawIndex = 1.5
MinimumEnergy = 10^2 GeV
MaximumEnergy = 10^6 GeV
```

so the generated neutrino energy follows `E^-1.5` between `100 GeV` and `1e6 GeV`.

### Direction

The azimuth is sampled uniformly in the configured azimuth range. For the current runscript this is the full range from `0` to `2*pi`.

The zenith is sampled by drawing uniformly in `cos(zenith)`, then converting back to `zenith`. This gives an isotropic direction distribution on the sphere. For the current runscript, the zenith range is `0` to `pi`, so `cos(zenith)` is uniform between `-1` and `1`.

### PCA Sampling

After the neutrino direction is sampled, LeptonInjector samples the `pca` point on a disk perpendicular to that direction. The disk is centered at the detector-coordinate origin and has radius `InjectionRadius`.

The `pca` point is sampled uniformly over the disk area, not uniformly in radius. This uses the same disk-sampling method as volume mode: the radial distribution is chosen so that equal-area regions of the disk have equal probability. 

For the current runscript:

```text
InjectionRadius = 1000 m
EndcapLength = 1100 m
```

The `pca` point satisfies:

```text
|pca| <= InjectionRadius
```

in the plane perpendicular to the sampled neutrino direction. This means `pca` is the closest approach point of the extrapolated neutrino path to the detector-coordinate origin.

### Vertex Position

The allowed sampling region is defined along the extrapolated neutrino path. It contains a fixed endcap segment around `pca` (`EndcapLength`, in meters), plus an upstream extension set by the maximum possible range of the secondary charged lepton (`m.w.e.`, later converted to `g/cm^2`). In the current `nu_mu` charged-current run, this range allowance is computed as a muon range from the sampled neutrino energy. It is a maximum allowance because it is calculated from the neutrino energy, not from the secondary particle energy.

The range returned by EarthModelCalculator is in meter-water-equivalent (`m.w.e.`). It is converted to column depth using:

```text
1 m.w.e. = 100 g/cm^2
```

The total column depth used for vertex sampling is:

```text
totalColumnDepth =
    column depth of the maximum secondary range
  + column depth of the segment from pca - EndcapLength * dir to pca + EndcapLength * dir
```

If the calculated sampling region would extend beyond the material available in EarthModel, `totalColumnDepth` is clipped to the physically available column depth.

The interaction point is sampled uniformly in column depth:

```text
traversedColumnDepth ~ Uniform(0, totalColumnDepth)
```

This is the column depth traversed by the incoming neutrino within the allowed sampling region before interacting. Because EarthModel density can vary along the path, uniform sampling in column depth is not necessarily uniform sampling in physical distance. EarthModel converts the sampled column-depth coordinate into the corresponding physical vertex position along the extrapolated neutrino path.

## EventProperties

In range mode (`Ranged = True`), `MultiLeptonInjector` constructs `RangedLeptonInjector` instances. For each generated DAQ frame, the injector creates a `LeptonInjector::BasicEventProperties` object and writes it to the frame with the key `EventProperties`.

| Field | Unit | How it is computed |
|---|---:|---|
| `totalEnergy` | `GeV` | Sampled total neutrino energy. For the current energy distribution, see Sampling Formulas. |
| `zenith` | `radians` | Zenith angle of the sampled neutrino direction. For the direction sampling, see Sampling Formulas. |
| `azimuth` | `radians` | Azimuth angle of the sampled neutrino direction. For the direction sampling, see Sampling Formulas. |
| `finalStateX` | unitless | Bjorken `x`, sampled by the doubly differential cross-section model. |
| `finalStateY` | unitless | Bjorken `y` / inelasticity, sampled by the doubly differential cross-section model. The outgoing muon receives approximately `(1 - y) * totalEnergy`, while the hadronic system receives approximately `y * totalEnergy`. |
| `finalType1` | enum | The type of the first injected final-state particle. In the current `nu_mu` charged-current run this is `MuMinus` for the neutrino generator and `MuPlus` for the antineutrino generator. |
| `finalType2` | enum | The type of the second injected final-state particle. In the current charged-current run this is `Hadrons`. |
| `initialType` | enum | Deduced from `FinalType1` and `FinalType2` during configuration. For `MuMinus + Hadrons`, this becomes `NuMu`; for `MuPlus + Hadrons`, this becomes `NuMuBar`. This is the interacting neutrino type at the vertex. |
| `x` | `meters` | X coordinate of the sampled interaction vertex in detector coordinates. In range mode, this vertex is placed along the extrapolated neutrino path using column-depth sampling. |
| `y` | `meters` | Y coordinate of the sampled interaction vertex in detector coordinates. |
| `z` | `meters` | Z coordinate of the sampled interaction vertex in detector coordinates. In the current runscript, the detector-coordinate origin is placed 2100 m below the ice surface in EarthModel. |
| `totalColumnDepth` | `g/cm^2` | Total column depth of the allowed sampling region. It includes the maximum secondary range allowance converted to column depth, plus the column depth of the fixed endcap segment around `pca`, with clipping if the full region is not physically available in EarthModel. |
