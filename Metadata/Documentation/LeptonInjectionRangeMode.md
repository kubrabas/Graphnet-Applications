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

### Direction

### PCA and Vertex Position

## EventProperties
