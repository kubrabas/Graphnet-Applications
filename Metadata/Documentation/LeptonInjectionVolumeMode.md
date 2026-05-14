# Lepton Injection Volume Mode

This notebook documents how LeptonInjector works in volume mode.

## Coordinate System

In volume mode, the injection cylinder is centered at the detector-coordinate origin `(0, 0, 0)`. The cylinder is defined by the configured `CylinderRadius` and `CylinderHeight`.

For column-depth calculations, EarthModel converts detector coordinates to Earth-center coordinates internally. The conversion is a translation where the detector-coordinate origin is shifted into the EarthModel coordinate system. The vertical position of this origin is set by `DetectorDepth`, measured downward from the ice surface. In the current runscript, `DetectorDepth = (2600 - 500) m = 2100 m`, so the detector-coordinate origin `(0, 0, 0)` is placed 2100 m below the ice surface in EarthModel. With `CylinderHeight = 1100 m`, the sampled volume spans roughly 1550 m to 2650 m below the ice surface. This coordinate conversion is used when evaluating material density and column depth.

## Injection Process

In volume mode, each event is generated in the following order:

1. The injector samples the total neutrino energy from the configured energy spectrum. In the current runscript this is a power law between `10^2 GeV` and `10^6 GeV`.
2. The injector samples the neutrino direction from the configured zenith and azimuth ranges.
3. The injector samples the interaction vertex inside the injection cylinder.
4. The doubly differential cross-section model samples the final-state variables `x` and `y`. Here `y` sets the energy split between the outgoing charged lepton and the hadronic system.
5. The final-state particles are constructed at the sampled vertex. For the current `nu_e` charged-current run, the first final-state particle is either `EMinus` or `EPlus`, and the second is `Hadrons`.
6. A primary neutrino particle is added to the `I3MCTree`. Its type is inferred from the final-state particles: `EMinus + Hadrons` corresponds to `NuE`, and `EPlus + Hadrons` corresponds to `NuEBar`.
7. The injector computes the two points where the sampled direction intersects the injection cylinder. The total column depth is then evaluated along this cylinder chord using EarthModel.
8. The `I3MCTree` and `EventProperties` object are written to the DAQ frame.

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

The zenith is sampled by drawing uniformly in `cos(zenith)`, then converting back to `zenith`. This is the correct way to sample an isotropic direction on a sphere, because equal intervals in `zenith` do not correspond to equal solid angle. For the current runscript, the zenith range is `0` to `pi`, so `cos(zenith)` is uniform between `-1` and `1`.

### Vertex Position

The vertex is sampled uniformly inside the cylinder volume. First, a point is sampled uniformly in the disk of radius `CylinderRadius`; then `z` is sampled uniformly within the cylinder height. In other words, the horizontal `(x, y)` point is uniform over the disk, and the vertical coordinate is uniform between `-CylinderHeight/2` and `+CylinderHeight/2`.

Uniform sampling over the disk does not mean sampling the radius uniformly. Since a ring at larger radius has more area, the radial distribution must be proportional to radius. Equivalently, the probability of landing inside radius `r` is proportional to the area enclosed by that radius.

For the current runscript:

```text
CylinderRadius = 900 m
CylinderHeight = 1100 m
```

so the sampled vertices satisfy:

```text
sqrt(x^2 + y^2) <= 900 m
-550 m <= z <= 550 m
```

## EventProperties

In volume mode (`Ranged = False`), `MultiLeptonInjector` constructs `VolumeLeptonInjector` instances. For each generated DAQ frame, the injector creates a `LeptonInjector::BasicEventProperties` object and writes it to the frame with the key `EventProperties`.

| Field | Unit | How it is computed |
|---|---:|---|
| `totalEnergy` | `GeV` | Sampled total neutrino energy. For the current energy distribution, see Sampling Formulas. |
| `zenith` | `radians` | Zenith angle of the sampled neutrino direction. For the direction sampling, see Sampling Formulas. |
| `azimuth` | `radians` | Azimuth angle of the sampled neutrino direction. For the direction sampling, see Sampling Formulas. |
| `finalStateX` | unitless | Sampled by the doubly differential cross-section model via `crossSection.sampleFinalState(...)`. The header describes this as Bjorken `x`. |
| `finalStateY` | unitless | Sampled by the doubly differential cross-section model via `crossSection.sampleFinalState(...)`. This is the inelasticity/Bjorken `y`: the first final-state particle gets approximately `(1 - y) * totalEnergy`, while the hadronic system gets approximately `y * totalEnergy`. |
| `finalType1` | enum | The type of the first injected final-state particle. In the current `nu_e` charged-current run this is `EMinus` for the neutrino generator and `EPlus` for the antineutrino generator. |
| `finalType2` | enum | The type of the second injected final-state particle. In the current charged-current run this is `Hadrons`. |
| `initialType` | enum | Deduced from `FinalType1` and `FinalType2` during configuration using `deduceInitialType(...)`. For `EMinus + Hadrons`, this becomes `NuE`; for `EPlus + Hadrons`, this becomes `NuEBar`. |
| `x` | `meters` | X coordinate of the sampled interaction vertex in detector coordinates, measured relative to the injection-cylinder center at `(0, 0, 0)`. |
| `y` | `meters` | Y coordinate of the sampled interaction vertex in detector coordinates, measured relative to the injection-cylinder center at `(0, 0, 0)`. |
| `z` | `meters` | Z coordinate of the sampled interaction vertex in detector coordinates, measured relative to the injection-cylinder center at `(0, 0, 0)`. In the current runscript, this origin is placed 2100 m below the ice surface in EarthModel. |
| `totalColumnDepth` | `g/cm^2` | Computed after the vertex and direction are sampled. The code finds the two intersections between the injected particle direction and the injection cylinder, then calls `earthModel->GetColumnDepthInCGS(p_entry, p_exit)`. This is the total column depth along the chord through the cylinder, not the column depth from the Earth's surface or atmosphere to the detector. |
