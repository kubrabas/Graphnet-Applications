# LeptonInjector Weights (LIW)


## Section 1: My Understanding

This section summarizes my current interpretation of the LIW weights produced
with LeptonWeighter. The interpretation is based on the LeptonWeighter source
code and on the *LeptonWeighter Design and Specifications* document quoted below
in Section 2.

In particular, Section 2 states that the flux used by the weighter as an input is
the neutrino flux **arriving** at the detector. In this analysis, I use a constant
flux. Therefore, I interpret the result as an analysis under a flux that is
constant at the detector, both in energy and in direction. If needed, a physical
atmospheric or astrophysical flux model can also be used instead. My current
question is: for effective-area studies, is using a unit flux model the correct
choice?

A useful way to read the `sum(LIW)` plots is:

```text
sum(LIW) is an expected interaction-contribution distribution, not a flux distribution.
```

A flat arriving flux does not imply a flat interaction distribution. The
interaction contribution can still depend on energy, direction, cross section,
generation geometry, and the event's `totalColumnDepth`.

![Unweighted and LIW-weighted zenith distributions](figures/unweighted_vs_liw_zenith.png)

Figure: The unweighted `cos(zenith)` distribution is approximately flat, as
expected from uniform direction generation. The LIW-weighted distribution is not
a flux histogram; it shows the expected interaction contribution under a unit
neutrino flux arriving at the detector. The plot on the right is surprising to
me, so I want to check carefully whether I am interpreting and calculating LIW
correctly.

As shown in the figure, the expected interaction contribution depends strongly on
zenith. I think this may be related to the total column depth that the neutrino
travels through the Earth. To check this interpretation, I also plotted
`cos(zenith)` versus `totalColumnDepth` below.

Why the zenith dependence of LIW feels confusing to me:

If the flux model used in LeptonWeighter were the flux outside the Earth, rather
than the flux arriving at the P-ONE detector, this zenith dependence would feel
more intuitive. However, according to the LeptonWeighter specification, this is
not the convention: the flux is the neutrino flux arriving at the detector. This
raises the following question for me.

In LeptonInjector, are we not already assuming that the neutrino has passed
through the Earth and arrived near the detector? The LeptonInjector model seems
to generate neutrino interactions in the vicinity of the detector. If that is the
case, why should the amount of column depth traveled affect the final weight? Is
the probability for a neutrino to interact really something that accumulates as
"the more material it crosses, the more likely it is to interact"?

I checked the LeptonWeighter source code. The relevant schematic formula is:

```text
P_interact = 1 - exp(-sigma_total * N_A * X)
```

Here, `X` is the total column depth.

This comes from the LeptonWeighter generator-probability source code:

```cpp
// /usr/local/LeptonWeighter/private/LeptonWeighter/Generator.cpp
return differential_xs / (1. - exp(-total_xs * number_of_targets));

double RangeGenerator::number_of_targets(Event& e) const {
    return Constants::Na * e.total_column_depth;
}

double VolumeGenerator::number_of_targets(Event& e) const {
    return Constants::Na * e.total_column_depth;
}
```

For small interaction probability, this becomes approximately:

```text
P_interact ~= sigma_total * N_A * X
```

According to this formula, LIW clearly depends on `X`. As mentioned above, I do
not yet understand this intuitively. Either I am misunderstanding the overall
logic of LeptonWeighter, or I am only misunderstanding the meaning of this
specific formula and how the column-depth dependence enters the weight.

The second plot is the effective-area plot:

![Muon effective area by local zenith band](figures/muon_effective_area_by_local_zenith_band.png)

Figure: Muon generation-level effective area as a function of energy, split by
local `cos(zenith)` band. This includes all generated events, not only triggered
or selected events. When I compare effective areas for different layouts, such as
the 70-string baseline or the 102-string ROV-constrained layout, I plan to keep
only the events triggered by each specific layout.

For an energy bin and a local zenith band, the effective area is computed as:

```text
A_eff(E_bin, cosz_band) = sum(LIW_i) / (DeltaE * DeltaOmega)
```

where `LIW_i` is the LeptonInjector weight of event `i`, `DeltaE` is the
energy-bin width, and for a local zenith band `[cosz_min, cosz_max]`:

```text
DeltaOmega = 2 * pi * (cosz_max - cosz_min)
```

### Summary (Please correct me if I am wrong)

1. A possible confusion is to think that summing LIW weights in energy or zenith
   bins gives the physical neutrino flux. This is not correct. The flux is an
   input to the weighting, not an output of summing weights.

2. LIW-weighted histograms are expected interaction-contribution histograms under
   a unit neutrino flux arriving at the detector. Equivalently:

```text
LIW_i = the interaction contribution represented by MC event i under a unit neutrino flux arriving at the detector
```

Therefore, `sum(LIW)` versus energy answers:

```text
If a unit neutrino flux arrived at P-ONE, how much interaction contribution would each energy bin produce?
```

Similarly, `sum(LIW)` versus zenith answers:

```text
If a unit neutrino flux arrived from each direction, how much interaction contribution would each zenith bin produce?
```

### My Plan (Please correct me if this does not make sense)

The goal is to compare different detector layouts. I calculated event weights
using all available events. Ideally, I think the correct pipeline is:

1. Calculate event weights from the available data and store them in a table such
   as:

```text
EventID, LIW
1, 3
2, 8
...
```

2. For a given detector layout, take the subset of events triggered by that
   layout. In other words, select the corresponding events from the table from the first step.
   Then use those selected events and their LIW values to compute the effective
   area using the formula above. Repeat this for each geometry.

3. To understand whether the comparison is statistically meaningful, also compute
   `sigma(A_eff)` for each effective-area curve. Then draw the effective-area
   curves for the compared geometries on the same plot. My current interpretation
   is that if the curves differ by more than the statistical uncertainty bands,
   then the difference in effective area is statistically meaningful.

4. I may also use Maria's figure of merit from MSU as an additional comparison
   metric.

### Some Other Helper Plots

![P-ONE zenith convention](figures/PONE.png)

Figure: The zenith convention on the x-axis should be interpreted using the
P-ONE local coordinate convention.

![Column depth versus local zenith](figures/column_depth_vs_local_zenith.png)

Figure: The event `totalColumnDepth` depends strongly on local zenith. This helps
explain why the LIW-weighted zenith distribution is not flat under a unit
arriving flux.

## LeptonWeighter Specification

The LeptonWeighter specification inside the IceTray container can be opened with:

```bash
module --force purge
module load StdEnv/2020 gcc/11.3.0 apptainer scipy-stack/2023b

apptainer exec \
  -B /cvmfs/software.pacific-neutrino.org/ \
  /cvmfs/software.pacific-neutrino.org/containers/itray_v1.17.1 \
  bash -lc 'less /usr/local/LeptonWeighter/resources/docs/specification'
```

I just copy and paste it here:

```text
Hi everyone,

   As promised on the call today, below is the specification I wrote up for the weighting software we need. This version has been updated to discuss the physics, so that it now says what to do, in addition to saying how to do it. Please let me know if anything is incorrect or unclear!

   Chris



# LeptonWeighter Design and Specifications

## Purpose and Goals

   LeptonInjector is designed to produce neutrino interaction final states in the vicinity of the detector. This is intended to decouple per-event simulation (particularly including computationally-expensive light propagation) from the simulation of neutrino propagation through the Earth. However, since LeptonInjector does not address such propagation, we need a suitable way to wire it up with another tool which does so that physics users can easily produce observational expectations. The boils down to computing a weight for each generated event according to a model of the neutrino flux arriving at the detector, produced by some other code.

   There are multiple (or at least two) software packages suitable for producing propagated fluxes (neutrino-generator and nuSQuIDS), so we want to provide users with a simpler interface which can use either internally. The resulting interface should be simple for users to deal with, i.e. not much more complex than neutrinoflux or NewNuFlux.

## Tentative Requirements

### Building
   - Various users would like to invoke this weighting code from their analysis code, and would like to do so without having to link against IceTray and its potentially large collection of dependencies. This project should be able to be built either within the IceTray build system (for users who consider it convenient to build everything in one bundle and were planning to link against IceTray anyway) or as a stand-alone library. This could be accomplished either by providing both an IceTray-compatible CMakeLists and a separate makefile for the stand-alone build, or with a sufficiently clever CMakeLists which uses IceTray infrastructure only if it is available.

### Backends

   - Both nuSQuIDS and neutrino-generator should be supported as backends
   - Neither backend should be a hard dependency if it requires using outside code. In particular, nuSQuIDS requires a C++11 compiler which some users are still not able to get conveniently. Therefore, if this library is not available when the project is configured, it should simply not be used. It may not be necessary to actually link against neutrino-generator (since its output will need to be summarized anyway), so the same handling may not be necessary in that case.
   - For the nuSQuIDS backend, it would be sufficient as a starting point to support loading an already computed nuSQuIDS object from a file, but it would also be desirable to allow the user to use an object which has been computed on the fly. The primary use case would be working with nuSQUIDSAtm<> to support fluxes with standard physics over the whole sky (or at least some angular range), but it would be nice to support other instantiations of nuSQUIDSAtm with altered physics. This might be practical only through the C++ interface, as nuSQuIDS itself does not currently have good support for dealing with altered physics from Python.
   - Since neutrino-generator propagates individual particles, it can only estimate fluxes as the aggregate of many single propagations. Some assistance (instructions, scripts) should be provided to help users do this. Since neutrino-generator's output will need to be summarized (basically, turned into a histogram), the easiest route may be to create scripts which produce this as an intermediate file or files for LeptonWeighter to read, which would allow it to be independent of neutrino-generator's (icetray's) datatypes and interfaces, so that this backend could then be included unconditionally. The data transfer format(s) between neutrino-generator (or the summary scripts) and LeptonWeighter must be defined. One possibility would be Dashi-compatible histograms stored as HDF5, as this format is reasonably simple, sensible, and can be read/written by at least two of the collaboration's several unofficial histogramming impementations (Dashi and PhysTools). However, something else might be simpler to implement.

### Front-ends

    - Users should be able to call on this code from C++ and Python.
- The core code should probably be written in C++ for best interfacing with nuSQuIDS. (Additionally, users calling from C++ would probably prefer not to link against python and start a python interpreter.)
    - If possible, the C++ interface should be simple enough that ROOT 5.x/Cint doesn't choke on it. For this use case much of nuSQuIDS backend may have to be hidden or disabled.

## Details

### Physics

    The weight for an event is a product of the following quantities:

    - The neutrino flux. Note that this is the flux of neutrinos arriving at the detector with the flavor, energy, and direction of the simulated interacting neutrino.
- The doubly-differential cross section evaluated at the properties of the simulated interaction: the incoming energy, fractional parton momentum (x), and fractional energy transfer (y)
    - Avogadro's Number
    - The total column depth in which the simulated event could have interacted
    - One over the sum of probabilities for any of the generators to have produced the event being considered.

    The generation probability for a given generator to produce a particular event is a product of:

    - The total number of events generated by that generator
    - The probability for that generator to generate an event with that energy (Only power law distributions need to be considered for LeptonInjector output)
    - The probability for that generator to generate an event with that direction (Only uniform distributions in azimuth and cosine of zenith need to be considered for LeptonInjector output, so this is one over the generation solid angle for events which are in bounds, and zero otherwise)
- The probability for that generator to generate an event with that position (Only uniform distributions need to be considered for LeptonInjector output, so this is one over the generation area for events which are in bounds, and zero otherwise)
    - The probability for that generator to generate an event with that final state type
- The probability for that generator to generate an event with that final state x and y (This is the ratio of the doubly differential cross section used by the generator evaluated at this energy, x, and y to the total cross section used by the generator evaluated at this energy)

### Generation Overlap

    It is critical that this code should be able to correctly weight collections of simulation sets covering overlapping phase space. In concept this is not difficult: For any event being weighted the generation probability from every generator must be computed and summed. However, it is important to note that LeptonInjector's 'ranged' and 'volume' generation modes are not as independent as they might initially appear. The use-case for the 'ranged' mode are types of final states which tend to produce muons, which can be seen by the detector after being produced far away. However, it must simulate _all_ ranges which can be seen, which includes events which start in the detector, thus creating overlap with 'volume' generation intended to produce events of the same type 'in' the detector. As we increasingly study different event topologies in unified analyses it needs to be possible to work with both of these at the same time. So, when weighting a 'ranged' event, it is necessary to also consider whether it is within the phase space simulated by each 'volume' generator, and vice versa. This is a key point which our existing prototype code does not, and is not suited to, address.

### LeptonInjector Generation Information

    In order to correctly evaluate a given event's generation probability, the weighter must be aware of settings of all event generators contributing to the loaded datasets. To facilitate this, LeptonInjector stores 'InjectionConfiguration' objects in S frames, and also provides a way to write these out into simple self contained files. These latter are expected to be read as input by the weighter (although it would also be useful to allow users to construct and specify generation information manually). The format of these files is described here:

    Generation information is serialized into data 'blocks' consisting of little-endian fields. Each block begins with a header equivalent to this pseudo-struct:

        BlockHeader{
uint64_t: block length
              size_t: name length
              char[name length]: block type name
              uint8_t: block type version
        }

Immediately following the block header are (block length - (17 + name length)) bytes of data, whose contents depend on the block type name and block type version. Currently the three types of blocks which are defined/used are "EnumDef", which records the defined enumerators of an enum type, "RangedInjectionConfiguration", which records the settings of a 'ranged' LeptonInjector generator, and "VolumeInjectionConfiguration", which records the settings of a 'volume' LeptonInjector generator. Their layouts are as follows:

EnumDef{
size_t: enum name length
            char[enum name length]: enum name
            uint32_t: number of enumerators
            Enumerator{
int64_t: enumerator value
             size_t: enumerator name length
             char[enumerator name length]: enumerator name
            }[number of enumerators]: enumerators
}

EnumDef is currently used to record the values of the I3Particle::ParticleType enum, in order to provide compatibility in case of future changes. LeptonInjector shall ensure that an EnumDef block has been written for this enum (or any other enum which might be used) before any block using that enum is written.

RangedInjectionConfiguration{
uint32_t: number of events
              double: minimum energy (GeV)
              double: maximum energy (GeV)
              double: energy power law index
              double: minimum azimuth angle (radians)
              double: maximum azimuth angle (radians)
              double: minimum zenith angle (radians)
              double: maximum zenith angle (radians)
              I3Particle::ParticleType: final particle type 1
              I3Particle::ParticleType: final particle type 2
              size_t: cross section data length
              char[cross section data length]: cross section data
              double: injection radius (meter)
              double: end cap length (meter)
}

VolumeInjectionConfiguration{
uint32_t: number of events
              double: minimum energy (GeV)
              double: maximum energy (GeV)
              double: energy power law index
              double: minimum azimuth angle (radians)
              double: maximum azimuth angle (radians)
              double: minimum zenith angle (radians)
              double: maximum zenith angle (radians)
              I3Particle::ParticleType: final particle type 1
              I3Particle::ParticleType: final particle type 2
              size_t: cross section data length
              char[cross section data length]: cross section data
              double: cylinder radius (meter)
              double: cylinder height (meter)
}

The cross section data stored in each InjectionConfiguration is a data blob produced by and suitable for passing back to the photospline library (a FITS file stored in memory), representing a three dimensional spline whose dimensions are:

1. log10 of incoming neutrino energy in GeV
2. log10 of fractional parton momentum, x
3. log10 of fractional energy transfer, y

The result of evaulating the spline is a cross section in cm^2.
```



## Section 3: Software Used (Just note for myself)

The LIW calculation uses **LeptonWeighter** to compute flux-free event weights for
LeptonInjector Monte Carlo events. The main calculation is implemented in
`calculate_LIW.py`. For each matched DAQ event, the script builds a
`LeptonWeighter.Event` object from the corresponding `EventProperties` stored in
the generator I3 file.

The script uses:

```python
import LeptonWeighter as LW
```

and constructs the weight calculator with:

```python
generators = LW.MakeGeneratorsFromLICFile(lic_path)
weighter = LW.Weighter(cross_sections, generators)
oneweight = weighter.get_oneweight(event)
```

The Slurm job runs inside the P-ONE IceTray Apptainer container:

```text
/cvmfs/software.pacific-neutrino.org/containers/itray_v1.17.1
```

The relevant software versions observed inside this container are:

```text
IceTray Python package: icecube.icetray 1.17.0
LeptonWeighter:        v1.1.5
LeptonWeighter commit: 5d86629fa9433e13640389ced2cd6e87eeec549f
P-ONE offline:         /cvmfs/software.pacific-neutrino.org/pone_offline/v2.0
```

The LeptonWeighter source code used by the container is available inside the
container at:

```text
/usr/local/LeptonWeighter/
```

The most relevant LeptonWeighter source files for this calculation are:

```text
/usr/local/LeptonWeighter/private/LeptonWeighter/Weighter.cpp
/usr/local/LeptonWeighter/private/LeptonWeighter/Generator.cpp
/usr/local/LeptonWeighter/private/LeptonWeighter/CrossSection.cpp
/usr/local/LeptonWeighter/private/LeptonWeighter/LeptonInjectorConfigReader.cpp
```



## Section 4: Flux-Free LIW Calculation (Note to myself, some workarounds)

The goal of this step is to compute flux-free LeptonInjector weights. In the
current script, the stored event weight is obtained with:

```python
oneweight = weighter.get_oneweight(event)
```

This is important: `get_oneweight()` does not multiply by any flux model. The
stored `oneweight` is the flux-independent part of the weight. Physical flux
models, such as atmospheric or astrophysical spectra, should be applied later in
the analysis stage.

For each event, LeptonWeighter accounts for the generated phase space and the
target interaction model. The generation probability includes the energy
generation spectrum, the generated direction range, the injection geometry, the
number of generated events in the LIC configuration, the generated final state,
and the interaction kinematics. The cross-section term is evaluated from the
configured differential cross-section splines.

In source-code form, LeptonWeighter computes:

```text
oneweight = cross_section(event) / generation_weight(event)
```

More explicitly, the denominator can be written schematically as:

```text
generation_weight =
    P(E)
  * P(direction)
  * P(position or injection area)
  * N_generated
  * P(final_state)
  * P(interaction kinematics)
```

The final LIW column written by this script then applies the sample-size
normalization:

```text
LIW = oneweight * N_GEN_PER_FILE / n_accessible_events
```

where the normalization is done separately for neutrinos and antineutrinos:

```text
LIW_nu     = oneweight * 100 / n_accessible_nu
LIW_antinu = oneweight * 100 / n_accessible_antinu
```

### Normalization Note

The LIC files used here contain 100 generated events per particle type per
generator file. This is why the script uses:

```python
N_GEN_PER_FILE = 100
```

The script processes the pre-PMT I3 sample and only includes events that are
accessible from readable, matched files and frames. Some files or frames can be
corrupt or otherwise unreadable, so the normalization uses the number of events
that are actually accessible to this weighting pipeline. Neutrinos and
antineutrinos are counted separately because they correspond to separate
generation components.

With approximately 100 accessible generated events per file for a given particle
type, the factor:

```text
100 / n_accessible_events
```

acts like an effective `1 / number_of_accessible_files` normalization. This
treats the successfully readable generator files as the Monte Carlo sample that
will be propagated to later PMT response, trigger, and selection stages.
