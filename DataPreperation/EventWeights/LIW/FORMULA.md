# LeptonInjector Weight Formula Notes

This note documents what `calculate_LIW.py` currently does when producing the
`oneweight` column. For now, this document focuses only on the raw
LeptonWeighter `get_oneweight(event)` calculation.

## Source Files

The relevant script in this repository is:

```text
DataPreperation/EventWeights/LIW/calculate_LIW.py
```

The LeptonWeighter object is built from two ingredients:

```python
generators = LW.MakeGeneratorsFromLICFile(lic_path)
weighter = LW.Weighter(_xs, generators)
```

where:

```text
lic_path  = matched LeptonInjector .lic file for the current batch
_xs       = cross-section object loaded from CSMS differential spline files
generators = generator configuration read from the .lic file
```

The cross-section object is loaded as:

```python
_xs = LW.CrossSectionFromSpline(
    XS_PATH + "dsdxdy_nu_CC_iso.fits",
    XS_PATH + "dsdxdy_nubar_CC_iso.fits",
    XS_PATH + "dsdxdy_nu_NC_iso.fits",
    XS_PATH + "dsdxdy_nubar_NC_iso.fits",
)
```

## Per-Frame Inputs

For each readable DAQ frame, the script requires:

```text
I3EventHeader
EventProperties
```

The event identity is taken from `I3EventHeader`:

```text
RunID       = hdr.run_id
SubrunID    = hdr.sub_run_id
EventID     = hdr.event_id
SubEventID  = hdr.sub_event_id
```

The physics quantities are taken from `EventProperties` and copied into a
`LeptonWeighter.Event` object:

```text
EventProperties.totalEnergy       -> event.energy
EventProperties.zenith            -> event.zenith
EventProperties.azimuth           -> event.azimuth
EventProperties.finalStateX       -> event.interaction_x
EventProperties.finalStateY       -> event.interaction_y
EventProperties.initialType       -> event.primary_type
EventProperties.finalType1        -> event.final_state_particle_0
EventProperties.finalType2        -> event.final_state_particle_1
EventProperties.impactParameter   -> event.radius
EventProperties.totalColumnDepth  -> event.total_column_depth
EventProperties.x                 -> event.x
EventProperties.y                 -> event.y
EventProperties.z                 -> event.z
```

In code, the object is built like this:

```python
event = LW.Event()
event.energy = props.totalEnergy
event.zenith = props.zenith
event.azimuth = props.azimuth
event.interaction_x = props.finalStateX
event.interaction_y = props.finalStateY
event.primary_type = primary
event.final_state_particle_0 = fs0
event.final_state_particle_1 = fs1
event.radius = props.impactParameter
event.total_column_depth = props.totalColumnDepth
event.x = props.x
event.y = props.y
event.z = props.z
```

The particle-type fields are converted from `EventProperties` particle codes to
LeptonWeighter particle enums before they are assigned:

```python
primary = to_lw_particle(props.initialType)
fs0 = to_lw_particle(props.finalType1)
fs1 = to_lw_particle(props.finalType2)
```

## Oneweight Column

The actual weight stored in the `oneweight` column is produced by one call:

```python
oneweight = weighter.get_oneweight(event)
```

For one event `i`, define:

```math
E_i      = \texttt{event.energy}
```

```math
\theta_i = \texttt{event.zenith}
```

```math
\phi_i   = \texttt{event.azimuth}
```

```math
x_i      = \texttt{event.interaction\_x}
```

```math
y_i      = \texttt{event.interaction\_y}
```

```math
X_i      = \texttt{event.total\_column\_depth}
```

```math
\vec{r}_i =
(\texttt{event.x}, \texttt{event.y}, \texttt{event.z})
```

The full `oneweight` formula is:

```math
\boxed{
\mathrm{oneweight}_i =
\frac{
    \sigma_i
}{
    \sum_{g \in \mathrm{generators}} G_g(i)
}
}
```

Here `generators` is the list read from the matched LIC file:

```python
generators = LW.MakeGeneratorsFromLICFile(lic_path)
```

For the current `MC000002-nu_mu-2_7` generation script, one file contains two
ranged injectors:

```text
generator 1: FinalType1 = MuMinus, FinalType2 = Hadrons, Ranged = True
generator 2: FinalType1 = MuPlus,  FinalType2 = Hadrons, Ranged = True
```

So for this sample, the denominator for one matched LIC/I3 batch is not a sum
over all LIC files in the dataset. It is the sum over the generators stored in
that matched LIC file. In this case:

```math
\sum_{g \in \mathrm{generators}} G_g(i)
=
G_{\mu^-}(i) + G_{\mu^+}(i)
```

### Numerator

```math
\sigma_i =
10^4
\times
S_{\nu/\bar{\nu},\,CC/NC}^{\mathrm{external}}
\left(
    \log_{10} E_i,
    \log_{10} x_i,
    \log_{10} y_i
\right)
```

The selected spline comes from the cross-section object `_xs` built in
`calculate_LIW.py`:

```python
_xs = LW.CrossSectionFromSpline(
    XS_PATH + "dsdxdy_nu_CC_iso.fits",
    XS_PATH + "dsdxdy_nubar_CC_iso.fits",
    XS_PATH + "dsdxdy_nu_NC_iso.fits",
    XS_PATH + "dsdxdy_nubar_NC_iso.fits",
)
```

The spline choice is:

```text
event.primary_type is neutrino     + charged final particle -> nu_CC
event.primary_type is neutrino     + no charged final       -> nu_NC
event.primary_type is antineutrino + charged final particle -> nubar_CC
event.primary_type is antineutrino + no charged final       -> nubar_NC
```

This term is returned by LeptonWeighter as a double-differential cross section in
`cm^2`.

### Denominator

The denominator is the summed generation weight from every generator in the LIC
file:

```math
G_{\mathrm{total}}(i) =
\sum_{g \in \mathrm{generators}} G_g(i)
```

For one generator `g`, LeptonWeighter uses:

```math
G_g(i) =
N_g
\times
P_E^g(E_i)
\times
P_\Omega^g(\theta_i, \phi_i)
\times
P_A^g
\times
P_{\mathrm{pos}}^g(\vec{r}_i,\theta_i,\phi_i)
\times
P_{\mathrm{fs}}^g
\times
P_{\mathrm{int}}^g(E_i,x_i,y_i,X_i)
```

where `N_g` is the number of events stored in the LIC generator configuration.
If the event is outside the generator phase space, the relevant probability is
zero and therefore `G_g(i)=0`.

The generated-energy probability is:

```math
P_E^g(E_i) =
\begin{cases}
C_g E_i^{-\alpha_g}, & E_{\min,g} \le E_i \le E_{\max,g} \\
0, & \mathrm{otherwise}
\end{cases}
```

with:

```math
C_g =
\begin{cases}
\dfrac{1-\alpha_g}
{E_{\max,g}^{1-\alpha_g} - E_{\min,g}^{1-\alpha_g}},
& \alpha_g \ne 1 \\
\dfrac{1}{\ln(E_{\max,g}/E_{\min,g})},
& \alpha_g = 1
\end{cases}
```

The generated-direction probability is:

```math
P_\Omega^g(\theta_i,\phi_i) =
\begin{cases}
\dfrac{1}
{(\phi_{\max,g}-\phi_{\min,g})
(\cos\theta_{\min,g}-\cos\theta_{\max,g})},
& \theta_i,\phi_i \mathrm{\ inside\ bounds} \\
0, & \mathrm{otherwise}
\end{cases}
```

The final-state probability is:

```math
P_{\mathrm{fs}}^g =
\begin{cases}
1, & \mathrm{event.final\_state\_particle\_0/1\ match\ the\ generator\ final\ state} \\
0, & \mathrm{otherwise}
\end{cases}
```

The target-count factor is:

```math
T_i = N_A X_i
```

with:

```math
N_A = 6.022140857 \times 10^{23}
```

The interaction-kinematics probability is:

```math
P_{\mathrm{int}}^g(E_i,x_i,y_i,X_i) =
\frac{
    S_{\mathrm{diff}}^g(\log_{10}E_i,\log_{10}x_i,\log_{10}y_i)
}{
    1 -
    \exp\left[
        -S_{\mathrm{tot}}^g(\log_{10}E_i,\log_{10}x_i,\log_{10}y_i)
        T_i
    \right]
}
```

where `S_diff^g` and `S_tot^g` are the differential and total cross-section
splines stored in the LIC generator configuration. In the source code, the
stored spline values are exponentiated as `10^(spline value)`.

Now split the generator contribution by generator type.

### Range Generator Formula

For one ranged generator `g`, the source code has:

```math
P_A^g =
\frac{1}
{10^4 \pi R_{\mathrm{inj},g}^{2}}
```

```math
P_{\mathrm{pos}}^g = 1
```

Therefore the ranged-generator contribution is:

```math
\boxed{
G_{\mathrm{range},g}(i) =
N_g
P_E^g(E_i)
P_\Omega^g(\theta_i,\phi_i)
P_{\mathrm{fs}}^g
P_{\mathrm{int}}^g(E_i,x_i,y_i,X_i)
\frac{1}{10^4 \pi R_{\mathrm{inj},g}^{2}}
}
```

If the LIC file contains only ranged generators, the oneweight is:

```math
\boxed{
\mathrm{oneweight}_i^{\mathrm{range}} =
\frac{
    \sigma_i
}{
    \sum_{g \in \mathrm{range\ generators}}
    G_{\mathrm{range},g}(i)
}
}
```

### Volume Generator Formula

For one volume generator `g`, the source code has:

```math
P_A^g = 1
```

```math
P_{\mathrm{pos}}^g(\vec{r}_i,\theta_i,\phi_i) =
\begin{cases}
\dfrac{
    L_{\mathrm{eff}}^g(\vec{r}_i,\theta_i,\phi_i)
}{
    10^4 \pi R_{\mathrm{cyl},g}^{2} H_{\mathrm{cyl},g}
},
& \vec{r}_i \mathrm{\ inside\ the\ generation\ cylinder} \\
0, & \mathrm{otherwise}
\end{cases}
```

Here `L_eff` is the chord length through the generation cylinder in the event
direction.

Therefore the volume-generator contribution is:

```math
\boxed{
G_{\mathrm{volume},g}(i) =
N_g
P_E^g(E_i)
P_\Omega^g(\theta_i,\phi_i)
P_{\mathrm{fs}}^g
P_{\mathrm{int}}^g(E_i,x_i,y_i,X_i)
P_{\mathrm{pos}}^g(\vec{r}_i,\theta_i,\phi_i)
}
```

If the LIC file contains only volume generators, the oneweight is:

```math
\boxed{
\mathrm{oneweight}_i^{\mathrm{volume}} =
\frac{
    \sigma_i
}{
    \sum_{g \in \mathrm{volume\ generators}}
    G_{\mathrm{volume},g}(i)
}
}
```

### Mixed Generator Formula

If the LIC file contains both ranged and volume generators, LeptonWeighter sums
all generator contributions:

```math
\boxed{
\mathrm{oneweight}_i =
\frac{
    \sigma_i
}{
    \sum_{g \in \mathrm{range\ generators}}
    G_{\mathrm{range},g}(i)
    +
    \sum_{g \in \mathrm{volume\ generators}}
    G_{\mathrm{volume},g}(i)
}
}
```

The range and volume sizes in these formulas come from the LIC generator
configuration.
