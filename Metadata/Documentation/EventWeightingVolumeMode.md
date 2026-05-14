# Volume Mode

This notebook explains how events generated with the volume-mode setup
documented in
`/project/def-nahee/kbas/Graphnet-Applications/Metadata/Documentation/LeptonInjectionVolumeMode.md`
are weighted.

In the current MC pipeline, each generated event can come from only one
matching generator. The full LeptonWeighter oneweight expression contains a
sum over all generators in the LIC file. For these datasets, all non-matching
generators have zero final-state probability for a given event, so that sum
reduces to the single matching generator term. Therefore, this notebook uses
the simplified oneweight expression appropriate for the current pipeline.

## OneWeight

$$
\text{oneweight}
=
\frac{
\left(\dfrac{d^2\sigma}{dx\,dy}\right)_{\mathrm{phys}}
}{
N \times P_E \times P_{\mathrm{direction}} \times P_{\mathrm{interaction}} \times P_{\mathrm{position}}
}
$$

This is the `oneweight` column in the CSV produced by
`/project/def-nahee/kbas/Graphnet-Applications/DataPreperation/EventWeights/LIW/calculate_LIW.py`.



## Number of generated events

$$
N = 100
$$

$N$ is a unitless event count.



## Direction probability

$$
P_{\mathrm{direction}}
=
\frac{1}{
(\phi_{\max,\mathrm{LIC}}-\phi_{\min,\mathrm{LIC}})
(\cos\theta_{\min,\mathrm{LIC}}-\cos\theta_{\max,\mathrm{LIC}})
}
$$

Here, $\theta$ is the zenith angle and $\phi$ is the azimuth angle. The limits
are read from the matching generator stored in the LIC file. For the current
full-sky datasets,

$$
\phi_{\min,\mathrm{LIC}} = 0,\quad
\phi_{\max,\mathrm{LIC}} = 2\pi,\quad
\theta_{\min,\mathrm{LIC}} = 0,\quad
\theta_{\max,\mathrm{LIC}} = \pi,
$$


The unit of $P_{\mathrm{direction}}$ is $\mathrm{sr}^{-1}$.



## Energy probability

$$
P_E(E)
=
C_{\mathrm{LIC}} E^{-\gamma_{\mathrm{LIC}}}
$$

where

$$
C_{\mathrm{LIC}}
=
\frac{
1-\gamma_{\mathrm{LIC}}
}{
E_{\max,\mathrm{LIC}}^{1-\gamma_{\mathrm{LIC}}}
-
E_{\min,\mathrm{LIC}}^{1-\gamma_{\mathrm{LIC}}}
}
\quad
(\gamma_{\mathrm{LIC}} \neq 1).
$$

The energy limits and power-law index are read from the matching generator
stored in the LIC file. For the current datasets,

$$
\gamma_{\mathrm{LIC}} = 1.5,\quad
E_{\min,\mathrm{LIC}} = 10^2\,\mathrm{GeV},\quad
E_{\max,\mathrm{LIC}} = 10^6\,\mathrm{GeV},
$$

so

$$
P_E(E)
=
-\frac{1}{2}
\frac{
E^{-1.5}
}{
(10^6)^{-1/2} - (10^2)^{-1/2}
}.
$$

The unit of $P_E$ is $\mathrm{GeV}^{-1}$.



## Position probability

$$
P_{\mathrm{position}}
=
\frac{
L_{\text{eff}}
}{
10^4 \times \pi \times r_{\mathrm{LIC}}^2 \times h_{\mathrm{LIC}}
}
$$

Here, $r_{\mathrm{LIC}}$ and $h_{\mathrm{LIC}}$ are the volume-mode cylinder
radius and height read from the matching generator stored in the LIC file.
They correspond to `CylinderRadius` and `CylinderHeight` in the LeptonInjector
configuration.

$L_{\text{eff}}$ is the length of the chord through the injection cylinder
along the sampled neutrino direction, passing through the sampled interaction
vertex. Since the event vertex coordinates `EventProperties.x`,
`EventProperties.y`, and `EventProperties.z` are stored in meters, and the LIC
cylinder radius and height are also in meters, $L_{\text{eff}}$ is measured in
meters.

The unit of $P_{\mathrm{position}}$ is therefore $\mathrm{m}^{-2}$.



## Interaction probability

$$
P_{\mathrm{interaction}}
=
\frac{
\left(\dfrac{d^2\sigma}{dx\,dy}\right)_{\mathrm{LIC}}
}{
1 -
\exp
\left(
-N_A \, \mathrm{EventProperties.totalColumnDepth} \, \left(\sigma_{\mathrm{tot}}\right)_{\mathrm{LIC}}
\right)
}
$$



Here, $N_A = 6.022140857 \times 10^{23}$.

In this interaction-probability term, both cross-section inputs come from the
matching generator stored in the LIC file: the double-differential cross-section
spline and the total cross-section spline. This is separate from the
cross-section in the numerator of the oneweight expression, which is loaded by
`calculate_LIW.py` through `LW.CrossSectionFromSpline(...)`.

The exponential argument is dimensionless. Since $x$ and $y$ are dimensionless,
the unit of
$\left(d^2\sigma/dx\,dy\right)_{\mathrm{LIC}}$ is the same as the unit of a
cross section. Therefore, the unit of $P_{\mathrm{interaction}}$ is $\mathrm{cm}^2$.


## Notes

1. The double-differential cross section is a function of the interaction
   variables $x$, $y$, and the neutrino energy $E$:

$$
\frac{d^2\sigma}{dx\,dy}
=
\text{function of } x, y, E
$$




2. The total cross section is a function of the neutrino energy $E$:

$$
\sigma_{\text{tot}}
=
\int
\left(
\frac{d^2\sigma}{dx\,dy}
\right)
dx\,dy
=
\text{function of } E
$$



3. The effective length is a function of the sampled vertex position and the
   neutrino direction:

$$
L_{\text{eff}}
=
\text{function of vertex position and neutrino direction}
$$
