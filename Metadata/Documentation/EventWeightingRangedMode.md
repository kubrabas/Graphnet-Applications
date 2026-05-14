# Ranged Mode

This notebook explains how events generated with the ranged-mode setup
documented in
`/project/def-nahee/kbas/Graphnet-Applications/Metadata/Documentation/LeptonInjectionRangeMode.md`
are weighted.

In the current MC pipeline, each generated event can come from only one
matching generator. The full LeptonWeighter oneweight expression contains a
sum over all generators in the LIC file. For these datasets, all non-matching
generators have zero final-state probability for a given event, so that sum
reduces to the single matching generator term. Therefore, this notebook uses
the simplified oneweight expression appropriate for the current pipeline.

For the current muon-neutrino charged-current datasets, the matching generator
is selected by the final-state particles. A `MuMinus + Hadrons` event matches
the `NuMu` generator, and a `MuPlus + Hadrons` event matches the `NuMuBar`
generator.

## OneWeight

```math
\text{oneweight}
=
\frac{
\left(\dfrac{d^2\sigma}{dx\,dy}\right)_{\mathrm{phys}}
}{
N \times P_E \times P_{\mathrm{direction}} \times P_{\mathrm{interaction}} \times P_{\mathrm{area}}
}
```

This is the `oneweight` column in the CSV produced by
`/project/def-nahee/kbas/Graphnet-Applications/DataPreperation/EventWeights/LIW/calculate_LIW.py`.



## Number of generated events

```math
N = 100
```

$N$ is a unitless event count.



## Direction probability

```math
P_{\mathrm{direction}}
=
\frac{1}{
(\phi_{\max,\mathrm{LIC}}-\phi_{\min,\mathrm{LIC}})
(\cos\theta_{\min,\mathrm{LIC}}-\cos\theta_{\max,\mathrm{LIC}})
}
```

Here, $\theta$ is the zenith angle and $\phi$ is the azimuth angle. The limits
are read from the matching generator stored in the LIC file. For the current
full-sky datasets,

```math
\phi_{\min,\mathrm{LIC}} = 0,\quad
\phi_{\max,\mathrm{LIC}} = 2\pi,\quad
\theta_{\min,\mathrm{LIC}} = 0,\quad
\theta_{\max,\mathrm{LIC}} = \pi,
```

so

```math
P_{\mathrm{direction}} = \frac{1}{4\pi}.
```

The unit of $P_{\mathrm{direction}}$ is $\mathrm{sr}^{-1}$.



## Energy probability

```math
P_E(E)
=
C_{\mathrm{LIC}} E^{-\gamma_{\mathrm{LIC}}}
```

where

```math
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
```

The energy limits and power-law index are read from the matching generator
stored in the LIC file. For the current datasets,

```math
\gamma_{\mathrm{LIC}} = 1.5,\quad
E_{\min,\mathrm{LIC}} = 10^2\,\mathrm{GeV},\quad
E_{\max,\mathrm{LIC}} = 10^6\,\mathrm{GeV},
```

so

```math
P_E(E)
=
-\frac{1}{2}
\frac{
E^{-1.5}
}{
(10^6)^{-1/2} - (10^2)^{-1/2}
}.
```

The unit of $P_E$ is $\mathrm{GeV}^{-1}$.



## Area probability

```math
P_{\mathrm{area}}
=
\frac{1}{
10^4 \times \pi \times R_{\mathrm{LIC}}^2
}
```

Here, $R_{\mathrm{LIC}}$ is the ranged-mode injection disk radius read from the
matching generator stored in the LIC file. It corresponds to `InjectionRadius`
in the LeptonInjector configuration.

In ranged mode, LeptonInjector samples the closest-approach point, `pca`, on a
disk perpendicular to the sampled neutrino direction. The disk is centered on
the detector-coordinate origin and has radius $R_{\mathrm{LIC}}$. In
LeptonWeighter, this disk-area factor is handled by `probability_area`.

The factor $10^4$ converts the area factor from $\mathrm{m}^2$ to
$\mathrm{cm}^2$. Therefore, the unit of $P_{\mathrm{area}}$ is
$\mathrm{cm}^{-2}$.

For ranged mode, the source code sets the position probability to 1:

```math
P_{\mathrm{position}} = 1.
```

The physical vertex is sampled along the allowed column-depth region during
event generation, but LeptonWeighter does not include an additional
volume-position probability for ranged mode.



## Interaction probability

```math
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
```

Here, `EventProperties.totalColumnDepth` is copied into
`LW.Event.total_column_depth` by `calculate_LIW.py` before calling
`weighter.get_oneweight(event)`.

Here, $N_A = 6.022140857 \times 10^{23}$.

In this interaction-probability term, both cross-section inputs come from the
matching generator stored in the LIC file: the double-differential cross-section
spline and the total cross-section spline. This is separate from the
cross-section in the numerator of the oneweight expression, which is loaded by
`calculate_LIW.py` through `LW.CrossSectionFromSpline(...)`.

The exponential argument is dimensionless. Since $x$ and $y$ are dimensionless,
the unit of

```math
\left(\frac{d^2\sigma}{dx\,dy}\right)_{\mathrm{LIC}}
```

is the same as the unit of a cross section. Therefore, the unit of
$P_{\mathrm{interaction}}$ is $\mathrm{cm}^2$.


## OneWeight unit

The units entering the simplified ranged-mode expression are

```math
\left(\frac{d^2\sigma}{dx\,dy}\right)_{\mathrm{phys}}
\rightarrow \mathrm{cm}^2,
```

```math
N \rightarrow 1,\quad
P_E \rightarrow \mathrm{GeV}^{-1},\quad
P_{\mathrm{direction}} \rightarrow \mathrm{sr}^{-1},
```

```math
P_{\mathrm{interaction}} \rightarrow \mathrm{cm}^2,\quad
P_{\mathrm{area}} \rightarrow \mathrm{cm}^{-2}.
```

Therefore,

```math
\text{oneweight}
\rightarrow
\frac{\mathrm{cm}^2}{
\mathrm{GeV}^{-1}\,\mathrm{sr}^{-1}\,\mathrm{cm}^2\,\mathrm{cm}^{-2}
}
=
\mathrm{GeV}\,\mathrm{sr}\,\mathrm{cm}^2.
```


## Notes

1. The double-differential cross section is a function of the interaction
   variables $x$, $y$, and the neutrino energy $E$:

```math
\frac{d^2\sigma}{dx\,dy}
=
\text{function of } x, y, E
```

2. The total cross section is a function of the neutrino energy $E$:

```math
\sigma_{\text{tot}}
=
\int
\left(
\frac{d^2\sigma}{dx\,dy}
\right)
dx\,dy
=
\text{function of } E
```

3. `EventProperties.totalColumnDepth` is the total column depth of the allowed
   ranged-mode sampling region, as documented in
   `LeptonInjectionRangeMode.md`.
