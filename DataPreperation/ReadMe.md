# Truth Extractor Output Fields

## Event Identity
- **RunID**  
  Simulation run number (from I3EventHeader).

- **SubrunID**  
  Sub-run number. 4294967295 (=2^32−1) means undefined.

- **EventID**  
  Event number within the run (from I3EventHeader).

- **SubEventID**  
  Sub-event number. Always 0 in PONE (no frame splitting).

- **event_time**  
  DAQ timestamp in UTC. 0 in most PONE simulations.

- **sim_type**  
  Simulator that produced the event.  
  `LeptonInjector` when EventProperties is present in the frame; `data` for real data.

---

## Primary Particle
- **pid**  
  PDG encoding of the primary particle.

  ```
  12 / -12  = νe / ν̄e
  14 / -14  = νμ / ν̄μ
  16 / -16  = ντ / ν̄τ
  13 / -13  = μ⁻ / μ⁺  (atmospheric or MuonGun)
  ```

- **energy**  
  Total energy of the primary particle in GeV (`MCTree[0].energy`).

- **position_x / position_y / position_z**  
  Neutrino interaction vertex in metres (`MCTree[0].pos`).  
  For primary muons this is the muon's starting position.

- **azimuth**  
  Azimuth angle of the primary in radians.

- **zenith**  
  Zenith angle of the primary in radians.

---

## Interaction
- **interaction_type**

  ```
  1  = CC (Charged Current, e.g. νμ → μ + hadrons)
  2  = NC (Neutral Current, e.g. νμ → νμ + hadrons)
 
  ```

- **elasticity**  

  Fraction of neutrino energy carried away by the outgoing lepton.

  ```
  elasticity = 1 - BjorkenY = E_lepton / E_nu
  ```

  Derived from `EventProperties.finalStateY` for LeptonInjector.

- **inelasticity**  
  Fraction of neutrino energy deposited as hadrons.

  ```
  inelasticity = BjorkenY = E_hadrons / E_nu
  elasticity + inelasticity = 1.0
  ```

---

## Energy Decomposition
- **energy_track**  
  Energy of the outgoing lepton in GeV.

  For LeptonInjector:

  ```
  energy_track = energy * (1 - BjorkenY)
  ```

  Valid for all flavours:

  ```
  νμ → μ
  νe → e
  ντ → τ
  NC → outgoing ν
  ```

- **energy_cascade**  
  Energy deposited as hadrons in GeV.

  ```
  energy_cascade = energy * BjorkenY
  energy_track + energy_cascade = energy
  ```

---

## Double-Bang
- **dbang_decay_length**  
  Distance in metres between the neutrino vertex and the tau decay point for ντ CC events,  
  or between the two hadronic cascades for HNL events.

  Behavior by decay channel:

  - τ → hadrons (~65%) → equals `tau_decay_length`
  - τ → e (~18%) → equals `tau_decay_length`
  - τ → μ (~17%) → `-1` (no second hadronic cascade)

  `-1` for all non-ντ topologies.

---

## NuMu Muon (νμ CC only, pid=±14)

The muon produced at the νμ CC interaction vertex (`νμ → μ + hadrons`).

All `numu_muon_*` fields are `-1` for non-νμ events (νe, ντ, atmospheric μ).

- **numu_muon_track_length**  
  Path length of the muon in metres, taken directly from  
  `muon_particle.length` written by PROPOSAL into `I3MCTree_postprop`.

- **numu_muon_final_x / numu_muon_final_y / numu_muon_final_z**  
  Estimated stopping position in metres, computed by propagating from  
  the CC vertex along the muon direction by `numu_muon_track_length`.

- **numu_muon_stopped_convex_hull**  
  True if the stopping position lies within the convex hull of the detector  
  (shrunk by 100 m horizontally and vertically).

  ⚠ Unreliable for non-convex or multi-cluster geometries because the convex hull  
  fills the empty space between string clusters.

- **numu_muon_stopped_near_string**  
  True if the stopping position is within `string_proximity_threshold` metres (XY)  
  of the nearest string **and** within the instrumented Z range.

  Geometry-agnostic and works for any detector layout.

---

## Tau Decay Muon (ντ CC only, pid=±16, τ → μ decay mode ~17%)

- **tau_decay_length**  
  Path length of the tau lepton in metres  
  (`tau_particle.length` from `I3MCTree_postprop`).

  Distance from the neutrino vertex to the tau decay point.

  Notes:

  - Near zero at hundreds of GeV
  - Meaningful at PeV energies
  - `-1` when `tau.length` is NaN or zero

- **tau_decay_muon_track_length**  
  Path length of the muon from tau decay in metres.

- **tau_decay_muon_final_x / tau_decay_muon_final_y / tau_decay_muon_final_z**  
  Estimated stopping position of that muon.

- **tau_decay_muon_stopped_convex_hull**  
  Convex hull containment check (same caveats as `numu_muon_stopped_convex_hull`).

- **tau_decay_muon_stopped_near_string**  
  True if within `string_proximity_threshold` metres (XY) of the nearest string  
  and within the instrumented Z range.

Notes:

- `tau_decay_length` is filled for all ντ CC events where a tau is found.
- `tau_decay_muon_*` are `-1` when the tau does **not** decay to a muon  
  (`τ → hadrons` or `τ → e`).
- All `tau_decay_*` fields are `-1` for non-ντ events.

---

## Atmospheric / MuonGun Muon (primary μ only, pid=±13)

- **atmo_muon_track_length**  
  Path length of the primary muon in metres.

- **atmo_muon_final_x / atmo_muon_final_y / atmo_muon_final_z**  
  Estimated stopping position in metres.

- **atmo_muon_stopped_convex_hull**  
  Same convex hull containment check as above, applied to the primary muon's stopping point.

- **atmo_muon_stopped_near_string**  
  Same proximity-based check as above.

All `atmo_muon_*` fields are `-1` for neutrino events (`pid = ±12/14/16`).

---

## Vertex Containment

- **is_starting_convex_hull**  
  True if the interaction vertex lies within the 3D convex hull built from GCD sensor positions.

  Fast but treats empty space between string clusters as instrumented volume.

- **is_starting_near_string**  
  True if the interaction vertex is within `string_proximity_threshold` metres (XY)  
  of the nearest string **and** within the instrumented Z range.

  Correct for any detector geometry including non-convex and multi-cluster layouts.