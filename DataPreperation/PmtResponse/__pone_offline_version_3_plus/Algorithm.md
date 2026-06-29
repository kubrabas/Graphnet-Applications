# PONE offline v3 PMT response algorithm

This note describes the current v3 worker:

```text
DataPreperation/PmtResponse/__pone_offline_version_3/apply_pmt_response_with_first_3_layers.py
```

The important difference from the older PMT response flow is that v3 produces
the full 340-string detector response first, then derives the 160-string and
102-string maps by subselecting strings from the full 340-string maps.

## Physical flow

For each DAQ frame:

1. `I3Photons` is read from the input frame.
2. `OMAcceptance` is applied once for the full 340-string geometry.
3. The accepted photons are written as:

```text
Accepted_PulseMap_340String
```

4. Dark noise and K40 noise are generated using `Accepted_PulseMap_340String`:

```text
Noise_Dark_340_String
Noise_K40_340_String
```

5. `DOMSimulation` combines signal + noise and applies the PMT response model.
6. The full 340-string response maps are written:

```text
PMT_Response_340_String
PMT_Response_nonoise_340_String
```

7. The 160-string and 102-string maps are derived by string subselection from
   the full 340-string maps.
8. Trigger flags and trigger times are computed separately for the 340, 160 and
   102 string layouts.
9. The DAQ and Simulation frames are written to the output I3 file.

## Output frame keys

The original input / metadata keys are preserved when present, for example:

```text
I3Photons
I3MCTree
I3MCTree_postprop
I3MCTree_RNGState
MMCTrackList
EventProperties
I3EventHeader
```

The PMT response worker adds these full-geometry keys:

```text
Accepted_PulseMap_340String

Noise_Dark_340_String
Noise_K40_340_String

PMT_Response_340_String
PMT_Response_nonoise_340_String
```

It also adds these subgeometry keys:

```text
Noise_Dark_160_String
Noise_K40_160_String
PMT_Response_160_String
PMT_Response_nonoise_160_String

Noise_Dark_102_String
Noise_K40_102_String
PMT_Response_102_String
PMT_Response_nonoise_102_String
```

The old generic names are deleted if they already exist in the input frame:

```text
Accepted_PulseMap
Noise_Dark
Noise_K40
PMT_Response
PMT_Response_nonoise
```

This prevents stale maps from previous processing from surviving into the new
output.

## Acceptance model

Acceptance is applied only once, for the full 340-string geometry:

```text
I3Photons -> Accepted_PulseMap_340String
```

The acceptance module splits photons by PMT and applies the P-OM acceptance
model. Accepted photons are converted to `I3MCPE` entries with:

```text
time = photon.time
npe  = 1
```

There is no separate `Accepted_PulseMap_160_String` or
`Accepted_PulseMap_102_String` in the current v3 flow. The subgeometries are
created later from the already-produced 340-string noise and response maps.

## Noise model

Noise is generated only for the full 340-string flow:

```text
Accepted_PulseMap_340String -> Noise_Dark_340_String
Accepted_PulseMap_340String -> Noise_K40_340_String
```

The noise modules use the accepted signal map to determine the time bounds.
Manual bounds are not used, so the noise window is determined from the signal
hit time range plus the module padding:

```text
first accepted hit time - 2000 ns
last accepted hit time + 10000 ns
```

### Dark noise

Dark noise is generated as an independent Poisson process per PMT.

- Rate: `0.000001 pulses/ns`.
- Each dark hit is stored as an `I3MCPE`.
- The generated full-geometry map is `Noise_Dark_340_String`.

### K40 noise

K40 noise is generated from the K40 characterization file.

- Event times are sampled from an exponential process.
- Events can be single-fold or multi-fold.
- Multi-PMT combinations and time offsets are sampled from the characterization.
- The generated full-geometry map is `Noise_K40_340_String`.

## PMT response model

`DOMSimulation` runs only once, using the full 340-string accepted signal and
full 340-string noise maps:

```text
input_map  = Accepted_PulseMap_340String
dark_map   = Noise_Dark_340_String
k40_map    = Noise_K40_340_String
output_map = PMT_Response_340_String
```

The module writes a signal+noise response:

```text
PMT_Response_340_String
```

and a signal-only response. Internally, `DOMSimulation` first writes:

```text
PMT_Response_340_String_nonoise
```

The worker immediately renames that to:

```text
PMT_Response_nonoise_340_String
```

Main response effects:

- PMT transit-time spread,
- late pulses,
- afterpulses,
- PMT dead time,
- charge smearing,
- pulse merging,
- thresholding,
- saturation.

The worker uses:

```text
min_time_sep = 0.2 ns
```

## Subgeometry maps

The 160-string and 102-string maps are not regenerated independently. They are
created by filtering the full 340-string maps by string id.

For 160 strings:

```text
Noise_Dark_340_String          -> Noise_Dark_160_String
Noise_K40_340_String           -> Noise_K40_160_String
PMT_Response_340_String        -> PMT_Response_160_String
PMT_Response_nonoise_340_String -> PMT_Response_nonoise_160_String
```

For 102 strings:

```text
Noise_Dark_340_String          -> Noise_Dark_102_String
Noise_K40_340_String           -> Noise_K40_102_String
PMT_Response_340_String        -> PMT_Response_102_String
PMT_Response_nonoise_340_String -> PMT_Response_nonoise_102_String
```

The string lists come from:

```text
Metadata/GeometryFiles/340StringMC/102_string.csv
Metadata/GeometryFiles/340StringMC/160_string.csv
Metadata/GeometryFiles/string_coordinates_340_string_mc.csv
```

## Trigger calculation

Triggering is computed independently for each geometry:

```text
340_string trigger uses PMT_Response_340_String
160_string trigger uses PMT_Response_160_String
102_string trigger uses PMT_Response_102_String
```

So yes: each geometry uses its own noisy PMT response map.

The no-noise maps are not used for the trigger:

```text
PMT_Response_nonoise_340_String
PMT_Response_nonoise_160_String
PMT_Response_nonoise_102_String
```

are signal-only diagnostic maps, not trigger inputs.

The trigger condition is:

```text
inside one DOM
within a 10 ns window
hits on at least 3 distinct PMTs
```

The first time at which this condition is satisfied becomes the trigger time
for that geometry.

The worker writes:

```text
triggered_340_string
trigger_time_340_string

triggered_160_string
trigger_time_160_string

triggered_102_string
trigger_time_102_string
```

If no DOM satisfies the trigger condition:

```text
triggered_<layout> = 0.0
trigger_time_<layout> = -1.0
```

If at least one DOM satisfies the trigger condition:

```text
triggered_<layout> = 1.0
trigger_time_<layout> = earliest trigger time
```

## Frame writing behavior

In the current v3 worker, the trigger is not used as a frame filter. It writes
trigger flags and trigger times, then sends the DAQ frame to `I3Writer`.

So the current behavior is:

```text
DAQ frame sent to writer =
    DAQ frame was processed successfully
```

not:

```text
DAQ frame sent to writer =
    frame passed a trigger cut
```

This is different from workflows that build an `EventPulseSeries` only after a
trigger. The current v3 worker does not create `EventPulseSeries` or
`EventPulseSeries_nonoise`.

## Output location

For `STRING340MC`, submit writes outputs to:

```text
/home/kbas/scratch/String340MC_pone_offline_version3/<Flavor>_PMT_Response
```

and logs to:

```text
/home/kbas/scratch/String340MC_pone_offline_version3/Logs/<Flavor>_pmt_response
```

For example:

```text
/home/kbas/scratch/String340MC_pone_offline_version3/Electron_PMT_Response
/home/kbas/scratch/String340MC_pone_offline_version3/Logs/Electron_pmt_response
```

## Code sources

Sources used for this description:

- `DataPreperation/PmtResponse/__pone_offline_version_3/apply_pmt_response_with_first_3_layers.py`
- `DataPreperation/PmtResponse/__pone_offline_version_3/submit_pmt_response.py`
- `/project/def-nahee/kbas/my_pone_offline/DOM/OMAcceptance.py`
- `/project/def-nahee/kbas/my_pone_offline/NoiseGenerators/DarkNoise.py`
- `/project/def-nahee/kbas/my_pone_offline/NoiseGenerators/K40Noise.py`
- `/project/def-nahee/kbas/my_pone_offline/DOM/PONEDOMLauncher.py`
