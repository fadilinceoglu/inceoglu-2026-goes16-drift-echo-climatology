# GOES-16 pitch-angle climatology asset

`goes16_pitch_climatology.npz` contains only the pitch-angle arrays consumed by the current study pipeline. Load it with `numpy.load(path, allow_pickle=False)`.

| Key | Shape | Type | Meaning |
|---|---|---|---|
| `electron_pitch_deg` | `(1440, 5)` | float64 | Electron pitch angle in degrees |
| `proton_pitch_deg` | `(1440, 5)` | float64 | Proton pitch angle in degrees |
| `percentile` | scalar | int64 | 95 |

Rows correspond to UTC minute 0–1439; columns correspond to Telescopes 1–5. Both arrays are finite, without missing values, and range from 7.028385162353516 to 149.23992919921875 degrees. Each array occupies 57,600 bytes before compression.

The existing detector substitutes this climatology for GOES-16 days from January 1 through April 11, 2017, even when magnetometer data are present. For a padded detection day it repeats the first daily rows into the following day. From April 12 onward, the detector uses magnetometer-derived pitch angles; its existing ephemeris fallback supplies missing pitch angles rather than using this climatology.

The source labels its percentiles `[5, 10, 25, 50, 75, 90, 95]`; the detector uses index 6, which is the **95th percentile**. R1 describes a median substitution. This asset preserves the current code's 95th-percentile behavior exactly; resolving that manuscript/code discrepancy is a separate scientific decision.

Provenance: `g16_mpshi_pas_model_201704-11_89p5W.p` (21,780,982 bytes). Source SHA-256:

`33a9dcce0cc3e68b49cfc10198c8ef9984328a33a0aec976e20d035002b2ca1a`

Exported entries are `percentiles_electron_pitch_angles[:, :, 6]` and `percentiles_proton_pitch_angles[:, :, 6]`, retaining float64 values without rounding. The original pickle contains other percentiles, summary statistics, and the full April–November 2017 time series; these unused arrays and object timestamps are excluded. No original pickle or duplicate arrays are needed to consume this pitch-angle asset. The paper uses calibrated Level-2 v2 fluxes and effective energies embedded in the NetCDF products, so no calibration LUT is bundled in this release path.

Regeneration, if needed, must use the known trusted source pickle matching the SHA-256 above. Select the two exact index-6 arrays, verify their shape and finite values, and save them under the keys above with scalar percentile 95 using `numpy.savez_compressed`. No separate converter executable is needed in the public release. The export was checked by loading with `allow_pickle=False` and comparing every element and dtype against the source slices; both arrays round-trip exactly.
