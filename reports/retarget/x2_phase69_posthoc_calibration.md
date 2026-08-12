# X2 Phase69 posthoc calibration

- Decision: `POSTHOC_CALIBRATION_NO_PROMOTION`
- Phase69 remains `FAIL_ATTRIBUTION_INVALID_STOP`; the bundle does not contain the actual RSL storage advantage vector.
- RSL-order CPU recurrence vs saved offline advantage max error: `5.662e-06`
- Pitch reward → positive signed pitch direction cosine: `0.995726`; standalone projection `0.985532`; common-denominator reward-gradient projection `0.194591`
- Support reward → lower support outside direction cosine: `0.997050`; standalone projection `1.011072`; common-denominator reward-gradient projection `0.206912`
- Components projected on the Phase68 total: locomotion `1.3916`, pitch `-0.2043`, support `-0.2005`; the formal Phase69 total → positive pitch projection remained negative.
- Full old-critic terminal vs zero direction cosine: `-0.930713`
- Mean-centered old-critic terminal vs zero direction cosine: `0.999892`
- Most opposing independently normalized locomotion-term directions: `track_ang_vel_z_exp` (-1.0115), `track_lin_vel_xy_exp` (-0.9699), `feet_air_time` (-0.8918)
- Full old-critic sensitivity is retained for provenance; the centered sensitivity only isolates its state-varying part and is not a corrected estimator.
- This posthoc analysis does not change Phase69 or unlock Phase70 execution or any optimizer.
