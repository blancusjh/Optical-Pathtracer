Raw linear PFM; full-image relMSE is the target. No denoising.

Non-transport time includes scene loading, compilation, upload, resolve and output; it is not a load-only timer.

| scene | estimator | spp | transport s | wall s | non-transport s | peak host MiB | relMSE | masked relMSE | mean/ref | Mpaths/s | inconsistencies | leaks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| the_lens | path | 16 | 0.017872 | 0.34856 | 0.33069 | 322.15 | 0.066301 | 0.06257 | 0.93388 | 44.003 | 3 | 2 |
| the_lens | path | 64 | 0.069931 | 0.41094 | 0.34101 | 322.23 | 0.61742 | 0.019693 | 0.9337 | 44.984 | 5 | 2 |
| prism_in_sunlight | path | 16 | 0.019321 | 0.38773 | 0.36841 | 321.72 | 0.17845 | 0.17845 | 0.7515 | 40.703 | 5 | 0 |
| prism_in_sunlight | path | 64 | 0.077121 | 0.45226 | 0.37514 | 321.82 | 0.044361 | 0.044361 | 0.7515 | 40.79 | 6 | 4 |
| the_peacock_lamp | vcm | 16 | 18.396 | 40.974 | 22.578 | 2621.2 | 0.9353 | 0.89424 | 0.89889 | 1.002 | 1.1628e+05 | 0 |
| the_peacock_lamp | vcm | 64 | 63.268 | 84.976 | 21.708 | 2619.6 | 4.0305 | 0.25632 | 0.88699 | 1.1653 | 4.6657e+05 | 0 |
