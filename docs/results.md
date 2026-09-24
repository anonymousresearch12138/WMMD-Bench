# Compact results and paper mapping

All reported Student measurements describe one training run with seed 42. Empty
CSV values/JSON null mean unreported, not zero. Strings beginning `N/A` retain
detector applicability limits. Native score directions and thresholds are in
`configs/evaluation/rules.json`.

| File in `results/` | Schema / paper use |
|---|---|
| `ownership_3b_seed42.csv` | 10 method rows; metric, Teacher/reference, clean Llama, Ba/Bb/Bc, clean Qwen, XBa/XBb; separate trajectory final value and exploratory EverTracer values. Table 1, Figure 3 and Figure 4. |
| `utility_3b_seed42.csv` | One recorded method/shared run and condition per row; ARC acc-norm, TruthfulQA MC2 and matched evaluation context. Appendix utility tables and Figure A1 deltas. |
| `active_trajectories_seed42.json` | Five method keys, 11 observations each; step, native detector fields, loss and its observation step, LR. Figure 2 and Appendix E. |
| `transformation_and_exposure.json` | Five active method keys; parent origin, paired counts, transformation modes, direct/transformed token totals and registered-signal exposure counts. Appendix C/F. |
| `transformation_shared.json` | Shared passive transformed-response processing counts (20k records), Appendix C. |
| `bc_native_details.json` | Method/field/value triples supplementing Bc's primary score. Appendix Bc details. |
| `extension_7b_seed42.json` | Eight PN-FP/AWM reference/Student rows; counts, native similarity, threshold, recall/retention and utility. Figure 3 7B panel, Appendix I tables and Figure A2. |

Rendering preserves full source precision; table rounding is presentation only.
Utility changes are Student minus the matched clean Qwen utility, not averaging
over runs. Native means (e.g. AWM layer score, iSeal BLEU or transformed-answer
length) retain their method-specific meanings.

`paper/tables/master.tex` reproduces the 80 Table 1 cells. Native-value tables retain
the unrounded values. Trajectory tables include the measured loss-observation step;
no missing step-0 loss is invented. 7B detector/utility tables are emitted separately.
The renderer produces all five numerical figure families (trajectories, active
supervision, passive lineage, utility and 7B extension). Layouts may differ from
the manuscript typesetting; data and displayed units must agree. The conceptual
Figure 1 is an illustration and is not a numerical experiment output.

The manuscript and this release report one seed-42 training run per Student
condition. Endpoints are descriptive single-seed measurements, not averages over
repeated training seeds. No experimental values were changed.
