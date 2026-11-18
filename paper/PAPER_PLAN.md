# Paper plan: ICWSM-DataHub (ICWSM 2027, Dataset Track)

**Working title.** ICWSM-DataHub: A Curated Meta-Dataset, Accessibility Audit, and Benchmark Catalog of Computational Social Science Datasets (2020–2026)

**One-sentence pitch.** We coded every ICWSM Dataset Track paper from 2020 to 2026, tested whether each dataset can still be used, and found that availability and usability have drifted apart. We release the catalog, the audit and a living explorer so the community can close that gap.

Check the current ICWSM call for papers for the dataset-paper page limit, required sections, checklist and deadlines before fixing the outline below.

## Contributions

1. **Resource.** A catalog of all Dataset Track papers: 65 extracted attributes plus deterministic, audit and bibliometric fields. Released as CSV, Parquet and JSON on Zenodo with a DOI, Croissant metadata, a data dictionary and a datasheet.
2. **Audit.** The first empirical test of whether ICWSM datasets can be retrieved and used years after publication. It goes beyond “does the link resolve” to gated, empty and login-walled records, and to IDs that need a closed API to rehydrate.
3. **Longitudinal analysis.** How platforms, collection routes, labeling and FAIR practice changed across seven years, including the 2023–24 API closures measured on data time.
4. **Method.** A verified LLM-extraction protocol: a controlled-vocabulary schema, verbatim evidence quotes checked against the source text, and blind human validation with per-field agreement. Other venues can reuse it to audit their own dataset tracks.

## Narrative arc (what we borrow from Hans Rosling)

Rosling’s talks work because they start with the audience’s wrong guess, then let one moving picture correct it, one number at a time. The paper follows the same arc.

1. **Open with the question, not the method.** First paragraph: “What share of datasets published at ICWSM since 2020 can a researcher download and use today, as described? Most people we asked guessed ___.” Optionally, poll 15–20 colleagues before writing; the gap between their guesses and the answer is a finding in itself.
2. **Answer it with Figure 1 (the funnel).** Every later section explains one drop in the funnel.
3. **One finding per section, one number per finding.** Section headings are claims (“Twitter/X fell from \kn{twitter_share_first} to \kn{twitter_share_last}”), not topics (“Platforms”).
4. **Time is the protagonist.** Every figure shows 2020–2026; the animated Gapminder chart is in the explorer and the conference talk.
5. **End with what to do.** Concrete recommendations for authors and for track chairs, each tied to a measured gap.

All numbers in the text come from `results/key_numbers.tex` (`\input` it, then write `\kn{pct_usable}`), so a re-audit never leaves a stale number in the paper.

## Research questions, evidence and figures

| RQ | Question | Evidence | Figure or table |
|---|---|---|---|
| RQ1 | What has the track collected, and how has its composition changed? | Platform families, modalities, languages, geography by year; diversity entropy | Fig 2a, Table 1 |
| RQ2 | Did the 2023–24 API closures change what is collected and how? | Platform shares by collection-end year; chi-square pre/post Feb 2023; trend in researcher programs and data donation | Fig 2b, Table 2 |
| RQ3 | How has labeling changed? | Label sources by year; LLM share; agreement reporting | Fig 4, Table 2 |
| RQ4 | What share of datasets is usable today, and what predicts survival? | Funnel; fate by cohort; logit of link survival on age, DOI, archival host | Fig 1, Fig 3, Table 3 |
| RQ5 | Which FAIR practices improved, and which did not? | 10 indicators by year | Fig 5 |
| RQ6 | Is usability associated with reuse? | Negative binomial of citations; Zenodo-only downloads by release form | Fig 7, Table 4 |
| RQ7 | Who studies whom? | Author regions vs. data regions | Fig 6 |

RQ6 is associational. Say so plainly: better-resourced teams may both archive well and attract citations.

## Outline

1. **Introduction.** The question; the funnel answer; contributions; the explorer link.
2. **Background.** Dataset documentation (Datasheets, Data Statements, Data Cards, Croissant); FAIR and its automated assessment (for example F-UJI); link-rot and reference-rot studies; the post-2023 data-access landscape (Twitter/X API, Pushshift, Reddit API, CrowdTangle, TikTok Research API, EU DSA Article 40); meta-research on dataset papers in other venues.
3. **Building the catalog.** Corpus and inclusion; schema design (controlled vocabularies, sentinels, evidence quotes); extraction with Claude; deterministic features; normalization; curation overrides.
4. **Validating the catalog.** Automatic grounding rates (quotes, URLs, DOIs); human double-coding; per-field accuracy and kappa (Table, from `02_verify.py --score`); known weak fields.
5. **The 2026 accessibility audit.** What counts as alive, gated, empty or dead; repository APIs; Wayback recovery; timing and re-runs.
6. **Findings.** One subsection per RQ, each headed by its claim.
7. **Recommendations.** For authors: a dataset DOI on an archival repository; open formats; a license; a datasheet; code to load the data; avoid IDs-only releases where the law and the platform terms allow. For the track: require a DOI at camera-ready; spot-check one year later; publish a living audit.
8. **The resource itself.** Formats, schema, Croissant, license, versioning, maintenance (monthly audit), explorer, how to contribute corrections.
9. **Limitations and ethics.**

Appendices: the full schema and codebook; the datasheet for this dataset; the prompt; the extraction cost and token counts; the per-field validation table.

## Validation protocol

1. **Pilot.** Extract 20 papers (three per year) in sync mode. Read every record against its PDF. Fix the codebook and prompt. Bump `PROMPT_VERSION`. Re-extract the pilot.
2. **Full extraction** with the frozen prompt and schema.
3. **Automatic grounding** (`02_verify.py`). Report the share of quotes, URLs and DOIs found in the source text. Hand-check every record that has a consistency flag or ungrounded URL, and fix it through `overrides.csv`.
4. **Human validation.** `02_verify.py --make-sample 60` draws about 18% of papers, stratified by year. Two coders work blind from the PDFs using `validation/CODEBOOK.md`.
   - Report human–human agreement first: it is the ceiling.
   - Then report LLM–human agreement against the adjudicated gold standard.
   - Give accuracy for every field, kappa for categorical fields and Jaccard for multi-label fields.
5. **Audit robustness.** Run the audit twice, at least a week apart. Classify a link as dead only if it fails both times. Report how many labels changed between the runs.
6. **Responsible disclosure.** Before release, email the corresponding authors of dead, empty or gated datasets with the audit result. Re-audit four weeks later and report how many datasets were restored after notification. This is both an ethical safeguard and a finding.

## Threats to validity

- **Extraction error.** Bounded by grounding checks and human validation; per-field accuracy is published, and low-confidence records are flagged in the catalog.
- **Audit timing.** A snapshot; transient outages are reduced by the double run; the monthly history makes this a time series.
- **Construct choices.** FAIR indicators and dataset fate are operationalizations. Report them alongside the raw components so readers can recombine them.
- **Heterogeneous counters.** Downloads are compared only within one repository; citations need a window, so control for age.
- **Scope.** One venue. ICWSM is a natural first case because it has had a dedicated dataset track since 2020; the pipeline is venue-agnostic.
- **Paper versus reality.** The catalog records what papers claim, and the audit records what exists. Disagreements between them are findings, not noise.

## Ethics notes for the checklist

- No human subjects: the material is published papers and public repository metadata.
- The audit sends a small number of polite, rate-limited requests, with a contact address in the User-Agent.
- Report link-rot findings in aggregate in the paper. The per-paper status in the explorer exists so readers can find working data, and it is accompanied by the notification step above.
- The catalog stores short evidence quotes and links to each paper; it does not redistribute papers or their full text.

## FAIR statement for this resource

Findable: Zenodo DOI, Croissant metadata, listed on the Hugging Face Hub. Accessible: open download, no account required. Interoperable: CSV, Parquet, JSON, documented controlled vocabularies. Reusable: CC BY 4.0, data dictionary, datasheet, versioned schema and prompt, code under MIT.

## Work plan

| Phase | Work | Output |
|---|---|---|
| 1 | Inventory, pilot extraction of 20 papers, codebook fixes | Frozen prompt and schema |
| 2 | Full batch extraction, grounding checks, overrides | Verified extraction |
| 3 | Human double-coding, adjudication, scoring | Validation table |
| 4 | Audit run 1, author notification, audit run 2, citations | Audit and history |
| 5 | Catalog, analysis, figures, explorer on GitHub Pages | All results |
| 6 | Writing, datasheet, Zenodo release, internal review | Submission |

## Anticipated reviewer questions

- **“Why trust an LLM to build a dataset about datasets?”** Because every key field is quote-grounded and automatically checked, a stratified sample is human-validated with agreement reported per field, and every correction is logged.
- **“Isn’t link rot already known?”** Link checking alone overstates usability. The novelty is the distance between a working link and usable data: gated, empty and login-walled records, and IDs that need a closed API.
- **“Is this just descriptive?”** The resource is the primary contribution, as the track asks. The analyses test specific hypotheses: a structural break in data time, survival by hosting, and reuse by release form.
- **“Will it stay current?”** The audit re-runs monthly through GitHub Actions, and new proceedings are added with the same pipeline.
