# ICWSM-DataHub data dictionary

Catalog version 1.0.0, schema version 2.0.0. 152 columns, 124 papers.

Source: **LLM** = extracted from the paper by Claude and verified (02_verify.py); **Code** = computed deterministically; **Audit** = observed in the repository audit; **OpenAlex** = bibliometrics.

| Column | Source | Type | Description |
|---|---|---|---|
| `paper_id` | Code | str | Stable identifier: <year>_<slug of PDF name>_<hash>. |
| `year` | Code | int64 | ICWSM proceedings year (from the folder the PDF came from). |
| `pdf_filename` | Code | str | Original PDF file name. |
| `title` | LLM | str | Full paper title exactly as printed. |
| `authors` | LLM | list | All author names in printed order. |
| `n_authors` | Code | int64 | Number of authors. |
| `author_countries` | LLM | list | ISO 3166-1 alpha-2 codes (uppercase) for countries of all author affiliations, deduplicated, e.g. ['US','IN']. [] if affiliations are not printed. |
| `author_regions` | Code | list | World-region(s) of author affiliations (World Bank-style regions). |
| `author_global_south` | Code | bool | True if any author affiliation is in the Global South (Brandt-line-style approximation, see normalize.py). |
| `is_industry_collaboration` | LLM | bool | True if at least one author lists a company or industry research lab affiliation. |
| `contribution_type` | LLM | str | Main type of resource the paper contributes. Values: New dataset / Extension or new version of an existing dataset / Benchmark or evaluation suite / Tool or infrastructure released with data / Ongoing archive or data collection effort / Other |
| `num_datasets_released` | LLM | int64 | Number of distinct datasets the paper releases. Usually 1. -1 if unclear. |
| `page_count` | Code | int64 | Number of PDF pages. |
| `checklist_detected` | Code | bool | True if the AAAI/ICWSM paper checklist text is found by regex. |
| `dataset_name` | LLM | str | Name or acronym the authors give the released dataset. '' if unnamed. |
| `platforms` | LLM | list | Every platform whose content is in the RELEASED dataset (not platforms only discussed). Values: Twitter/X / Reddit / YouTube / TikTok / Instagram / Facebook / Threads / Telegram / WhatsApp / Discord / Bluesky / Mastodon/Fediverse / 4chan/8kun/Imageboards / Alt-Tech (Gab/Parler/Truth Social/Rumble/BitChute) / LinkedIn / Twitch / Q&A and Forums (Quora/Stack Exchange/other forums) / Chinese platforms (Weibo/Douyin/Bilibili/WeChat) / Regional platforms (VK/ShareChat/Koo/Line/KakaoTalk) / Web/News/Blogs / Wikipedia/Wikidata / Search engines/Google Trends / AI chatbots and LLM output (ChatGPT logs, AI-generated content) / Other |
| `platform_families` | Code | list | Platforms grouped into 10 families used in figures and the explorer. |
| `n_platforms` | Code | int64 | Number of distinct platforms. |
| `platform_names_verbatim` | LLM | list | Platform names exactly as written in the paper, e.g. ['Twitter','X (formerly Twitter)']. |
| `collection_methods` | LLM | list | All methods used to gather the data. Values: Official platform API / Researcher access program (TikTok Research API, Meta Content Library, DSA Art. 40) / Third-party archive or service (Pushshift, CrowdTangle, Internet Archive, MediaCloud) / Web scraping or browser automation / Data donation (participants export their own data) / Survey or crowdsourced contribution / Rehydration or extension of an existing dataset / Synthetic, LLM-generated or simulated data / Not specified |
| `collection_tools` | LLM | list | Named APIs, libraries, services or tools used for collection, verbatim, e.g. ['Twitter Academic API v2','snscrape','CrowdTangle','yt-dlp']. [] if none named. |
| `closed_access_dependencies` | Code | list | Data-access routes used for collection that have since closed (e.g. Twitter/X Academic API, Pushshift, CrowdTangle). |
| `depends_on_closed_access` | Code | bool | True if collection relied on at least one closed access route. |
| `temporal_coverage_start` | LLM | str | Earliest date of content in the dataset as YYYY-MM-DD, YYYY-MM or YYYY. '' if not stated. |
| `temporal_coverage_end` | LLM | str | Latest date of content in the dataset as YYYY-MM-DD, YYYY-MM or YYYY. '' if not stated. |
| `coverage_end_year` | Code | float64 | Year of the last content in the dataset. |
| `coverage_days` | Code | float64 | Days between first and last content (null if either date missing). |
| `primary_record_unit` | LLM | str | The main unit counted in the dataset size. Values: Posts or tweets / Comments or replies / Users or accounts / Videos / Images or memes / News articles or web pages / Conversations or threads / Chat messages / Channels, groups or communities / Audio or podcast episodes / Survey responses or participants / Other |
| `total_records` | LLM | float64 | Total count of the primary unit as an integer (14.2 million -> 14200000). -1 if not stated. |
| `log10_records` | Code | float64 | log10 of total_records. |
| `num_accounts_or_users` | LLM | float64 | Number of distinct accounts, users or channels covered. -1 if not stated. |
| `scale_description_verbatim` | LLM | str | The authors' own phrase describing dataset size, e.g. '14.2M tweets from 450K users'. |
| `modalities` | LLM | list | Data modalities present in the RELEASED files. Values: Text / Image / Video / Audio / Network or graph / Tabular user or post metadata / Time series or temporal logs / Geospatial or mobility |
| `is_multimodal` | Code | bool | True if more than one content modality (text, image, video, audio) is released. |
| `languages_iso` | LLM | list | ISO 639-1 codes (or ISO 639-3 when no 639-1 exists) of languages in the data, lowercase. Use ['mul'] only when the paper says multilingual without listing languages. [] if not stated. |
| `num_languages` | LLM | float64 | Number of languages, if stated or countable. -1 if unknown. |
| `is_multilingual` | Code | bool | True if more than one language. |
| `includes_code_mixing` | LLM | bool | True only if the paper explicitly mentions code-mixed or code-switched text. |
| `data_geography` | LLM | list | ISO alpha-2 codes of the countries the data is ABOUT or collected FROM (not author countries). ['GLOBAL'] if worldwide without a focus. [] if not stated. |
| `data_regions` | Code | list | World-region(s) the data is about. |
| `data_global_south` | Code | bool | True if the data is about at least one Global South country. |
| `event_category` | LLM | str | Type of real-world event the dataset centres on, if any. Values: Elections / Pandemic or public health crisis / Armed conflict or war / Protest or social movement / Natural disaster or climate event / Economic or financial event / Sports or entertainment event / Platform policy or ownership change / No specific event |
| `event_name` | LLM | str | Specific event, e.g. '2022 Brazilian general election'. '' if none. |
| `involves_vulnerable_groups` | LLM | bool | True if the data centres on minors, patients, refugees, LGBTQ+ people, or other groups the paper itself flags as vulnerable. |
| `research_domains` | LLM | list | CSS research areas the dataset serves (1-3 values). Values: Misinformation and fact-checking / Hate speech, toxicity and harassment / Political communication and elections / Polarization and ideology / Public health and COVID-19 / Mental health and wellbeing / Bots, coordinated inauthentic behavior and propaganda / Extremism and radicalization / Crisis, disaster and conflict informatics / Social movements and activism / News and journalism / Content moderation and platform governance / Gender, race and identity / Online communities and collective behavior / Influencers, marketing and the platform economy / Science communication and education / Privacy and surveillance / AI-generated content and LLM use / Other |
| `supported_tasks` | LLM | list | Downstream tasks the authors propose or demonstrate. Values: Topic or content classification / Stance detection / Sentiment or emotion analysis / Hate or toxicity detection / Misinformation or claim detection / Bot or coordinated account detection / Network or information diffusion analysis / Virality or popularity prediction / Recommendation or ranking / Retrieval or question answering / Summarization or text generation / Entity or information extraction / Multimodal understanding (image/video/audio) / User profiling or demographic inference / Causal or intervention analysis / Descriptive or exploratory social analysis / Other |
| `tasks_verbatim` | LLM | list | Task names exactly as the authors phrase them. |
| `annotation_paradigms` | LLM | list | How labels or enrichments were produced. Use 'Unlabeled raw corpus' if no labels are released. Values: Unlabeled raw corpus / Expert, author or trained student annotation / Crowdworkers (MTurk, Prolific, etc.) / Distant or weak supervision (hashtags, domain lists, heuristics) / Automated classifier or lexicon (BERT, Perspective API, LIWC) / LLM or generative AI labels / Hybrid human and LLM-in-the-loop / Platform-native signals (Community Notes, moderation logs, votes) |
| `uses_llm_annotation` | Code | bool | True if labels come from an LLM or a hybrid human+LLM pipeline. |
| `uses_crowd` | Code | bool | True if crowdworkers annotated. |
| `uses_expert` | Code | bool | True if experts, authors or trained students annotated. |
| `llms_used` | LLM | list | Specific LLMs used for labeling or generating data, e.g. ['GPT-4o','Llama-3-70B']. [] if none. |
| `num_annotated_records` | LLM | float64 | Number of manually or automatically labeled items. -1 if none or not stated. |
| `num_annotators` | LLM | float64 | Number of human annotators per item or in total, as stated. -1 if not stated. |
| `iaa_reported` | LLM | bool | True if any inter-annotator agreement value is reported. |
| `iaa_metrics` | LLM | list | Agreement metrics reported. [] if none. Values: Cohen's kappa / Fleiss' kappa / Krippendorff's alpha / Percent agreement / Correlation or ICC / F1 or accuracy against gold standard / Other metric |
| `iaa_primary_value` | LLM | float64 | The main (or average) agreement value reported, as a number between -1 and 1 (convert percentages to 0-1). -1.0 if none. |
| `iaa_summary_verbatim` | LLM | str | Verbatim sentence or phrase reporting agreement. '' if none. |
| `baseline_families` | LLM | list | Families of models or analyses the authors run ON THIS DATASET in the paper. Values: No experiments / Descriptive statistics only / Statistical models (regression, time series) / Classical ML (SVM, logistic regression, random forest) / Fine-tuned transformer encoders (BERT family) / LLM prompting (zero- or few-shot) / Graph or network models / Vision, audio or multimodal models |
| `includes_baselines` | Code | bool | True if any model or analysis beyond descriptive statistics is run. |
| `release_status` | LLM | str | Whether the data was actually released when the paper was published. Values: Released at publication / Partially released / Promised or forthcoming / Available on request only / Not released / Unclear |
| `hosting_repositories` | LLM | list | Where the dataset files are hosted. Values: Zenodo / Dataverse (Harvard or other instance) / Figshare / Open Science Framework (OSF) / Hugging Face Hub / GitHub or GitLab / Kaggle / Domain archive (ICPSR, GESIS, Dryad, Mendeley Data) / Personal or lab website, Google Drive, Dropbox / Available on request only / Other / Not stated |
| `primary_host` | Code | str | First listed hosting venue, shortened. |
| `dataset_urls` | LLM | list | Every URL pointing to the dataset, its landing page or project page. Repair URLs split across lines or hyphenated by the PDF. Do NOT include URLs of other datasets cited in related work. |
| `persistent_doi` | LLM | str | Dataset DOI as a bare string, e.g. '10.5281/zenodo.1234567'. Not the paper's own DOI. '' if none. |
| `has_doi` | Code | bool | True if a dataset DOI (not the paper DOI) is reported. |
| `code_repository_url` | LLM | str | URL of code for collection, hydration or baselines. '' if none. |
| `distribution_form` | LLM | str | What content is shared: full posts, scrubbed full posts, IDs needing re-collection, or derived data. Values: Full raw content / Anonymized or PII-scrubbed full content / Dehydrated IDs only / Aggregated statistics, embeddings or derived features only / Mixed (IDs plus derived features) / Not specified |
| `access_mechanism` | LLM | str | How a researcher obtains the files. Values: Open public download / Credentialed access, form or data use agreement / Email the authors for approval / Not specified |
| `license_verbatim` | LLM | str | License exactly as named in the paper, e.g. 'CC BY-NC 4.0'. '' if none. |
| `license_category` | LLM | str | License family of the dataset. Values: CC0 or public domain / CC BY / CC BY-SA / CC BY-NC (any NC variant) / Open Data Commons (ODbL, ODC-By, PDDL) / Software license (MIT, Apache, GPL, BSD) / Custom terms or data use agreement / Platform terms of service only / Not specified |
| `data_file_formats` | LLM | list | File formats of the release, uppercase, e.g. ['CSV','JSONL','PARQUET']. [] if not stated. |
| `explicitly_mentions_fair` | LLM | bool | True if the paper names the FAIR principles. |
| `documentation_framework` | LLM | str | Standard documentation shipped with the data. Values: Datasheets for Datasets / Data Statements / Data Card or dataset card / Dataset Nutrition Label / Croissant or other machine-readable metadata / Custom README or codebook / None mentioned |
| `provides_collection_or_hydration_code` | LLM | bool | True if code to collect, hydrate or load the data is released. |
| `irb_status` | LLM | str | Outcome of institutional ethics review as reported. Values: Approved by IRB or ethics board / Exempt or not human-subjects research (as determined by a board) / Self-assessed or checklist only / Not mentioned |
| `pii_anonymization_applied` | LLM | bool | True if usernames, user IDs, mentions or other PII were removed, hashed or masked before release. |
| `anonymization_details` | LLM | str | Short description of privacy steps. '' if none. |
| `discusses_platform_tos` | LLM | bool | True if compliance with platform terms of service is discussed. |
| `discusses_limitations` | LLM | bool | True if the paper discusses biases or limitations of the dataset. |
| `discusses_misuse_risks` | LLM | bool | True if potential misuse or harms of releasing the data are discussed. |
| `includes_paper_checklist` | LLM | bool | True if the AAAI/ICWSM paper or ethics checklist appears in the text. |
| `ai_usage_disclosed` | LLM | bool | True if the authors disclose using generative AI for writing or coding. |
| `availability_quote` | LLM | str | Verbatim sentence (max 50 words) stating where or how the dataset can be accessed. '' if none. |
| `scale_quote` | LLM | str | Verbatim sentence (max 50 words) stating dataset size or collection period. '' if none. |
| `annotation_quote` | LLM | str | Verbatim sentence (max 50 words) describing how labels were produced. '' if none. |
| `ethics_quote` | LLM | str | Verbatim sentence (max 50 words) on ethics review or privacy. '' if none. |
| `license_quote` | LLM | str | Verbatim sentence (max 50 words) naming the license. '' if none. |
| `extraction_confidence` | LLM | str | How clearly the paper reports its dataset metadata. Values: High / Medium / Low |
| `ambiguity_notes` | LLM | str | One or two sentences naming fields that were contradictory, vague or missing. '' if none. |
| `quotes_grounded_share` | Code | float64 | Share of evidence quotes found (near-)verbatim in the paper text. |
| `urls_grounded_share` | Code | float64 | Share of extracted dataset URLs that appear in the paper text. |
| `n_flags` | Code | int64 | Number of automatic consistency flags (see 02_verify.py). |
| `flags` | Code | str | Automatic consistency flags raised by 02_verify.py. |
| `audit_manual_check` | Code | str | Date and note of a manual browser check that replaced the automatic link result. |
| `audited_at` | Audit | str | UTC time of the repository audit. |
| `any_url_alive` | Audit | bool | True if at least one dataset link or the DOI resolved to live content. |
| `primary_url_status` | Audit | str | Outcome category for the first dataset link (Alive, Dead (404), Soft 404, Login wall, ...). |
| `doi_resolves` | Audit | bool | True if https://doi.org/<doi> resolved to live content. |
| `verified_repo` | Audit | str | Repository whose API confirmed the record (Zenodo, Dataverse, Figshare, Hugging Face, GitHub, DataCite). |
| `file_count` | Audit | float64 | Number of files in the repository record (API). |
| `total_size_mb` | Audit | float64 | Total size of files in MB (API). |
| `file_extensions` | Audit | str | File extensions observed in the repository (API). |
| `is_restricted` | Audit | bool | True if all files are restricted, embargoed or need manual approval. |
| `files_empty` | Audit | bool | True if the repository record exists but contains no files. |
| `login_wall` | Audit | bool | True if links only lead to a sign-in page. |
| `account_required` | Audit | bool | True if a free account is needed (Kaggle, gated Hugging Face). |
| `downloads_total` | Audit | float64 | Total downloads reported by the repository (definitions differ across repositories). |
| `downloads_unique` | Audit | float64 | Unique downloads where reported (Zenodo, Dataverse). |
| `views_total` | Audit | float64 | Total views reported by the repository. |
| `downloads_metric_source` | Audit | str | Which counter downloads_total comes from. Compare downloads only within one source. |
| `datacite_registered` | Audit | bool | True if the DOI is registered with DataCite. |
| `datacite_citations` | Audit | float64 | Citations to the dataset DOI recorded by DataCite. |
| `code_url_alive` | Audit | object | True if the code repository link resolves. |
| `code_stars` | Audit | float64 | GitHub stars of the code repository. |
| `dead_link_in_wayback` | Audit | object | For dead links: True if the Wayback Machine holds a snapshot of the landing page. |
| `paper_doi` | OpenAlex | str | DOI of the ICWSM paper (OpenAlex). |
| `citations` | OpenAlex | float64 | Citations to the ICWSM paper (OpenAlex). |
| `years_since_publication` | Code | float64 | Years between June of the ICWSM year and the audit date. |
| `citations_per_year` | Code | float64 | citations divided by years since publication. |
| `downloads_per_year` | Code | float64 | downloads_total divided by years_since_publication. |
| `fair_F1` | Code + Audit | bool | FAIR indicator F1. Persistent identifier: A dataset DOI is reported and is registered with DataCite or resolves. |
| `fair_F2` | Code + Audit | bool | FAIR indicator F2. Rich metadata: A standard documentation framework (datasheet, data card, data statement, Croissant) is used. |
| `fair_F4` | Code + Audit | bool | FAIR indicator F4. Indexed: The dataset DOI is registered and findable in DataCite. |
| `fair_A1` | Code + Audit | bool | FAIR indicator A1. Retrievable: At least one dataset link resolves in the 2026 audit. |
| `fair_A1_1` | Code + Audit | bool | FAIR indicator A1.1. Open protocol: Files can be downloaded openly, without forms, approval or login walls. |
| `fair_A2` | Code + Audit | bool | FAIR indicator A2. Persistent hosting: Hosted on an archival repository (Zenodo, Dataverse, Figshare, OSF, ICPSR, GESIS...). |
| `fair_I1` | Code + Audit | bool | FAIR indicator I1. Open formats: Released in open, widely readable formats (CSV, JSON(L), Parquet, GraphML...). |
| `fair_R1_1` | Code + Audit | bool | FAIR indicator R1.1. Clear license: A license or explicit terms of use are stated. |
| `fair_R1_2` | Code + Audit | bool | FAIR indicator R1.2. Provenance: Collection method and temporal coverage (start and end) are reported. |
| `fair_R1_3` | Code + Audit | bool | FAIR indicator R1.3. Reusable pipeline: Code to collect, hydrate or load the data is released. |
| `fair_score` | Code | float64 | Share (0-100) of the 10 operational FAIR indicators met. |
| `fate_2026` | Code | str | Mutually exclusive 2026 state: Open and usable; Open, needs closed API to rehydrate; Restricted access (request or agreement); Alive but empty or login-walled; Link dead; Never released / no link. |
| `needs_closed_api` | Code | bool | True if content must be rehydrated through a platform API that is no longer free (Twitter/X). |
| `funnel_release_claimed` | Code | bool | Paper claims a release (now, partially, later or on request). |
| `funnel_link_provided` | Code | bool | Paper gives a dataset link or DOI. |
| `funnel_link_alive` | Code | bool | Link alive in the audit. |
| `funnel_open_download` | Code | bool | Files openly downloadable (not gated, empty or login-walled). |
| `funnel_usable` | Code | bool | Openly downloadable and usable without a closed API. |
| `n_manual_overrides` | Code | int64 | Number of fields corrected by human curators. |
| `body_mentions_misuse` | Code | bool | Regex: misuse/harm language appears in the main text (before References), i.e. not only in the checklist. |
| `body_mentions_fair` | Code | bool | Regex: 'FAIR' (upper case) appears in the main text (before References), i.e. not only in the checklist. |
| `body_mentions_datasheet` | Code | bool | Regex: datasheet / data card / data statement named in the main text (before References). |
| `repo_license` | Code | str | License recorded on the repository record itself (Zenodo, Dataverse, Figshare, Hugging Face, GitHub). |
| `repo_license_present` | Code | object | True if the repository record carries a license (only for records the audit could verify). |
| `model` | Code | str | LLM requested for extraction. |
| `served_model` | Code | str | Model id the API reported serving for this record. |
| `api_host` | Code | object | API host used for extraction (gateway or Anthropic API). |
| `structured_mode` | Code | str | native (constrained decoding), prompt (schema in prompt + validation) or prompt-repair. |
| `prompt_version` | Code | str | Extraction prompt version. |
| `schema_version` | Code | str | Extraction schema version. |
| `extracted_at` | Code | str | UTC time of extraction. |
