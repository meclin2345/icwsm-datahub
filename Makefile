# ICWSM-DataHub pipeline. Run `make demo` first to see everything work on synthetic data.
PY ?= python3
PAPERS ?= data/papers

.PHONY: review review-apply demo inventory check-api extract-dry extract-pilot extract-all extract-batch extract-collect verify sample audit cite catalog analysis figures all test serve

demo:            ## full stack on synthetic data (no API key)
	$(PY) scripts/05_build_catalog.py --demo
	$(PY) scripts/06_analysis.py --demo
	$(PY) scripts/07_figures.py --demo --animate

inventory:       ## PDFs -> cached text + deterministic features
	$(PY) scripts/00_inventory.py --papers-dir $(PAPERS)

check-api:       ## probe the gateway once: models, budget, structured output, batches
	$(PY) scripts/check_api.py

extract-dry:     ## count tokens and estimate 5-hour windows before spending
	$(PY) scripts/01_extract.py --mode dry-run

extract-pilot:   ## 20 papers, read them by hand
	$(PY) scripts/01_extract.py --mode sync --limit 20

extract-all:     ## everything else (sync; waits for token-window resets, resumable)
	$(PY) scripts/01_extract.py --mode sync

extract-batch:   ## Anthropic API only: Batch API (Opus Code has none)
	$(PY) scripts/01_extract.py --mode batch-submit

extract-collect: ## download finished batches
	$(PY) scripts/01_extract.py --mode batch-collect

verify:
	$(PY) scripts/02_verify.py

sample:          ## blind human-coding sheet
	$(PY) scripts/02_verify.py --make-sample 60

review:          ## build data/curation/review_queue.csv from verification flags
	$(PY) scripts/review_flags.py

review-apply:    ## turn decisions into overrides (make review-apply CURATOR=MJ)
	$(PY) scripts/review_flags.py --apply --curator $(CURATOR)

audit-checklist: ## links the audit could not confirm, for a browser check
	$(PY) scripts/audit_checklist.py

doc-audit:       ## list documentation claims to classify
	$(PY) scripts/documentation_audit.py list

audit:
	$(PY) scripts/03_audit.py

cite:
	$(PY) scripts/04_enrich_citations.py --mailto $${AUDIT_CONTACT_EMAIL:?set AUDIT_CONTACT_EMAIL}

catalog:
	$(PY) scripts/05_build_catalog.py
	$(PY) scripts/08_social_card.py

analysis:
	$(PY) scripts/06_analysis.py

figures:
	$(PY) scripts/07_figures.py --animate

all: verify audit cite catalog analysis figures

test:
	$(PY) -m pytest -q tests

serve:           ## open http://localhost:8000
	$(PY) -m http.server -d docs 8000
