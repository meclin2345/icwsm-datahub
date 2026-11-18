"""
ICWSM-DataHub extraction schema (v2).

Design rules
------------
1. Everything the LLM fills is REQUIRED. Claude structured outputs cap a schema at
   24 optional fields and 16 union (Optional/anyOf) fields, and every optional field
   enlarges the compiled grammar. "Unknown" is therefore encoded with sentinels:
       strings -> ""        integers -> -1        floats -> -1.0      lists -> []
   `normalize.py` converts sentinels back to None after extraction.
2. Every attribute that will be plotted is a controlled-vocabulary Enum, so
   "Twitter", "X" and "Twitter API v2" never have to be reconciled by hand.
   Free-text "verbatim" siblings keep the author's own wording for auditing.
3. Fields the code can compute deterministically (year, page count, file hash,
   checklist presence, URLs in text) are NOT asked of the LLM. See pdftext.py.
4. Key claims carry a verbatim evidence quote, which 02_verify.py checks
   against the PDF text to catch hallucinations.

Field descriptions are sent to the model as part of the JSON schema, so they double
as the annotation codebook.
"""
from __future__ import annotations

import copy
import difflib
import re
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "2.0.0"
UNKNOWN_INT = -1
UNKNOWN_FLOAT = -1.0
UNKNOWN_STR = ""


ENUM_COERCIONS: List[tuple] = []   # (enum, value the model wrote, member it was mapped to); written out after a run


def _enum_key(text: str) -> str:
    """Comparison key: lower-case, parenthetical examples removed, whitespace collapsed."""
    text = re.sub(r"\s*\([^)]*\)", "", str(text))
    return " ".join(text.lower().replace("\u2013", "-").split()).strip(" .")


class CIEnum(str, Enum):
    """String enum that tolerates harmless variation in model output.

    Without constrained decoding the model may write 'Crowdworkers' or 'Distant or weak supervision'
    for 'Crowdworkers (MTurk, Prolific, etc.)'. Matching order:
      1. case-insensitive exact value or member name
      2. same text once parenthetical examples are removed
      3. unique prefix (model shortened the label, or added words)
      4. a single very close fuzzy match
    Anything else still fails validation, which triggers the repair round. Every coercion
    is recorded in ENUM_COERCIONS so the prompt can be fixed at the source.
    """

    @classmethod
    def _missing_(cls, value: object):
        if not isinstance(value, str):
            return None
        v = " ".join(value.strip().lower().split())
        for member in cls:
            if member.value.lower() == v or member.name.lower() == v:
                return member
        k = _enum_key(value)
        if not k:
            return None
        keys = {member: _enum_key(member.value) for member in cls}
        hit = [m for m, mk in keys.items() if mk == k]
        if len(hit) != 1:
            hit = [m for m, mk in keys.items() if (len(k) >= 6 and mk.startswith(k)) or (len(mk) >= 12 and k.startswith(mk))]
        if len(hit) != 1:
            close = difflib.get_close_matches(k, list(keys.values()), n=2, cutoff=0.86)
            hit = [m for m, mk in keys.items() if len(close) == 1 and mk == close[0]]
        if len(hit) == 1:
            ENUM_COERCIONS.append((cls.__name__, value, hit[0].value))
            return hit[0]
        return None


# ==========================================================================
# Controlled vocabularies
# ==========================================================================

class ContributionType(CIEnum):
    NEW_DATASET = "New dataset"
    EXTENSION = "Extension or new version of an existing dataset"
    BENCHMARK = "Benchmark or evaluation suite"
    TOOL_WITH_DATA = "Tool or infrastructure released with data"
    ARCHIVE_COLLECTION = "Ongoing archive or data collection effort"
    OTHER = "Other"


class PlatformType(CIEnum):
    TWITTER_X = "Twitter/X"
    REDDIT = "Reddit"
    YOUTUBE = "YouTube"
    TIKTOK = "TikTok"
    INSTAGRAM = "Instagram"
    FACEBOOK = "Facebook"
    THREADS = "Threads"
    TELEGRAM = "Telegram"
    WHATSAPP = "WhatsApp"
    DISCORD = "Discord"
    BLUESKY = "Bluesky"
    MASTODON = "Mastodon/Fediverse"
    CHAN_FORUMS = "4chan/8kun/Imageboards"
    ALT_TECH = "Alt-Tech (Gab/Parler/Truth Social/Rumble/BitChute)"
    LINKEDIN = "LinkedIn"
    TWITCH = "Twitch"
    QA_FORUMS = "Q&A and Forums (Quora/Stack Exchange/other forums)"
    CHINESE = "Chinese platforms (Weibo/Douyin/Bilibili/WeChat)"
    REGIONAL = "Regional platforms (VK/ShareChat/Koo/Line/KakaoTalk)"
    WEB_NEWS = "Web/News/Blogs"
    WIKIPEDIA = "Wikipedia/Wikidata"
    SEARCH = "Search engines/Google Trends"
    AI_PLATFORMS = "AI chatbots and LLM output (ChatGPT logs, AI-generated content)"
    OTHER = "Other"


class CollectionMethod(CIEnum):
    OFFICIAL_API = "Official platform API"
    RESEARCHER_PROGRAM = "Researcher access program (TikTok Research API, Meta Content Library, DSA Art. 40)"
    THIRD_PARTY_ARCHIVE = "Third-party archive or service (Pushshift, CrowdTangle, Internet Archive, MediaCloud)"
    WEB_SCRAPING = "Web scraping or browser automation"
    DATA_DONATION = "Data donation (participants export their own data)"
    SURVEY_CROWDSOURCING = "Survey or crowdsourced contribution"
    EXISTING_DATASET = "Rehydration or extension of an existing dataset"
    SYNTHETIC = "Synthetic, LLM-generated or simulated data"
    NOT_SPECIFIED = "Not specified"


class RecordUnit(CIEnum):
    POSTS = "Posts or tweets"
    COMMENTS = "Comments or replies"
    USERS = "Users or accounts"
    VIDEOS = "Videos"
    IMAGES = "Images or memes"
    ARTICLES = "News articles or web pages"
    THREADS = "Conversations or threads"
    MESSAGES = "Chat messages"
    COMMUNITIES = "Channels, groups or communities"
    AUDIO = "Audio or podcast episodes"
    SURVEY = "Survey responses or participants"
    OTHER = "Other"


class ModalityType(CIEnum):
    TEXT = "Text"
    IMAGE = "Image"
    VIDEO = "Video"
    AUDIO = "Audio"
    NETWORK = "Network or graph"
    METADATA = "Tabular user or post metadata"
    TIME_SERIES = "Time series or temporal logs"
    GEOSPATIAL = "Geospatial or mobility"


class EventCategory(CIEnum):
    ELECTIONS = "Elections"
    HEALTH_CRISIS = "Pandemic or public health crisis"
    ARMED_CONFLICT = "Armed conflict or war"
    PROTEST = "Protest or social movement"
    DISASTER = "Natural disaster or climate event"
    ECONOMIC = "Economic or financial event"
    ENTERTAINMENT = "Sports or entertainment event"
    PLATFORM_CHANGE = "Platform policy or ownership change"
    NONE = "No specific event"


class ResearchDomain(CIEnum):
    MISINFORMATION = "Misinformation and fact-checking"
    HATE_TOXICITY = "Hate speech, toxicity and harassment"
    POLITICAL_COMM = "Political communication and elections"
    POLARIZATION = "Polarization and ideology"
    PUBLIC_HEALTH = "Public health and COVID-19"
    MENTAL_HEALTH = "Mental health and wellbeing"
    INAUTHENTIC = "Bots, coordinated inauthentic behavior and propaganda"
    EXTREMISM = "Extremism and radicalization"
    CRISIS = "Crisis, disaster and conflict informatics"
    ACTIVISM = "Social movements and activism"
    NEWS = "News and journalism"
    MODERATION = "Content moderation and platform governance"
    IDENTITY = "Gender, race and identity"
    COMMUNITIES = "Online communities and collective behavior"
    ECONOMY = "Influencers, marketing and the platform economy"
    SCIENCE_EDU = "Science communication and education"
    PRIVACY = "Privacy and surveillance"
    AI_CONTENT = "AI-generated content and LLM use"
    OTHER = "Other"


class TaskType(CIEnum):
    CLASSIFICATION = "Topic or content classification"
    STANCE = "Stance detection"
    SENTIMENT = "Sentiment or emotion analysis"
    TOXICITY = "Hate or toxicity detection"
    MISINFO = "Misinformation or claim detection"
    BOT = "Bot or coordinated account detection"
    NETWORK = "Network or information diffusion analysis"
    POPULARITY = "Virality or popularity prediction"
    RECOMMENDATION = "Recommendation or ranking"
    RETRIEVAL = "Retrieval or question answering"
    GENERATION = "Summarization or text generation"
    EXTRACTION = "Entity or information extraction"
    MULTIMODAL = "Multimodal understanding (image/video/audio)"
    USER_PROFILING = "User profiling or demographic inference"
    CAUSAL = "Causal or intervention analysis"
    DESCRIPTIVE = "Descriptive or exploratory social analysis"
    OTHER = "Other"


class AnnotationParadigm(CIEnum):
    UNLABELED = "Unlabeled raw corpus"
    EXPERT = "Expert, author or trained student annotation"
    CROWD = "Crowdworkers (MTurk, Prolific, etc.)"
    DISTANT = "Distant or weak supervision (hashtags, domain lists, heuristics)"
    CLASSICAL_ML = "Automated classifier or lexicon (BERT, Perspective API, LIWC)"
    LLM = "LLM or generative AI labels"
    HYBRID = "Hybrid human and LLM-in-the-loop"
    PLATFORM_NATIVE = "Platform-native signals (Community Notes, moderation logs, votes)"


class IAAMetric(CIEnum):
    COHENS_KAPPA = "Cohen's kappa"
    FLEISS_KAPPA = "Fleiss' kappa"
    KRIPPENDORFF = "Krippendorff's alpha"
    PERCENT = "Percent agreement"
    CORRELATION = "Correlation or ICC"
    F1_ACCURACY = "F1 or accuracy against gold standard"
    OTHER = "Other metric"


class BaselineFamily(CIEnum):
    NONE = "No experiments"
    DESCRIPTIVE = "Descriptive statistics only"
    STATISTICAL = "Statistical models (regression, time series)"
    CLASSICAL_ML = "Classical ML (SVM, logistic regression, random forest)"
    TRANSFORMER = "Fine-tuned transformer encoders (BERT family)"
    LLM_PROMPTING = "LLM prompting (zero- or few-shot)"
    GRAPH = "Graph or network models"
    MULTIMODAL = "Vision, audio or multimodal models"


class ReleaseStatus(CIEnum):
    RELEASED = "Released at publication"
    PARTIAL = "Partially released"
    FORTHCOMING = "Promised or forthcoming"
    ON_REQUEST = "Available on request only"
    NOT_RELEASED = "Not released"
    UNCLEAR = "Unclear"


class HostingRepository(CIEnum):
    ZENODO = "Zenodo"
    DATAVERSE = "Dataverse (Harvard or other instance)"
    FIGSHARE = "Figshare"
    OSF = "Open Science Framework (OSF)"
    HUGGING_FACE = "Hugging Face Hub"
    GITHUB = "GitHub or GitLab"
    KAGGLE = "Kaggle"
    DOMAIN_ARCHIVE = "Domain archive (ICPSR, GESIS, Dryad, Mendeley Data)"
    PERSONAL_CLOUD = "Personal or lab website, Google Drive, Dropbox"
    ON_REQUEST = "Available on request only"
    OTHER = "Other"
    NOT_FOUND = "Not stated"


class DistributionForm(CIEnum):
    FULL_RAW = "Full raw content"
    ANONYMIZED = "Anonymized or PII-scrubbed full content"
    DEHYDRATED = "Dehydrated IDs only"
    AGGREGATED = "Aggregated statistics, embeddings or derived features only"
    MIXED = "Mixed (IDs plus derived features)"
    NOT_SPECIFIED = "Not specified"


class AccessMechanism(CIEnum):
    OPEN = "Open public download"
    CREDENTIALED = "Credentialed access, form or data use agreement"
    AUTHOR_APPROVAL = "Email the authors for approval"
    NOT_SPECIFIED = "Not specified"


class LicenseCategory(CIEnum):
    CC0 = "CC0 or public domain"
    CC_BY = "CC BY"
    CC_BY_SA = "CC BY-SA"
    CC_BY_NC = "CC BY-NC (any NC variant)"
    ODC = "Open Data Commons (ODbL, ODC-By, PDDL)"
    CODE_LICENSE = "Software license (MIT, Apache, GPL, BSD)"
    CUSTOM = "Custom terms or data use agreement"
    PLATFORM_TOS = "Platform terms of service only"
    NOT_SPECIFIED = "Not specified"


class DocumentationFramework(CIEnum):
    DATASHEET = "Datasheets for Datasets"
    DATA_STATEMENT = "Data Statements"
    DATA_CARD = "Data Card or dataset card"
    NUTRITION_LABEL = "Dataset Nutrition Label"
    CROISSANT = "Croissant or other machine-readable metadata"
    README_CODEBOOK = "Custom README or codebook"
    NONE = "None mentioned"


class IRBStatus(CIEnum):
    APPROVED = "Approved by IRB or ethics board"
    EXEMPT = "Exempt or not human-subjects research (as determined by a board)"
    SELF_ASSESSED = "Self-assessed or checklist only"
    NOT_MENTIONED = "Not mentioned"


class Confidence(CIEnum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


# ==========================================================================
# LLM-facing sections. Every field is required (see module docstring).
# ==========================================================================

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class PaperInfo(_Strict):
    title: str = Field(description="Full paper title exactly as printed.")
    authors: List[str] = Field(description="All author names in printed order.")
    author_countries: List[str] = Field(
        description="ISO 3166-1 alpha-2 codes (uppercase) for countries of all author affiliations, "
        "deduplicated, e.g. ['US','IN']. [] if affiliations are not printed."
    )
    is_industry_collaboration: bool = Field(
        description="True if at least one author lists a company or industry research lab affiliation."
    )
    contribution_type: ContributionType = Field(description="Main type of resource the paper contributes.")
    num_datasets_released: int = Field(
        description="Number of distinct datasets the paper releases. Usually 1. -1 if unclear."
    )


class CollectionSpecs(_Strict):
    dataset_name: str = Field(description="Name or acronym the authors give the released dataset. '' if unnamed.")
    platforms: List[PlatformType] = Field(
        description="Every platform whose content is in the RELEASED dataset (not platforms only discussed)."
    )
    platform_names_verbatim: List[str] = Field(
        description="Platform names exactly as written in the paper, e.g. ['Twitter','X (formerly Twitter)']."
    )
    collection_methods: List[CollectionMethod] = Field(description="All methods used to gather the data.")
    collection_tools: List[str] = Field(
        description="Named APIs, libraries, services or tools used for collection, verbatim, e.g. "
        "['Twitter Academic API v2','snscrape','CrowdTangle','yt-dlp']. [] if none named."
    )
    temporal_coverage_start: str = Field(
        description="Earliest date of content in the dataset as YYYY-MM-DD, YYYY-MM or YYYY. '' if not stated."
    )
    temporal_coverage_end: str = Field(
        description="Latest date of content in the dataset as YYYY-MM-DD, YYYY-MM or YYYY. '' if not stated."
    )
    primary_record_unit: RecordUnit = Field(description="The main unit counted in the dataset size.")
    total_records: int = Field(
        description="Total count of the primary unit as an integer (14.2 million -> 14200000). -1 if not stated."
    )
    num_accounts_or_users: int = Field(
        description="Number of distinct accounts, users or channels covered. -1 if not stated."
    )
    scale_description_verbatim: str = Field(
        description="The authors' own phrase describing dataset size, e.g. '14.2M tweets from 450K users'."
    )
    modalities: List[ModalityType] = Field(description="Data modalities present in the RELEASED files.")
    languages_iso: List[str] = Field(
        description="ISO 639-1 codes (or ISO 639-3 when no 639-1 exists) of languages in the data, lowercase. "
        "Use ['mul'] only when the paper says multilingual without listing languages. [] if not stated."
    )
    num_languages: int = Field(description="Number of languages, if stated or countable. -1 if unknown.")
    includes_code_mixing: bool = Field(description="True only if the paper explicitly mentions code-mixed or code-switched text.")
    data_geography: List[str] = Field(
        description="ISO alpha-2 codes of the countries the data is ABOUT or collected FROM (not author countries). "
        "['GLOBAL'] if worldwide without a focus. [] if not stated."
    )
    event_category: EventCategory = Field(description="Type of real-world event the dataset centres on, if any.")
    event_name: str = Field(description="Specific event, e.g. '2022 Brazilian general election'. '' if none.")
    involves_vulnerable_groups: bool = Field(
        description="True if the data centres on minors, patients, refugees, LGBTQ+ people, or other groups "
        "the paper itself flags as vulnerable."
    )


class AnnotationAndTasks(_Strict):
    research_domains: List[ResearchDomain] = Field(description="CSS research areas the dataset serves (1-3 values).")
    supported_tasks: List[TaskType] = Field(description="Downstream tasks the authors propose or demonstrate.")
    tasks_verbatim: List[str] = Field(description="Task names exactly as the authors phrase them.")
    annotation_paradigms: List[AnnotationParadigm] = Field(
        description="How labels or enrichments were produced. Use 'Unlabeled raw corpus' if no labels are released."
    )
    llms_used: List[str] = Field(
        description="Specific LLMs used for labeling or generating data, e.g. ['GPT-4o','Llama-3-70B']. [] if none."
    )
    num_annotated_records: int = Field(description="Number of manually or automatically labeled items. -1 if none or not stated.")
    num_annotators: int = Field(description="Number of human annotators per item or in total, as stated. -1 if not stated.")
    iaa_reported: bool = Field(description="True if any inter-annotator agreement value is reported.")
    iaa_metrics: List[IAAMetric] = Field(description="Agreement metrics reported. [] if none.")
    iaa_primary_value: float = Field(
        description="The main (or average) agreement value reported, as a number between -1 and 1 "
        "(convert percentages to 0-1). -1.0 if none."
    )
    iaa_summary_verbatim: str = Field(description="Verbatim sentence or phrase reporting agreement. '' if none.")
    baseline_families: List[BaselineFamily] = Field(
        description="Families of models or analyses the authors run ON THIS DATASET in the paper."
    )


class FAIRAndDistribution(_Strict):
    release_status: ReleaseStatus = Field(description="Whether the data was actually released when the paper was published.")
    hosting_repositories: List[HostingRepository] = Field(description="Where the dataset files are hosted.")
    dataset_urls: List[str] = Field(
        description="Every URL pointing to the dataset, its landing page or project page. Repair URLs split "
        "across lines or hyphenated by the PDF. Do NOT include URLs of other datasets cited in related work."
    )
    persistent_doi: str = Field(
        description="Dataset DOI as a bare string, e.g. '10.5281/zenodo.1234567'. Not the paper's own DOI. '' if none."
    )
    code_repository_url: str = Field(description="URL of code for collection, hydration or baselines. '' if none.")
    distribution_form: DistributionForm = Field(
        description="What content is shared: full posts, scrubbed full posts, IDs needing re-collection, or derived data."
    )
    access_mechanism: AccessMechanism = Field(description="How a researcher obtains the files.")
    license_verbatim: str = Field(description="License exactly as named in the paper, e.g. 'CC BY-NC 4.0'. '' if none.")
    license_category: LicenseCategory = Field(description="License family of the dataset.")
    data_file_formats: List[str] = Field(
        description="File formats of the release, uppercase, e.g. ['CSV','JSONL','PARQUET']. [] if not stated."
    )
    explicitly_mentions_fair: bool = Field(description="True if the paper names the FAIR principles.")
    documentation_framework: DocumentationFramework = Field(description="Standard documentation shipped with the data.")
    provides_collection_or_hydration_code: bool = Field(
        description="True if code to collect, hydrate or load the data is released."
    )


class EthicsAndGovernance(_Strict):
    irb_status: IRBStatus = Field(description="Outcome of institutional ethics review as reported.")
    pii_anonymization_applied: bool = Field(
        description="True if usernames, user IDs, mentions or other PII were removed, hashed or masked before release."
    )
    anonymization_details: str = Field(description="Short description of privacy steps. '' if none.")
    discusses_platform_tos: bool = Field(description="True if compliance with platform terms of service is discussed.")
    discusses_limitations: bool = Field(description="True if the paper discusses biases or limitations of the dataset.")
    discusses_misuse_risks: bool = Field(description="True if potential misuse or harms of releasing the data are discussed.")
    includes_paper_checklist: bool = Field(description="True if the AAAI/ICWSM paper or ethics checklist appears in the text.")
    ai_usage_disclosed: bool = Field(description="True if the authors disclose using generative AI for writing or coding.")


class EvidenceAudit(_Strict):
    availability_quote: str = Field(
        description="Verbatim sentence (max 50 words) stating where or how the dataset can be accessed. '' if none."
    )
    scale_quote: str = Field(description="Verbatim sentence (max 50 words) stating dataset size or collection period. '' if none.")
    annotation_quote: str = Field(description="Verbatim sentence (max 50 words) describing how labels were produced. '' if none.")
    ethics_quote: str = Field(description="Verbatim sentence (max 50 words) on ethics review or privacy. '' if none.")
    license_quote: str = Field(description="Verbatim sentence (max 50 words) naming the license. '' if none.")
    extraction_confidence: Confidence = Field(description="How clearly the paper reports its dataset metadata.")
    ambiguity_notes: str = Field(
        description="One or two sentences naming fields that were contradictory, vague or missing. '' if none."
    )


class ICWSMExtraction(_Strict):
    """Everything the LLM extracts from one ICWSM dataset paper."""
    paper: PaperInfo
    collection: CollectionSpecs
    annotation: AnnotationAndTasks
    distribution: FAIRAndDistribution
    ethics: EthicsAndGovernance
    evidence: EvidenceAudit


# Split variants, used automatically if the full schema is ever rejected as too
# complex to compile. The two halves are merged back into ICWSMExtraction.
class ExtractionPartA(_Strict):
    paper: PaperInfo
    collection: CollectionSpecs
    annotation: AnnotationAndTasks


class ExtractionPartB(_Strict):
    distribution: FAIRAndDistribution
    ethics: EthicsAndGovernance
    evidence: EvidenceAudit


SECTION_MODELS = {
    "paper": PaperInfo,
    "collection": CollectionSpecs,
    "annotation": AnnotationAndTasks,
    "distribution": FAIRAndDistribution,
    "ethics": EthicsAndGovernance,
    "evidence": EvidenceAudit,
}


# ==========================================================================
# JSON-schema helpers
# ==========================================================================

def to_strict_json_schema(model: type[BaseModel]) -> Dict[str, Any]:
    """Pydantic model -> JSON schema that Claude structured outputs accepts.

    * inlines every $ref (avoids allOf+$ref, which is unsupported)
    * marks every property required and sets additionalProperties: false
    * strips titles/defaults and any numeric/string constraints
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = copy.deepcopy(defs[node["$ref"].split("/")[-1]])
                extra = {k: v for k, v in node.items() if k != "$ref"}
                target.update(extra)
                return resolve(target)
            if "allOf" in node and len(node["allOf"]) == 1:
                merged = resolve(node["allOf"][0])
                merged.update({k: v for k, v in node.items() if k != "allOf"})
                return resolve(merged)
            out = {}
            for k, v in node.items():
                if k in ("title", "default", "minimum", "maximum", "exclusiveMinimum",
                         "exclusiveMaximum", "minLength", "maxLength", "maxItems", "multipleOf"):
                    continue
                out[k] = resolve(v)
            if out.get("type") == "object" and "properties" in out:
                out["required"] = list(out["properties"].keys())
                out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(raw)


def schema_complexity(schema: Dict[str, Any]) -> Dict[str, int]:
    """Count the things Claude's structured-output limits care about."""
    stats = {"properties": 0, "optional": 0, "unions": 0, "enums": 0, "objects": 0}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                stats["objects"] += 1
                req = set(node.get("required", []))
                for name, sub in node["properties"].items():
                    stats["properties"] += 1
                    if name not in req:
                        stats["optional"] += 1
                    if "anyOf" in sub or isinstance(sub.get("type"), list):
                        stats["unions"] += 1
            if "enum" in node:
                stats["enums"] += 1
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return stats


def leaf_fields(model: type[BaseModel] = ICWSMExtraction) -> List[Dict[str, str]]:
    """Flat list of (section, field, type, description) for the data dictionary."""
    rows = []
    for section, sub in model.model_fields.items():
        sub_model = sub.annotation
        for fname, finfo in sub_model.model_fields.items():
            ann = finfo.annotation
            type_name = getattr(ann, "__name__", str(ann)).replace("typing.", "")
            if hasattr(ann, "__args__") and ann.__args__:
                inner = ann.__args__[0]
                type_name = f"List[{getattr(inner, '__name__', str(inner))}]"
            allowed = ""
            enum_cls = ann.__args__[0] if hasattr(ann, "__args__") and ann.__args__ else ann
            if isinstance(enum_cls, type) and issubclass(enum_cls, Enum):
                allowed = " | ".join(m.value for m in enum_cls)
            rows.append({
                "section": section,
                "field": fname,
                "type": type_name,
                "description": finfo.description or "",
                "allowed_values": allowed,
            })
    return rows


if __name__ == "__main__":
    import json
    s = to_strict_json_schema(ICWSMExtraction)
    print(json.dumps(schema_complexity(s), indent=2))
    print(f"{len(leaf_fields())} LLM-extracted attributes")
