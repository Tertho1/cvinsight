import re
import logging
from typing import Optional
import spacy
from src.extractor.utils import try_parse_structured

logger = logging.getLogger(__name__)

# ── Load spaCy model with fallback chain ──────────────────────────
_SPACY_MODEL = "en_core_web_sm"
for model in ("en_core_web_trf", "en_core_web_md", "en_core_web_sm"):
    try:
        nlp = spacy.load(model)
        _SPACY_MODEL = model
        logger.info(f"Loaded {model}")
        break
    except OSError:
        continue
else:
    raise OSError("No spaCy model found (tried en_core_web_trf, en_core_web_md, en_core_web_sm)")

# ── Known tech terms that NER models commonly mislabel as PERSON ─────
_TECH_TERMS = {
    # Programming languages & runtimes
    "java", "python", "javascript", "typescript", "kotlin", "swift",
    "ruby", "php", "perl", "scala", "groovy", "clojure", "haskell",
    "elixir", "rust", "go", "dart", "c#", "c++", "c", "cuda",
    "node.js", "nodejs", "deno", "bun", ".net", "dotnet",
    # Java ecosystem
    "spring", "spring boot", "spring cloud", "spring mvc", "spring security",
    "spring data", "spring framework", "hibernate", "jpa", "jdbc",
    "struts", "mybatis", "j2ee", "jee", "ejb", "servlet", "jsp", "jsf",
    "vaadin", "wicket", "play framework", "quarkus", "micronaut",
    # Python ecosystem
    "django", "flask", "fastapi", "tornado", "bottle", "pyramid",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "opencv", "pillow", "nltk", "spacy", "transformers", "langchain",
    "celery", "redis", "sqlalchemy", "pydantic",
    # JS/TS ecosystem
    "react", "react.js", "reactjs", "angular", "vue", "vue.js",
    "next.js", "nextjs", "nuxt", "svelte", "express", "express.js",
    "node", "jquery", "redux", "webpack", "vite", "babel",
    "gatsby", "remix", "solid.js", "solidjs", "alpine.js",
    # Frameworks & platforms
    "docker", "kubernetes", "k8s", "jenkins", "ansible", "terraform",
    "puppet", "chef", "vagrant", "helm", "istio", "linkerd",
    "apache", "nginx", "tomcat", "jboss", "wildfly", "jetty",
    "kafka", "spark", "hadoop", "flink", "storm", "airflow",
    "rabbitmq", "activemq", "pulsar",
    # Cloud providers
    "aws", "azure", "gcp", "google cloud", "amazon web", "amazon web services",
    "amazon s3", "ec2", "lambda", "cloudformation", "cloudfront",
    "route 53", "dynamodb", "sns", "sqs", "ecs", "eks", "fargate",
    # Databases
    "mysql", "postgresql", "postgres", "mongodb", "mongo db",
    "redis", "cassandra", "sqlite", "mariadb", "oracle",
    "sql server", "elasticsearch", "couchbase", "couchdb", "neo4j",
    "influxdb", "clickhouse", "snowflake", "bigquery", "redshift",
    "firebase", "supabase",
    # Data & ML
    "machine learning", "deep learning", "computer vision", "natural language",
    "data science", "data scientist", "data engineering", "data analysis",
    "business intelligence", "tableau", "power bi", "looker", "etl", "elt",
    # Resume skill section artifacts
    "problem solving", "critical thinking", "time management",
    "project management", "communication skills", "leadership skills",
    "teamwork skills", "attention detail", "detail oriented",
    "decision making", "conflict resolution", "public speaking",
    "presentation skills", "written communication", "verbal communication",
    "interpersonal skills", "organizational skills", "analytical skills",
    "technical skills", "soft skills", "domain knowledge",
    # Tools
    "git", "github", "gitlab", "bitbucket", "jira", "confluence",
    "grafana", "prometheus", "datadog", "new relic", "splunk",
    "maven", "gradle", "ant", "npm", "yarn", "pnpm",
    "postman", "swagger", "openapi", "insomnia",
    "sentry", "logstash", "kibana", "filebeat",
    # Mobile
    "flutter", "react native", "xamarin", "ionic", "cordova",
    "android studio", "xcode", "swift ui", "uikit", "jetpack compose",
    # IDEs & editors
    "visual studio", "vs code", "intellij idea", "eclipse", "pycharm",
    "webstorm", "vim", "neovim",
    # Testing
    "junit", "mockito", "pytest", "jest", "mocha", "chai",
    "selenium", "cypress", "playwright", "cucumber", "testng",
    # Operating systems
    "linux", "ubuntu", "centos", "red hat", "debian", "alpine",
    "windows", "macos", "unix",
    # Common resume section artifacts
    "page", "hello", "about", "contact", "resume", "curriculum",
    "vitae", "profile", "objective", "portfolio",
    "references", "summary", "qualifications", "highlights",
    "frontend", "backend", "full stack", "fullstack", "devops",
    "software", "hardware", "agile", "scrum", "kanban",
    "cloud", "api", "rest", "restful api", "rest api", "graphql",
    "microservice", "blockchain", "ai", "ml", "nlp", "iot",
    "responsive", "scalable", "distributed",
    # New additions for false positives reported by user
    "routing protocols", "login registration", "a secret key", "secret key",
    "siemens s7", "siemens s7-300", "s7-300", "s7 300",
    "arya kanya",  # likely an organization/company name, not a person
    "data scientist",  # job title - reinforce
    "machine learning engineer", "data engineer", "software engineer",
    "problem solving", "restful api", "rest api",
    # Common project/description phrases
    "login", "registration", "authentication", "authorization",
    "inventory", "management system", "tracking system",
    "ecommerce", "e-commerce", "online shopping",
    "data mining", "web scraping", "data visualization",
    "sentiment analysis", "image processing", "signal processing",
    "natural language processing",
}

# Single-word tech items for per-word checks
_TECH_SINGLE_WORDS = {
    "java", "python", "spring", "docker", "kubernetes", "jenkins",
    "terraform", "ansible", "apache", "nginx", "kafka", "spark",
    "redis", "mongo", "mysql", "postgres", "oracle", "swift",
    "kotlin", "dart", "rust", "go", "ruby", "perl", "php",
    "react", "angular", "vue", "svelte", "django", "flask",
    "fastapi", "express", "node", "flutter", "git", "linux",
    "ubuntu", "centos", "agile", "scrum", "devops", "cloud",
    "api", "graphql", "rest", "soap", "microservice",
    "secret", "login", "routing", "protocol", "problem",
    "solving", "learning", "machine",  # "machine learning" is caught above, but individual words too
    "registration", "siemens", "data",
}

# Common English first names (to positively identify real names)
_COMMON_FIRST_NAMES = {
    "james", "john", "robert", "michael", "william", "david", "richard",
    "joseph", "thomas", "christopher", "charles", "daniel", "matthew",
    "anthony", "mark", "donald", "steven", "paul", "andrew", "joshua",
    "kenneth", "kevin", "brian", "george", "timothy", "ronald", "edward",
    "jason", "jeffrey", "ryan", "jacob", "gary", "nicholas", "eric",
    "jonathan", "stephen", "larry", "justin", "scott", "brandon",
    "benjamin", "samuel", "raymond", "gregory", "frank", "alexander",
    "patrick", "jack", "dennis", "jerry", "tyler", "aaron", "jose",
    "nathan", "henry", "douglas", "peter", "adam", "zachary",
    "mary", "patricia", "jennifer", "linda", "barbara", "elizabeth",
    "susan", "jessica", "sarah", "karen", "lisa", "nancy",
    "betty", "margaret", "sandra", "ashley", "dorothy", "kimberly",
    "emily", "donna", "michelle", "carol", "amanda", "melissa",
    "deborah", "stephanie", "rebecca", "sharon", "laura", "cynthia",
    "kathleen", "amy", "angela", "shirley", "anna", "brenda",
    "paula", "virginia", "jana", "nadia", "samantha", "sophia",
    "isabella", "olivia", "emma", "charlotte", "amelia", "harper",
    "evelyn", "abigail", "ella", "avery", "scarlett", "grace",
    "chloe", "victoria", "riley", "aria", "lily", "aurora",
    "zoe", "nora", "camila", "penelope", "layla", "luna",
    "muhammad", "noah", "liam", "oliver", "elijah",
    "william", "henry", "lucas", "benjamin", "theodore", "sebastian",
    "ezra", "levi", "mateo", "jackson", "leo", "owen",
    "aiden", "samuel", "jaxon", "logan", "josiah", "max",
    "luka", "mason", "ethan", "alex", "hugo", "nathan",
    "mohamed", "ibrahim", "abdul", "ali", "ahmed", "omar",
    "hassan", "hussain", "amir", "farhan", "zayn", "ayaan",
    "aditya", "yogesh", "arya",  # add common Indian names
    "fahed", "sherri", "artem",  # add names from dataset
}

# Word endings that suggest a technical/company term rather than a name
_TECH_SUFFIXES = {
    "ware", "base", "view", "logic", "query", "stream",
    "script", "scale", "score", "sense", "scope", "signal",
    "track", "trace", "trade", "train", "trait", "value",
    "cache", "cloud", "craft", "crud", "cycle", "debug",
    "drive", "event", "field", "flash", "frame", "graph",
    "guide", "layer", "light", "macro", "media", "metal",
    "micro", "model", "parse", "pixel", "print", "proof",
    "proto", "proxy", "pulse", "relay", "sight", "slash",
    "smart", "sniff", "space", "speed", "swarm", "sweet",
    "table", "touch", "tower", "tuner", "ultra", "union",
    "usage", "vault", "verse", "visor", "vista", "vital",
    "voice", "water", "weave", "whale", "wheel", "width",
}


def _is_tech_term(text: str) -> bool:
    lower = text.lower().strip()
    if lower in _TECH_TERMS:
        return True
    words = lower.split()
    if len(words) >= 2:
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                phrase = " ".join(words[i:j])
                if phrase in _TECH_TERMS:
                    return True
    return False


def _contains_tech_word(text: str) -> bool:
    for w in text.lower().split():
        w_clean = w.strip(".,;:!?-'\"")
        if w_clean in _TECH_SINGLE_WORDS:
            return True
    return False


def _has_tech_suffix(word: str) -> bool:
    w = word.lower().strip(".,;:!?-'\"")
    return any(w.endswith(suf) for suf in _TECH_SUFFIXES)


def _looks_like_real_name(text: str) -> bool:
    words = text.split()
    if not words:
        return False

    lower_words = [w.lower().strip(".,;:!?-'\"") for w in words]

    # If any word matches a known first name, that's a strong positive signal
    if any(w in _COMMON_FIRST_NAMES for w in lower_words):
        return True

    # Each word should start with uppercase (true name case)
    if not all(w[0].isupper() for w in words if w and w[0].isalpha()):
        return False

    # No word should contain digits
    if any(re.search(r"\d", w) for w in words):
        return False

    # No word should have a tech-suffix ending
    if any(_has_tech_suffix(w) for w in words):
        return False

    # No single-letter words (middle initials like "J." with a period are OK)
    for w in words:
        w_clean = w.strip(".")
        if len(w_clean) == 1 and w_clean.isalpha():
            return False

    # No all-lowercase words (real names are capitalized)
    if all(w[0].islower() for w in words if w and w[0].isalpha()):
        return False

    # Reject two-word phrases where BOTH words are common dictionary nouns
    # that aren't typically used as surnames (e.g. "Spring Boot", "Problem Solving")
    _COMMON_DICT_WORDS = {
        "spring", "boot", "problem", "solving", "login", "registration",
        "restful", "secret", "key", "data", "routing", "protocol",
        "machine", "learning", "cloud", "foundry", "project", "management",
        "critical", "thinking", "teamwork", "communication", "leadership",
        "time", "analytical", "technical", "software", "hardware",
        "frontend", "backend", "fullstack", "signal", "digital",
        "network", "system", "science", "engineering", "service",
        "solution", "development", "application", "design", "quality",
        "control", "support", "planning", "strategy", "operation",
        "api", "scientist", "engineer", "developer", "manager",
        "analyst", "architect", "director", "officer", "specialist",
    }
    if len(words) == 2:
        w1, w2 = lower_words[0], lower_words[1] if len(lower_words) > 1 else ""
        if w1 in _COMMON_DICT_WORDS and w2 in _COMMON_DICT_WORDS:
            return False

    return True


# ── Helpers for line-based extraction ─────────────────────────────

_NAME_SUFFIX_BLACKLIST = {
    "ltd", "limited", "inc", "corp", "corporation", "pvt", "private",
    "technologies", "consulting", "services", "solutions", "group",
    "systems", "industries", "labs", "llc",
}

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.\w{2,}")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?"
    r"(?:"
    r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"  # US/international 3-3-4 format
    r"|\d{5}[-.\s]?\d{5}"                    # Indian 5-5 format
    r")"
)
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w-]+", re.IGNORECASE)
_SALUTATIONS = {"mr", "ms", "mrs", "dr", "prof"}
_JSON_FRAGMENT_RE = re.compile(r'"[a-z_]+":\s*"')
_VERSION_RE = re.compile(r"\d+\.\d+")

_JOB_TITLE_INDICATORS = {
    "engineer", "developer", "manager", "analyst", "scientist",
    "architect", "designer", "lead", "head", "director", "officer",
    "specialist", "consultant", "coordinator", "administrator",
    "intern", "trainee", "associate", "executive",
}

_LOCATION_WORDS = {
    "pradesh", "bangalore", "mumbai", "delhi", "pune", "kolkata",
    "chennai", "hyderabad", "ahmedabad", "india", "nagpur", "jabalpur",
    "maharashtra", "gujarat", "karnataka", "rajasthan", "punjab",
    "haryana", "kerala", "bihar", "assam", "odisha",
    "bangladesh", "dhaka", "chittagong", "chattogram", "rajshahi",
    "khulna", "sylhet", "barisal", "rangpur", "mymensingh",
    "usa", "america", "california", "texas", "florida",
    "washington", "boston", "austin", "seattle", "chicago",
}


def _extract_name_from_contacts(contacts: dict) -> str:
    raw = contacts.get("personal_info", "")
    if raw and raw not in ("{}", ""):
        parsed = try_parse_structured(raw)
        if isinstance(parsed, dict):
            name = parsed.get("name", "")
            if name and name.lower() not in ("unknown", "not provided", ""):
                return name
    return ""


def _is_company_name(text: str) -> bool:
    words = text.lower().split()
    return bool(words and words[-1] in _NAME_SUFFIX_BLACKLIST)


def _is_job_title(text: str) -> bool:
    words = text.lower().split()
    return bool(words and any(w in _JOB_TITLE_INDICATORS for w in words))


def _extract_name_from_text(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # ── Path 1: First 10 lines ──────────────────────────────────
    name_candidates = []
    for line in lines[:10]:
        if not line or line.startswith("{") or line.startswith("["):
            continue
        if '"' in line or _JSON_FRAGMENT_RE.search(line):
            continue
        if "|" in line or "@" in line or "www." in line.lower():
            continue
        if "company" in line.lower() or "title" in line.lower():
            continue
        if _VERSION_RE.search(line):
            continue
        if re.search(r"\d", line):
            continue

        words = line.split()
        if len(words) not in (2, 3):
            continue
        if _is_tech_term(line):
            continue
        if _contains_tech_word(line):
            continue
        if _is_company_name(line) or _is_job_title(line):
            continue
        if any(w.lower() in ("college", "university", "school", "institute") for w in words):
            continue
        if any(w.lower() in _LOCATION_WORDS for w in words):
            continue
        if not _looks_like_real_name(line):
            continue

        score = sum(1 for ch in line if ch.isupper())
        if score >= 2:
            name_candidates.append(line)

    for line in lines[:5]:
        first_word = line.split()[0].lower() if line.split() else ""
        if first_word in _SALUTATIONS:
            after_title = line.split(None, 1)[-1] if len(line.split()) > 1 else line
            words = after_title.split()
            if len(words) in (2, 3) and not _is_tech_term(after_title):
                if after_title not in name_candidates:
                    name_candidates.insert(0, after_title)
            break

    if name_candidates:
        return name_candidates[0]

    # ── Path 2: spaCy NER on first 400 chars only ──────────────
    header = text[:400]
    doc = nlp(header)
    seen = set()
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        text_lower = ent.text.lower()
        words_lower = text_lower.split()
        # Accept 1-3 word PERSON entities; skip 4+ word (very unlikely to be a real name)
        if len(words_lower) not in (1, 2, 3):
            continue
        if _is_tech_term(ent.text):
            continue
        if _contains_tech_word(ent.text):
            continue
        if _is_company_name(ent.text) or _is_job_title(ent.text):
            continue
        if any(w in ("college", "university", "school", "institute") for w in words_lower):
            continue
        if any(w in _LOCATION_WORDS for w in words_lower):
            continue
        if _VERSION_RE.search(ent.text):
            continue
        if '"' in ent.text:
            continue
        if re.search(r"\d", ent.text):
            continue
        if not _looks_like_real_name(ent.text):
            continue
        key = text_lower
        if key not in seen:
            seen.add(key)
            name_candidates.append(ent.text)

    return name_candidates[0] if name_candidates else ""


def extract_contacts(text: str, contacts: Optional[dict] = None) -> dict:
    if contacts is None:
        contacts = {}

    name = _extract_name_from_contacts(contacts)
    if not name:
        name = _extract_name_from_text(text)

    email_matches = _EMAIL_RE.findall(text)
    email = email_matches[0] if email_matches else ""

    phone_matches = _PHONE_RE.findall(text)
    phone = phone_matches[0] if phone_matches else ""

    linkedin_matches = _LINKEDIN_RE.findall(text)
    linkedin = linkedin_matches[0] if linkedin_matches else ""

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
    }
