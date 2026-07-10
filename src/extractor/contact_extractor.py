import re
import spacy

nlp = spacy.load("en_core_web_sm")

_NAME_STOPWORDS = {
    "c++", "c", "python", "java", "javascript", "engineer", "developer",
    "manager", "analyst", "scientist", "student", "resume", "curriculum",
    "vitae", "cv", "objective", "summary", "education", "experience",
    "skills", "projects", "certifications", "languages", "achievements",
    "leadership", "references", "contact", "profile", "email", "phone",
    "address", "linkedin", "github", "technologies",
    "limited", "ltd", "inc", "corp", "company", "pvt", "private",
    "fresher", "unknown", "not provided", "college", "university",
    "core java", "java 1.6", "java 1.8", "cloud foundry", "prince2",
    "practitioner", "junior", "senior", "lead", "gateway solutions",
    "requirement", "m3", "solutions", "cloud foundry", "madhya pradesh",
    "andhra pradesh", "tamil nadu", "uttar pradesh", "west bengal",
}

_NAME_SUFFIX_BLACKLIST = {
    "ltd", "limited", "inc", "corp", "corporation", "pvt", "private",
    "technologies", "consulting", "services", "solutions", "group",
    "systems", "industries", "labs", "llc",
}

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.\w{2,}")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
)
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w-]+", re.IGNORECASE)

_SALUTATIONS = {"mr", "ms", "mrs", "dr", "prof"}


def _extract_name_from_contacts(contacts: dict) -> str:
    raw = contacts.get("personal_info", "")
    if raw and raw not in ("{}", ""):
        from src.extractor.utils import try_parse_structured
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
    title_indicators = {"engineer", "developer", "manager", "analyst",
                        "scientist", "architect", "designer", "lead",
                        "head", "director", "officer", "specialist",
                        "consultant", "coordinator", "administrator",
                        "intern", "trainee", "associate", "executive"}
    words = text.lower().split()
    return bool(words and any(w in title_indicators for w in words))


_JSON_FRAGMENT_RE = re.compile(r'"[a-z_]+":\s*"')


_VERSION_RE = re.compile(r"\d+\.\d+")

_LOCATION_WORDS = {
    "pradesh", "bangalore", "mumbai", "delhi", "pune", "kolkata",
    "chennai", "hyderabad", "ahmedabad", "india", "nagpur", "jabalpur",
    "maharashtra", "gujarat", "karnataka", "rajasthan", "punjab",
    "haryana", "kerala", "bihar", "assam", "odisha",
}


def _extract_name_from_text(text: str) -> str:
    name_candidates = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    for line in lines[:10]:
        line_lower = line.lower()
        if not line or line.startswith("{") or line.startswith("["):
            continue
        if '"' in line or _JSON_FRAGMENT_RE.search(line):
            continue
        if "|" in line or "@" in line or "www." in line.lower():
            continue
        if "company" in line_lower or "title" in line_lower:
            continue
        if line_lower in _NAME_STOPWORDS:
            continue
        if _VERSION_RE.search(line):
            continue
        words = line.split()
        if len(words) not in (2, 3):
            continue
        if _is_company_name(line) or _is_job_title(line):
            continue
        if any(w.lower() in ("college", "university", "school", "institute") for w in words):
            continue
        if any(w.lower() in _LOCATION_WORDS for w in words):
            continue

        score = sum(1 for ch in line if ch.isupper())
        if score >= 2:
            last_word = words[-1].lower().strip(".,")
            if last_word not in _NAME_STOPWORDS:
                name_candidates.append(line)

    for line in lines[:5]:
        first_word = line.split()[0].lower() if line.split() else ""
        if first_word in _SALUTATIONS:
            after_title = line.split(None, 1)[-1] if len(line.split()) > 1 else line
            if after_title not in name_candidates and not _is_company_name(after_title):
                name_candidates.insert(0, after_title)
            break

    if name_candidates:
        return name_candidates[0]

    doc = nlp(text[:2000])
    seen = set()
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        text_lower = ent.text.lower()
        words_lower = text_lower.split()
        if text_lower in _NAME_STOPWORDS:
            continue
        if any(w in _NAME_STOPWORDS for w in words_lower):
            continue
        if len(words_lower) not in (2, 3):
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
        key = text_lower
        if key not in seen:
            seen.add(key)
            name_candidates.append(ent.text)

    return name_candidates[0] if name_candidates else ""


def extract_contacts(text: str, contacts: dict = {}) -> dict:
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
