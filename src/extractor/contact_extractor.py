# src/extractor/contact_extractor.py
import re
import spacy

nlp = spacy.load("en_core_web_sm")


def extract_contacts(text: str) -> dict:
    email = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    phone = re.findall(r"(\+?\d[\d\s\-\(\)]{7,}\d)", text)
    linkedin = re.findall(r"linkedin\.com/in/[\w\-]+", text, re.IGNORECASE)

    doc = nlp(text)
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "")

    return {
        "name": name,
        "email": email[0] if email else "",
        "phone": phone[0].strip() if phone else "",
        "linkedin": linkedin[0] if linkedin else "",
    }
