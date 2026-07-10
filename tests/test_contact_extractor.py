import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.extractor.contact_extractor import extract_contacts


class TestExtractContacts:

    CV_WITH_CONTACTS = """John Doe
    john.doe@email.com | +1-555-123-4567
    linkedin.com/in/johndoe
    """

    CV_WITHOUT_STRUCTURED = """
    Email: jane@example.com
    Phone: +8801712345678
    linkedin.com/in/janesmith
    """

    def test_email_extraction(self):
        result = extract_contacts(self.CV_WITH_CONTACTS, contacts={})
        assert result["email"] == "john.doe@email.com"

    def test_phone_extraction(self):
        result = extract_contacts(self.CV_WITH_CONTACTS, contacts={})
        assert result["phone"] in ("+1-555-123-4567", "555-123-4567")

    def test_linkedin_extraction(self):
        result = extract_contacts(self.CV_WITH_CONTACTS, contacts={})
        assert result["linkedin"] == "linkedin.com/in/johndoe"

    def test_email_from_text_fallback(self):
        result = extract_contacts(self.CV_WITHOUT_STRUCTURED, contacts={})
        assert result["email"] == "jane@example.com"

    def test_phone_from_text_fallback(self):
        result = extract_contacts(self.CV_WITHOUT_STRUCTURED, contacts={})
        assert result["phone"] != ""

    def test_no_email_returns_empty(self):
        result = extract_contacts("This CV has no email address.", contacts={})
        assert result["email"] == ""

    def test_no_phone_returns_empty(self):
        result = extract_contacts("No phone number here.", contacts={})
        assert result["phone"] == ""

    def test_empty_text_returns_empty(self):
        result = extract_contacts("", contacts={})
        assert result["name"] == ""
        assert result["email"] == ""
        assert result["phone"] == ""

    def test_name_from_personal_info_structured(self):
        contacts = {"personal_info": '{"name": "Rahim Ahmed"}'}
        result = extract_contacts("some cv text", contacts=contacts)
        assert result["name"] == "Rahim Ahmed"

    def test_name_not_unknown(self):
        contacts = {"personal_info": '{"name": "Unknown"}'}
        result = extract_contacts("some cv text", contacts=contacts)
        assert result["name"] != "Unknown"

    def test_name_not_provided(self):
        contacts = {"personal_info": '{"name": "not provided"}'}
        result = extract_contacts("some cv text", contacts=contacts)
        assert result["name"] not in ("not provided", "Not Provided")

    def test_multiple_emails_returns_first(self):
        text = "a@b.com and c@d.com"
        result = extract_contacts(text, contacts={})
        assert result["email"] == "a@b.com"

    def test_multiple_phones_returns_first(self):
        text = "123-456-7890 and 987-654-3210"
        result = extract_contacts(text, contacts={})
        assert result["phone"] == "123-456-7890"

    def test_name_not_classified_from_empty_contacts(self):
        contacts = {"personal_info": "{}"}
        result = extract_contacts("Python Java C++ Engineer", contacts=contacts)
        assert result["name"] == ""

    def test_contacts_is_none(self):
        result = extract_contacts("test@email.com", contacts=None)
        assert result["email"] == "test@email.com"

    def test_linkedin_not_found(self):
        result = extract_contacts("no linkedin here", contacts={})
        assert result["linkedin"] == ""
