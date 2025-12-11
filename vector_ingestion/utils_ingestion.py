"""
Utility functions for candidate ingestion processing.
"""
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime
import tiktoken
import re
import logging

from .models import CandidateRawProfile

if TYPE_CHECKING:
    from .services.resume_summarizer import ResumeSummarizerService

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .services.resume_summarizer import ResumeSummarizerService

# Token counter for estimating token length (using cl100k_base which is close to Voyage AI)
_token_encoder = None


def get_token_count(text: str) -> int:
    """Estimate token count for a text string using tiktoken."""
    global _token_encoder
    if _token_encoder is None:
        # Use cl100k_base encoding (similar to what Voyage AI uses)
        _token_encoder = tiktoken.get_encoding("cl100k_base")
    return len(_token_encoder.encode(text))


def _safe_get(data: Dict, *keys, default=None):
    """Safely get nested dictionary values."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return default
        if current is None:
            return default
    return current if current is not None else default


def _format_date(date_obj: Dict) -> str:
    """Format date object to readable string."""
    if not date_obj or not isinstance(date_obj, dict):
        return ""
    date_str = date_obj.get("Date", "")
    if date_str:
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m")
        except:
            return date_str[:7] if len(date_str) >= 7 else ""
    return ""


def _format_year_range(start_date: str, end_date: str) -> str:
    """Format dates as year range (e.g., '2020–2025' or '2023–2024')."""
    if not start_date:
        return ""
    
    start_year = start_date[:4] if len(start_date) >= 4 else ""
    end_year = end_date[:4] if end_date and len(end_date) >= 4 else ""
    
    if start_year and end_year:
        return f"{start_year}–{end_year}"
    elif start_year:
        return f"{start_year}–Present"
    return ""


def _extract_section_from_plaintext(plaintext: str, section_patterns: List[str]) -> Optional[str]:
    """Extract a section from plaintext using multiple regex patterns."""
    if not plaintext:
        return None
    
    for pattern in section_patterns:
        match = re.search(pattern, plaintext, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Clean up the content
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            return content
    return None


def _extract_bullet_points(text: str, max_bullets: int = 15) -> List[str]:
    """Extract bullet points from text (●, •, -, *, etc.)."""
    if not text:
        return []
    
    # Pattern to match bullet points - handle various formats
    # Matches: ● text, • text, - text, * text, or lines starting with spaces/tabs followed by bullet
    bullet_pattern = r'(?:^|\n)[\s\t]*[●•\-\*]\s*([^\n●•]+?)(?=\n[\s\t]*[●•\-\*]|\n\n|$)'
    bullets = re.findall(bullet_pattern, text, re.MULTILINE)
    
    # Also try to match numbered lists or lines that look like responsibilities
    # Pattern for lines that start with common responsibility keywords
    responsibility_pattern = r'(?:^|\n)[\s\t]*(?:Configured|Managed|Led|Designed|Coordinated|Executed|Collaborated|Supported|Provided|Handled|Prepared|Conducted|Performed|Developed|Implemented|Oversaw|Facilitated|Participated|Delivered|Maintained|Addressed|Engaged|Orchestrated|Generated|Initiated|Played|Actively|Demonstrated|Trained|Customized|Offered|Reviewed|Part of)[^\n]+'
    responsibility_matches = re.findall(responsibility_pattern, text, re.MULTILINE | re.IGNORECASE)
    
    # Combine both
    all_bullets = bullets + [m.strip() for m in responsibility_matches]
    
    # Clean and limit bullets
    cleaned_bullets = []
    seen = set()
    for bullet in all_bullets[:max_bullets * 2]:  # Get more, then filter
        bullet = bullet.strip()
        # Remove leading/trailing punctuation
        bullet = re.sub(r'^[•●\-\*\s]+', '', bullet)
        bullet = bullet.strip()
        
        if len(bullet) > 10 and bullet.lower() not in seen:  # Filter out very short bullets
            seen.add(bullet.lower())
            # Capitalize first letter
            if bullet:
                bullet = bullet[0].upper() + bullet[1:] if len(bullet) > 1 else bullet.upper()
                cleaned_bullets.append(bullet)
        
        if len(cleaned_bullets) >= max_bullets:
                break
    
    return cleaned_bullets


def _extract_modules_and_tools(text: str) -> tuple[List[str], List[str]]:
    """Extract module expertise and tools/methods from text."""
    modules = []
    tools = []
    
    if not text:
        return modules, tools
    
    # Common module patterns (e.g., GL, AP, AR, FA, CM, etc.)
    module_keywords = [
        'GL', 'AP', 'AR', 'FA', 'CM', 'AGIS', 'BPM', 'Expense', 'Expense Management',
        'Accounts Payable', 'Accounts Receivable', 'General Ledger', 'Fixed Assets',
        'Cash Management', 'Oracle Fusion Financials', 'Oracle Fusion Cloud', 'Oracle Fusion ERP'
    ]
    
    # Extract modules - look for these keywords
    text_upper = text.upper()
    for keyword in module_keywords:
        if keyword.upper() in text_upper:
            # Find the actual occurrence to preserve case
            pattern = re.escape(keyword)
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                matched_text = match.group(0)
                if matched_text not in [m for m in modules]:
                    modules.append(matched_text)
    
    # Extract tools and methods
    tools_keywords = [
        'FBDI', 'CRP', 'SIT', 'UAT', 'AIM', 'OUM', 'Azure DevOps', 'Jira', 'Confluence',
        'SharePoint', 'MS Project', 'Oracle Fusion Cloud ERP', 'requirement analysis',
        'data migration', 'functional configuration', 'testing', 'integration',
        'GAP analysis', 'solution design', 'production support'
    ]
    
    for keyword in tools_keywords:
        if keyword.upper() in text_upper:
            pattern = re.escape(keyword)
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                matched_text = match.group(0)
                if matched_text.lower() not in [t.lower() for t in tools]:
                    tools.append(matched_text)
    
    return modules[:20], tools[:15]


def _extract_projects_from_plaintext(plaintext: str) -> List[str]:
    """Extract project/company names from plaintext."""
    projects = []
    
    if not plaintext:
        return projects
    
    # Look for company names followed by dates or job titles
    # Pattern: Company Name (optional dates) – Job Title
    project_pattern = r'([A-Z][A-Za-z\s&]+?)\s*(?:\([^)]+\))?\s*(?:–|–|:)?\s*([A-Z][^•\n]+?)(?:\n|•|$)'
    
    matches = re.finditer(project_pattern, plaintext, re.MULTILINE)
    seen = set()
    
    for match in matches:
        company = match.group(1).strip()
        role = match.group(2).strip() if match.group(2) else ""
        
        if company and len(company) > 2 and company.upper() not in seen:
            seen.add(company.upper())
            if role:
                projects.append(f"{company} – {role}")
            else:
                projects.append(company)
        
        if len(projects) >= 10:
            break
    
    return projects


def _parse_plaintext(plaintext: str) -> Dict[str, Any]:
    """Parse PlainText field and extract structured information."""
    if not plaintext:
        return {}
    
    result = {
        'summary': None,
        'modules': [],
        'responsibilities': [],
        'tools': [],
        'projects': [],
        'education': None
    }
    
    # Extract Professional Summary - look for "Professional Summary:" section
    summary_patterns = [
        r'Professional Summary[:\s]*\n(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:|\n●|EDUCATION|PERSONAL|$)',
        r'Summary[:\s]*\n(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:|\n●|EDUCATION|PERSONAL|$)',
        r'Career Objective[:\s]*\n(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:|\n●|EDUCATION|PERSONAL|$)',
    ]
    
    summary = _extract_section_from_plaintext(plaintext, summary_patterns)
    if summary:
        # Clean and create concise summary
        summary = re.sub(r'\s+', ' ', summary)
        # Take first 2-3 sentences
        sentences = re.split(r'[.!?]+', summary)
        summary = '. '.join([s.strip() for s in sentences[:3] if s.strip()])
        if summary and not summary.endswith('.'):
            summary += '.'
        if len(summary) > 400:
            summary = summary[:400].rsplit('.', 1)[0] + '.'
        result['summary'] = summary
    
    # Extract Responsibilities - look for bullet points (●, •, -)
    # Get the section between Professional Summary and EDUCATION
    body_section = re.search(
        r'(?:Professional Summary|Summary|Career Objective)[:\s]*\n.*?\n(.*?)(?=EDUCATION|PERSONAL INFORMATION|PERSONAL|$)',
        plaintext,
        re.DOTALL | re.IGNORECASE
    )
    
    if body_section:
        body_text = body_section.group(1)
        # Extract all bullet points
        bullets = _extract_bullet_points(body_text, max_bullets=20)
        result['responsibilities'] = bullets
    
    # Extract modules and tools from entire text
    modules, tools = _extract_modules_and_tools(plaintext)
    result['modules'] = modules
    result['tools'] = tools
    
    # Extract projects/companies - look for company names with dates
    # Pattern: Company Name (optional dates) – Role or Company Name followed by dates
    project_lines = []
    
    # Look for patterns like "Company Name (2020–2025) – Role" or "Company Name\n2020–2025"
    project_pattern1 = r'([A-Z][A-Za-z\s&,\.]+?)\s*(?:\(([0-9]{4}[–-][0-9]{4}|[0-9]{4}[–-]Present)\))?\s*(?:–|–|:)?\s*([A-Z][^•\n●]+?)(?=\n|•|●|$)'
    
    matches = re.finditer(project_pattern1, plaintext, re.MULTILINE)
    seen = set()
    
    for match in matches:
        company = match.group(1).strip()
        dates = match.group(2) if match.group(2) else ""
        role = match.group(3).strip() if match.group(3) else ""
        
        # Filter out common false positives
        if company and len(company) > 2 and company.upper() not in ['THE', 'AND', 'FOR', 'WITH']:
            if company.upper() not in seen:
                seen.add(company.upper())
                project_line = company
                if dates:
                    project_line += f" ({dates})"
                if role and len(role) < 100:
                    project_line += f" – {role}"
                project_lines.append(project_line)
    
    result['projects'] = project_lines[:10]
    
    # Extract Education
    education_patterns = [
        r'EDUCATION[:\s]*\n(.*?)(?=\n\n[A-Z]|PERSONAL|$)',
        r'Education[:\s]*\n(.*?)(?=\n\n[A-Z]|PERSONAL|$)',
    ]
    
    education_text = _extract_section_from_plaintext(plaintext, education_patterns)
    if education_text:
        # Clean up education text
        education_text = re.sub(r'\s+', ' ', education_text.strip())
        # Extract degree, school, and dates
        # Pattern: Degree Name\nSchool Name\nDates or Degree Name (Majors) Dates
        degree_match = re.search(r'([A-Z][^0-9\n]+?)\s*(?:\(([^)]+)\))?\s*([0-9]{4}[–-][0-9]{4}|[0-9]{4})', education_text)
        if degree_match:
            degree = degree_match.group(1).strip()
            majors = degree_match.group(2) if degree_match.group(2) else ""
            dates = degree_match.group(3) if degree_match.group(3) else ""
            
            edu_parts = [degree]
            if majors:
                edu_parts.append(f"({majors})")
            if dates:
                edu_parts.append(f"({dates})")
            
            result['education'] = " ".join(edu_parts)
        else:
            # Fallback: just take first line or first 100 chars
            lines = [l.strip() for l in education_text.split('\n') if l.strip()]
            if lines:
                result['education'] = lines[0][:100]
    
    return result


def _remove_pii_from_text(text: str, reserved_data: Optional[Dict[str, Any]] = None, contact_info: Optional[Dict[str, Any]] = None, candidate: Optional[Any] = None) -> str:
    """
    Remove PII from any text string using ReservedData, ContactInformation, or candidate object (new format).
    
    Args:
        text: Any text string to clean
        reserved_data: ReservedData dict containing PII to remove (legacy format)
        contact_info: ContactInformation dict containing PII to remove (legacy format)
        candidate: CandidateRawProfile object (new format) - extracts contact info from candidate fields
        
    Returns:
        Text with PII removed
    """
    if not text:
        return ""
    
    # Extract contact info from new format candidate object if provided
    if candidate and hasattr(candidate, 'candidate_id'):
        # Build contact info dict from new format fields
        new_contact_info = {}
        
        # Names
        if candidate.first_name:
            new_contact_info['first_name'] = candidate.first_name
        if candidate.last_name:
            new_contact_info['last_name'] = candidate.last_name
        if candidate.first_name and candidate.last_name:
            new_contact_info['full_name'] = f"{candidate.first_name} {candidate.last_name}"
        
        # Emails
        emails = []
        if candidate.email:
            emails.append(candidate.email)
        if candidate.alternateemail:
            emails.append(candidate.alternateemail)
        if emails:
            new_contact_info['emails'] = emails
        
        # Phones
        phones = []
        for phone_field in ['workphone', 'homephone', 'cellphone', 'phone1', 'phone2', 'phone3', 'phone4']:
            phone_value = getattr(candidate, phone_field, None)
            if phone_value:
                phones.append(phone_value)
        if phones:
            new_contact_info['phones'] = phones
        
        # Use new format contact info
        if new_contact_info:
            # Remove names
            if new_contact_info.get('full_name'):
                name = new_contact_info['full_name']
                text = text.replace(name, "")
                text = re.sub(re.escape(name), "", text, flags=re.IGNORECASE)
            if new_contact_info.get('first_name'):
                name = new_contact_info['first_name']
                text = text.replace(name, "")
                text = re.sub(re.escape(name), "", text, flags=re.IGNORECASE)
            if new_contact_info.get('last_name'):
                name = new_contact_info['last_name']
                text = text.replace(name, "")
                text = re.sub(re.escape(name), "", text, flags=re.IGNORECASE)
            
            # Remove emails
            for email in new_contact_info.get('emails', []):
                if email:
                    text = text.replace(email, "")
                    text = re.sub(re.escape(email), "", text, flags=re.IGNORECASE)
            
            # Remove phones
            for phone in new_contact_info.get('phones', []):
                if phone:
                    # Remove phone in various formats
                    phone_clean = phone.replace("-", "").replace("(", "").replace(")", "").replace(" ", "").replace(".", "")
                    text = text.replace(phone, "")
                    text = text.replace(phone_clean, "")
                    # Also try formatted versions
                    if len(phone_clean) == 10:
                        formatted = f"({phone_clean[:3]}) {phone_clean[3:6]}-{phone_clean[6:]}"
                        text = text.replace(formatted, "")
                    text = re.sub(re.escape(phone), "", text, flags=re.IGNORECASE)
    
    # Remove PII from ContactInformation if available (legacy format)
    if contact_info:
        # Remove names
        candidate_name = contact_info.get("CandidateName")
        if candidate_name:
            if isinstance(candidate_name, dict):
                # Try all name fields
                name_fields = ["FormattedName", "GivenName", "MiddleName", "FamilyName", "Prefix", "Suffix"]
                for field in name_fields:
                    name_value = candidate_name.get(field)
                    if name_value:
                        text = text.replace(name_value, "")
                        name_pattern = re.escape(name_value)
                        text = re.sub(name_pattern, "", text, flags=re.IGNORECASE)
            elif isinstance(candidate_name, str):
                text = text.replace(candidate_name, "")
                name_pattern = re.escape(candidate_name)
                text = re.sub(name_pattern, "", text, flags=re.IGNORECASE)
        
        # Remove telephones
        telephones = contact_info.get("Telephones")
        if telephones and isinstance(telephones, list):
            for tel in telephones:
                if isinstance(tel, dict):
                    # Try all phone fields
                    phone_fields = ["Raw", "Normalized", "SubscriberNumber"]
                    for field in phone_fields:
                        phone_value = tel.get(field)
                        if phone_value:
                            text = text.replace(phone_value, "")
                            phone_pattern = re.escape(phone_value)
                            text = re.sub(phone_pattern, "", text, flags=re.IGNORECASE)
                elif isinstance(tel, str):
                    text = text.replace(tel, "")
                    phone_pattern = re.escape(tel)
                    text = re.sub(phone_pattern, "", text, flags=re.IGNORECASE)
        
        # Remove email addresses
        email_addresses = contact_info.get("EmailAddresses")
        if email_addresses and isinstance(email_addresses, list):
            for email in email_addresses:
                if email:
                    text = text.replace(email, "")
                    email_pattern = re.escape(email)
                    text = re.sub(email_pattern, "", text, flags=re.IGNORECASE)
        
        # Remove web addresses (LinkedIn, etc.)
        web_addresses = contact_info.get("WebAddresses")
        if web_addresses and isinstance(web_addresses, list):
            for web_addr in web_addresses:
                if isinstance(web_addr, dict):
                    address = web_addr.get("Address")
                    if address:
                        text = text.replace(address, "")
                        url_pattern = re.escape(address)
                        text = re.sub(url_pattern, "", text, flags=re.IGNORECASE)
                elif isinstance(web_addr, str):
                    text = text.replace(web_addr, "")
                    url_pattern = re.escape(web_addr)
                    text = re.sub(url_pattern, "", text, flags=re.IGNORECASE)
    
    # Remove PII from ReservedData if available
    if reserved_data:
        # Remove phone numbers
        if "Phones" in reserved_data and isinstance(reserved_data["Phones"], list):
            for phone in reserved_data["Phones"]:
                if phone:
                    # Remove phone in various formats
                    phone_clean = phone.replace("Mob:", "").replace("Phone:", "").strip()
                    text = text.replace(phone, "")
                    text = text.replace(phone_clean, "")
                    # Also remove phone patterns
                    phone_pattern = re.escape(phone_clean)
                    text = re.sub(phone_pattern, "", text, flags=re.IGNORECASE)
        
        # Remove names
        if "Names" in reserved_data and isinstance(reserved_data["Names"], list):
            for name in reserved_data["Names"]:
                if name:
                    text = text.replace(name, "")
                    # Remove name patterns
                    name_pattern = re.escape(name)
                    text = re.sub(name_pattern, "", text, flags=re.IGNORECASE)
        
        # Remove email addresses
        if "EmailAddresses" in reserved_data and isinstance(reserved_data["EmailAddresses"], list):
            for email in reserved_data["EmailAddresses"]:
                if email:
                    text = text.replace(email, "")
                    # Remove email patterns
                    email_pattern = re.escape(email)
                    text = re.sub(email_pattern, "", text, flags=re.IGNORECASE)
        
        # Remove URLs
        if "Urls" in reserved_data and isinstance(reserved_data["Urls"], list):
            for url in reserved_data["Urls"]:
                if url:
                    text = text.replace(url, "")
                    # Remove URL patterns
                    url_pattern = re.escape(url)
                    text = re.sub(url_pattern, "", text, flags=re.IGNORECASE)
        
        # Remove other PII data (dates, nationality, etc.)
        if "OtherData" in reserved_data and isinstance(reserved_data["OtherData"], list):
            for other in reserved_data["OtherData"]:
                if other:
                    # Remove dates like "26/12/1993", "December 26, 1993", "D.O.B: 26/12/1993"
                    date_patterns = [
                        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                        r'(?:D\.O\.B|DOB|Date of Birth)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                        r'(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday),\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
                    ]
                    for pattern in date_patterns:
                        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
                    
                    # Remove nationality, gender patterns
                    if "Nationality:" in other or "Gender:" in other:
                        text = text.replace(other, "")
    
    # Additional regex checks for PII patterns in text (email, phone, LinkedIn)
    # Email patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_pattern, "", text, flags=re.IGNORECASE)
    
    # Phone number patterns (US format: (123) 456-7890, 123-456-7890, 123.456.7890, 1234567890)
    phone_patterns = [
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US phone format
        r'\d{10,}',  # Long number sequences (likely phones)
        r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # International format
    ]
    for pattern in phone_patterns:
        text = re.sub(pattern, "", text)
    
    # LinkedIn URL patterns
    linkedin_patterns = [
        r'linkedin\.com/in/[\w-]+',
        r'www\.linkedin\.com/in/[\w-]+',
        r'https?://(www\.)?linkedin\.com/in/[\w-]+',
        r'LinkedIn:\s*https?://(www\.)?linkedin\.com/in/[\w-]+',
    ]
    for pattern in linkedin_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces that might result from removals
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def _normalize_plaintext(plaintext: str, reserved_data: Optional[Dict[str, Any]] = None, contact_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Normalize PlainText by removing formatting (PII removal is done separately on complete string).
    
    Args:
        plaintext: The raw PlainText from resume
        reserved_data: ReservedData dict (not used here, kept for compatibility)
        contact_info: ContactInformation dict (not used here, kept for compatibility)
        
    Returns:
        PlainText with formatting cleaned (bullets, newlines, tabs, symbols, dates)
    """
    if not plaintext:
        return ""
    
    text = plaintext
    
    # Remove formatting characters
    text = text.replace("\n", " ")  # Replace newlines with spaces
    text = text.replace("\t", " ")  # Replace tabs with spaces
    text = text.replace("\r", " ")  # Replace carriage returns
    
    # Remove bullet points and symbols
    text = re.sub(r'[●•\-\*]\s*', '', text)  # Remove bullets
    text = re.sub(r'@\s*', '', text)  # Remove @ symbols
    
    # Remove date patterns (general cleanup - formatting only, not PII-specific dates)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)  # Remove dates like 26/12/1993
    text = re.sub(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', '', text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def build_candidate_text(candidate: CandidateRawProfile) -> str:
    """
    Build candidate text from resume_data string (new format).
    Only parses resume_data field, nothing else.
    """
    # Check for new format first (resume_data as string)
    if hasattr(candidate, 'resume_data') and candidate.resume_data:
        resume_text = candidate.resume_data
        if isinstance(resume_text, str) and resume_text.strip():
            return resume_text.strip()
    
    # Legacy format fallback (should not be used with new format)
    parts = []
    resume_data = getattr(candidate, 'resumeData', None)  # Legacy format only
    
    if not resume_data:
        # Try to get candidate_id from new or old format
        candidate_id = getattr(candidate, 'candidate_id', None) or getattr(candidate, 'candidateId', 'unknown')
        return f"Candidate ID: {candidate_id}"
    
    # Handle both Pydantic model and dict
    if hasattr(resume_data, 'dict'):
        resume_dict = resume_data.dict()
        resume_obj = resume_data
    else:
        resume_dict = resume_data if isinstance(resume_data, dict) else {}
        resume_obj = None
    
    # Get PlainText and ReservedData
    plaintext = None
    reserved_data = None
    
    # Try to get from ResumeMetadata first
    resume_metadata = None
    if resume_dict.get("ResumeMetadata") and isinstance(resume_dict.get("ResumeMetadata"), dict):
        resume_metadata = resume_dict["ResumeMetadata"]
        plaintext = resume_metadata.get("PlainText")
        reserved_data = resume_metadata.get("ReservedData")
    elif resume_obj and hasattr(resume_obj, 'ResumeMetadata'):
        resume_metadata = resume_obj.ResumeMetadata
        if resume_metadata:
            if hasattr(resume_metadata, 'PlainText'):
                plaintext = resume_metadata.PlainText
            elif isinstance(resume_metadata, dict):
                plaintext = resume_metadata.get("PlainText")
            
            if hasattr(resume_metadata, 'ReservedData'):
                reserved_data = resume_metadata.ReservedData
            elif isinstance(resume_metadata, dict):
                reserved_data = resume_metadata.get("ReservedData")
    
    # Fallback to top-level PlainText
    if not plaintext:
        if resume_obj and hasattr(resume_obj, 'PlainText'):
            plaintext = resume_obj.PlainText
        else:
            plaintext = resume_dict.get("PlainText")
    
    # Get ContactInformation for PII removal
    contact_info = None
    if resume_dict.get("ContactInformation"):
        contact_info = resume_dict["ContactInformation"]
    elif resume_obj and hasattr(resume_obj, 'ContactInformation'):
        contact_info = resume_obj.ContactInformation
        if hasattr(contact_info, 'dict'):
            contact_info = contact_info.dict()
    
    # === 1. NORMALIZED PLAINTEXT (Top Layer) ===
    # Note: We'll do basic formatting cleanup here, but final PII removal happens on complete string
    if plaintext:
        normalized_text = _normalize_plaintext(plaintext, None, None)  # Just formatting, no PII removal yet
        if normalized_text:
            parts.append(normalized_text)
    
    # === 2. OBJECTIVE ===
    objective = None
    if resume_obj and hasattr(resume_obj, 'Objective'):
        objective = resume_obj.Objective
    else:
        objective = resume_dict.get("Objective")
    
    if objective:
        parts.append(f"[OBJECTIVE]\n{objective}")
    
    # === 3. PROFESSIONAL SUMMARY ===
    prof_summary = None
    if resume_obj and hasattr(resume_obj, 'ProfessionalSummary'):
        prof_summary = resume_obj.ProfessionalSummary
    else:
        prof_summary = resume_dict.get("ProfessionalSummary")
    
    if prof_summary:
        parts.append(f"[PROFESSIONAL SUMMARY]\n{prof_summary}")
    
    # === 4. EDUCATION (High Level - Course and College Name) ===
    edu_details = _safe_get(resume_dict, "Education", "EducationDetails", default=[])
    if edu_details and isinstance(edu_details, list):
        # Get highest/most recent education
        sorted_edu = sorted(
            [e for e in edu_details if isinstance(e, dict) and e.get("LastEducationDate")],
            key=lambda e: _safe_get(e, "LastEducationDate", "Date", default=""),
            reverse=True
        )[:1]  # Only highest
        
        for edu in sorted_edu:
            degree = _safe_get(edu, "Degree", "Name", "Raw") or _safe_get(edu, "Degree", "Name", "Normalized")
            school_name = edu.get("SchoolName")
                
            if degree or school_name:
                edu_parts = []
                if degree:
                    edu_parts.append(degree)
                if school_name:
                    edu_parts.append(f"from {school_name}")
        
        if edu_parts:
                    parts.append(f"[EDUCATION]\n{' '.join(edu_parts)}")
    
    # === 5. EMPLOYMENT HISTORY ===
    employment_history = _safe_get(resume_dict, "EmploymentHistory")
    if employment_history and isinstance(employment_history, dict):
        exp_summary = employment_history.get("ExperienceSummary")
        positions = employment_history.get("Positions", [])
        
        # ExperienceSummary
    if exp_summary and isinstance(exp_summary, dict):
        exp_desc = exp_summary.get("Description")
        if exp_desc:
                parts.append(f"[EXPERIENCE SUMMARY]\n{exp_desc}")
        
        # ManagementStory
        if exp_summary and isinstance(exp_summary, dict):
            mgmt_story = exp_summary.get("ManagementStory")
            if mgmt_story:
                parts.append(f"[MANAGEMENT STORY]\n{mgmt_story}")
        
        # Top 5 Positions (sorted by date, most recent first)
        if positions and isinstance(positions, list):
            sorted_positions = sorted(
                [p for p in positions if isinstance(p, dict) and p.get("StartDate")],
                key=lambda p: _safe_get(p, "StartDate", "Date", default=""),
                reverse=True
            )[:5]  # Top 5 only
            
            position_parts = []
            for pos in sorted_positions:
                job_title = _safe_get(pos, "JobTitle", "Raw") or _safe_get(pos, "JobTitle", "Normalized")
                description = pos.get("Description", "")
                
                if job_title and description:
                    # Format as "JobTitle: Description"
                    pos_text = f"{job_title}: {description}"
                    position_parts.append(pos_text)
                elif job_title:
                    position_parts.append(job_title)
            
            if position_parts:
                parts.append(f"[EMPLOYMENT HISTORY]\n" + "\n\n".join(position_parts))
    
    # === 6. QUALIFICATIONS SUMMARY ===
    qualifications = resume_dict.get("QualificationsSummary")
    if not qualifications and resume_obj and hasattr(resume_obj, 'QualificationsSummary'):
        qualifications = resume_obj.QualificationsSummary
    
    if qualifications:
        # Handle both string and dict formats
        if isinstance(qualifications, str):
            parts.append(f"[QUALIFICATIONS SUMMARY]\n{qualifications}")
        elif isinstance(qualifications, dict):
            # If it's a dict, try to extract text
            quals_text = str(qualifications)
            parts.append(f"[QUALIFICATIONS SUMMARY]\n{quals_text}")
    
    # Combine all parts
    final_text = "\n\n".join(parts)
    
    # Remove PII from the complete final string before encoding
    final_text = _remove_pii_from_text(final_text, reserved_data, contact_info)
    
    return final_text


def summarize_if_needed(
    text: str, 
    summarizer_service: Optional['ResumeSummarizerService'] = None,
    candidate: Optional[CandidateRawProfile] = None
) -> str:
    """
    Summarize text if it exceeds 1500 tokens, then remove PII from summarized text.
    
    Args:
        text: The candidate text to potentially summarize
        summarizer_service: Optional summarizer service instance
        candidate: Optional candidate object to extract PII data for removal after summarization
        
    Returns:
        Original text if under 1500 tokens, or summarized text (with PII removed) if over
    """
    if not text or not text.strip():
        return text
    
    # Check token count
    token_count = get_token_count(text)
    TOKEN_THRESHOLD = 800
    
    #if token_count <= TOKEN_THRESHOLD:
    #    return text
    
    #Summarize if over threshold
    if summarizer_service is None:
        logger.warning(f"Text has {token_count} tokens (>{TOKEN_THRESHOLD}) but no summarizer service available. Returning original text.")
        return text
    
    logger.info(f"Text has {token_count} tokens (>{TOKEN_THRESHOLD}), summarizing...")
    try:
        summarized = summarizer_service.summarize(text)
        summarized_tokens = get_token_count(summarized)
        logger.info(f"Summarized: {token_count} tokens -> {summarized_tokens} tokens")
        
        # Remove PII from summarized text if candidate is provided
        if candidate:
            resume_data = getattr(candidate, 'resumeData', None)  # Legacy format only
            if resume_data:
                # Extract PII data similar to build_candidate_text
                if hasattr(resume_data, 'dict'):
                    resume_dict = resume_data.dict()
                    resume_obj = resume_data
                else:
                    resume_dict = resume_data if isinstance(resume_data, dict) else {}
                    resume_obj = None
                
                # Get ReservedData
                reserved_data = None
                resume_metadata = None
                if resume_dict.get("ResumeMetadata") and isinstance(resume_dict.get("ResumeMetadata"), dict):
                    resume_metadata = resume_dict["ResumeMetadata"]
                    reserved_data = resume_metadata.get("ReservedData")
                elif resume_obj and hasattr(resume_obj, 'ResumeMetadata'):
                    resume_metadata = resume_obj.ResumeMetadata
                    if resume_metadata:
                        if hasattr(resume_metadata, 'ReservedData'):
                            reserved_data = resume_metadata.ReservedData
                        elif isinstance(resume_metadata, dict):
                            reserved_data = resume_metadata.get("ReservedData")
                
                # Get ContactInformation
                contact_info = None
                if resume_dict.get("ContactInformation") and isinstance(resume_dict.get("ContactInformation"), dict):
                    contact_info = resume_dict["ContactInformation"]
                elif resume_obj and hasattr(resume_obj, 'ContactInformation'):
                    contact_info_obj = resume_obj.ContactInformation
                    if contact_info_obj:
                        if hasattr(contact_info_obj, 'dict'):
                            contact_info = contact_info_obj.dict()
                        elif isinstance(contact_info_obj, dict):
                            contact_info = contact_info_obj
                
                # Remove PII from summarized text
                if reserved_data or contact_info:
                    summarized = _remove_pii_from_text(summarized, reserved_data, contact_info)
                    logger.info("Removed PII from summarized text")
        
        return summarized
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        # Return original text if summarization fails
        return text


def _parse_location(location_str: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse location string into city and country.
    Simple parsing: assumes format like "City, Country" or "City, State, Country"
    """
    if not location_str:
        return None, None
    
    # Try to split by comma
    parts = [p.strip() for p in location_str.split(",")]
    
    if len(parts) >= 2:
        # Last part is likely country
        country = parts[-1]
        city = parts[0] if len(parts) > 1 else None
        return city, country
    elif len(parts) == 1:
        # Single value - assume it's city or country, try to guess
        return parts[0], None
    
    return None, None


def build_candidate_payload(candidate: CandidateRawProfile, text_summarized: bool = False) -> Dict[str, Any]:
    """
    Build metadata payload with required fields from new format:
    - candidate_id
    - client_id
    - primary_role_code
    - current_role (extracted from latest title_exp)
    - skills (array)
    - country, city
    - years_experience_total
    - text_summarized (True if Anthropic Claude summarization was used, False otherwise)
    """
    # Check for new format first
    if hasattr(candidate, 'candidate_id') and candidate.candidate_id:
        payload = {
            "candidate_id": candidate.candidate_id,
            "client_id": candidate.client_id,
        }
        
        # Extract from new format fields
        if candidate.years_experience_total is not None:
            payload["years_experience_total"] = candidate.years_experience_total
        
        if candidate.country:
            payload["country"] = candidate.country
        
        if candidate.city:
            payload["city"] = candidate.city
        
        if candidate.skills:
            payload["skills"] = candidate.skills
        
        if candidate.primary_role_code:
            payload["primary_role_code"] = candidate.primary_role_code
        
        # Extract current_role from latest title_exp entry
        if candidate.title_exp and len(candidate.title_exp) > 0:
            # Get the first entry (latest/most recent)
            latest_title = candidate.title_exp[0]
            if latest_title and latest_title.title:
                payload["current_role"] = latest_title.title
        elif candidate.current_role:
            payload["current_role"] = candidate.current_role
        
        # Add text_summarized metadata
        payload["text_summarized"] = text_summarized
        
        return payload
    
    # Legacy format fallback (should not be used with new format)
    resume_data = getattr(candidate, 'resumeData', None)  # Legacy format only
    payload = {
        "candidate_id": getattr(candidate, 'candidate_id', None) or getattr(candidate, 'candidateId', 'unknown'),
        "client_id": getattr(candidate, 'client_id', None) or getattr(candidate, 'clientId', None),
    }
    
    if not resume_data:
        return payload
    
    # years_experience_total
    exp_summary = _safe_get(resume_data.dict(), "EmploymentHistory", "ExperienceSummary")
    if exp_summary and isinstance(exp_summary, dict):
        months = exp_summary.get("MonthsOfWorkExperience")
        if months:
            payload["years_experience_total"] = round(months / 12, 1)
    
    # country, city (parse from location)
    location = _safe_get(resume_data.dict(), "PersonalAttributes", "CurrentLocation") or \
               _safe_get(resume_data.dict(), "PersonalAttributes", "PreferredLocation")
    if location:
        city, country = _parse_location(location)
        if city:
            payload["city"] = city
        if country:
            payload["country"] = country
    
    # skills (array of normalized names)
    skills_list = []
    skills_data = _safe_get(resume_data.dict(), "Skills", "Raw", default=[])
    if skills_data and isinstance(skills_data, list):
        for skill in skills_data[:50]:  # Top 50 skills
            if isinstance(skill, dict):
                skill_name = skill.get("Name", "").replace("\n", " ").strip()
                if skill_name:
                    skills_list.append(skill_name)
    if skills_list:
        payload["skills"] = skills_list
    
    # Get current position for primary_role_code and current_role
    positions = _safe_get(resume_data.dict(), "EmploymentHistory", "Positions", default=[])
    current_pos = None
    if positions:
        for pos in positions:
            if isinstance(pos, dict) and pos.get("IsCurrent"):
                current_pos = pos
                break
        if not current_pos and positions:
            current_pos = positions[0]
    
    if current_pos:
        # current_role (job title from current position)
        job_title = _safe_get(current_pos, "JobTitle", "Raw") or _safe_get(current_pos, "JobTitle", "Normalized")
        if job_title:
            payload["current_role"] = job_title
        
        # primary_role_code (from current position's NormalizedProfession)
        norm_prof = current_pos.get("NormalizedProfession")
        if norm_prof and isinstance(norm_prof, dict):
            # Try Profession.CodeId first, then ISCO.CodeId, then ONET.CodeId
            prof_code = _safe_get(norm_prof, "Profession", "CodeId")
            if prof_code:
                payload["primary_role_code"] = str(prof_code)
            else:
                isco_code = _safe_get(norm_prof, "ISCO", "CodeId")
                if isco_code:
                    payload["primary_role_code"] = str(isco_code)
                else:
                    onet_code = _safe_get(norm_prof, "ONET", "CodeId")
                    if onet_code:
                        payload["primary_role_code"] = str(onet_code)
    
    # Add text_summarized metadata (for legacy format too)
    payload["text_summarized"] = text_summarized
    
    return payload

