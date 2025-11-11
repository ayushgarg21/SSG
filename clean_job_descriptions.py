"""
Job Description Cleaner
Removes irrelevant information from job descriptions and creates structured, clean fields.

Removes:
- HTML tags
- Email addresses
- Phone numbers
- EA license numbers and recruitment agency boilerplate
- Non-job-related content

Creates new fields:
- job_description_clean: Cleaned description without contact info
- job_responsibilities: Extracted responsibilities section
- job_requirements: Extracted requirements section
- company_info: Extracted company description
"""

import pandas as pd
import re
from bs4 import BeautifulSoup
import html
from pathlib import Path

# Configuration
INPUT_CSV = "./jobtech_job_export_Q3_2025_20251110_203838.csv"
OUTPUT_CSV = "./jobtech_job_export_cleaned.csv"
SAMPLE_SIZE = None  # Set to a number to test on sample, None for full dataset

# Patterns to remove
BOILERPLATE_PATTERNS = [
    # Recruitment agency disclaimers
    r'We regret (?:to inform you )?that only shortlisted (?:candidate|applicant)s? (?:will be|shall be) (?:notified|contacted|informed)\.?',
    r'Only shortlisted (?:candidate|applicant)s? will be (?:notified|contacted)\.?',

    # EA License information
    r'EA Licen[cs]e(?:\s+No\.?|\s+Number)?:?\s*[\w\d]+',
    r'EA Reg(?:istration)?(?:\s+No\.?|\s+Number)?:?\s*[\w\d]+',
    r'EA Personnel(?:\s+No\.?|\s+Number)?:?\s*[\w\d]+',
    r'EA Personnel Name:?\s*[\w\s]+',
    r'GMP Recruitment Services.*?PDPA',
    r'e2i.*?(?:Employment and Employability Institute|manpower and skills upgrading)',

    # Contact/application instructions that don't add value
    r'(?:If you )?(?:wish to apply|interested candidates?)(?: for this (?:position|role|job))?,?\s*(?:please|kindly)?\s*(?:click|send|email|submit).*?(?:\.|<br/>)',
    r'To apply(?:,| for this (?:position|role))?,?\s*(?:please|kindly)?.*?(?:\.|<br/>)',

    # Common signoffs
    r'Thank you for your (?:interest|application)\.?',
    r'We look forward to (?:hearing from you|your application)\.?',
]

# Email and phone patterns
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
PHONE_PATTERN = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}'

# Section headers
SECTION_HEADERS = {
    'responsibilities': [
        r'(?:key\s+)?responsibilit(?:ies|y)',
        r'job\s+(?:description|responsibilities)',
        r'duties',
        r'what\s+you(?:\'ll|\s+will)\s+do',
        r'role\s+(?:description|responsibilities)',
    ],
    'requirements': [
        r'(?:key\s+)?requirements?',
        r'qualifications?',
        r'(?:skills?|experience)\s+(?:required|needed)',
        r'what\s+(?:we(?:\'re|\s+are)\s+looking\s+for|you\s+need)',
        r'candidate\s+profile',
    ],
    'company': [
        r'about\s+(?:us|the\s+company|our\s+company)',
        r'company\s+(?:overview|description|profile|background)',
        r'who\s+we\s+are',
        r'our\s+(?:client|company)',
    ]
}

def clean_html(text):
    """Remove HTML tags and decode entities"""
    if pd.isna(text) or text == '':
        return ''

    # Decode HTML entities
    text = html.unescape(text)

    # Parse HTML and extract text
    soup = BeautifulSoup(text, 'html.parser')

    # Replace <br> tags with newlines before extracting text
    for br in soup.find_all('br'):
        br.replace_with('\n')

    text = soup.get_text(separator=' ')

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()

    return text

def remove_contact_info(text):
    """Remove email addresses and phone numbers"""
    if pd.isna(text) or text == '':
        return ''

    # Remove emails
    text = re.sub(EMAIL_PATTERN, '[EMAIL REMOVED]', text, flags=re.IGNORECASE)

    # Remove phone numbers
    text = re.sub(PHONE_PATTERN, '[PHONE REMOVED]', text)

    # Clean up multiple consecutive removals
    text = re.sub(r'(?:\[EMAIL REMOVED\]\s*){2,}', '[EMAIL REMOVED]', text)
    text = re.sub(r'(?:\[PHONE REMOVED\]\s*){2,}', '[PHONE REMOVED]', text)

    return text

def remove_boilerplate(text):
    """Remove recruitment agency boilerplate and disclaimers"""
    if pd.isna(text) or text == '':
        return ''

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # Clean up resulting whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()

    return text

def extract_section(text, section_type):
    """
    Extract a specific section from the job description.

    Args:
        text: The cleaned job description
        section_type: 'responsibilities', 'requirements', or 'company'

    Returns:
        Extracted section text or empty string
    """
    if pd.isna(text) or text == '' or section_type not in SECTION_HEADERS:
        return ''

    # Build pattern for this section type
    header_patterns = SECTION_HEADERS[section_type]
    header_regex = '|'.join(header_patterns)

    # Find all section headers in the text
    all_headers = []
    for section_name, patterns in SECTION_HEADERS.items():
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                all_headers.append((match.start(), match.end(), section_name))

    # Sort by position
    all_headers.sort(key=lambda x: x[0])

    # Find the section we want
    section_text = ''
    for i, (start, end, name) in enumerate(all_headers):
        if name == section_type:
            # Find the end of this section (start of next section or end of text)
            section_start = end
            if i + 1 < len(all_headers):
                section_end = all_headers[i + 1][0]
            else:
                section_end = len(text)

            section_text = text[section_start:section_end].strip()
            break

    return section_text

def clean_job_description(row):
    """
    Clean a single job description and extract structured information.

    Returns a dictionary with cleaned fields.
    """
    raw_desc = row['job_description']

    # Step 1: Clean HTML
    clean_text = clean_html(raw_desc)

    # Step 2: Remove contact info
    clean_text = remove_contact_info(clean_text)

    # Step 3: Remove boilerplate
    clean_text = remove_boilerplate(clean_text)

    # Step 4: Extract sections
    responsibilities = extract_section(clean_text, 'responsibilities')
    requirements = extract_section(clean_text, 'requirements')
    company_info = extract_section(clean_text, 'company')

    return {
        'job_description_clean': clean_text,
        'job_responsibilities': responsibilities,
        'job_requirements': requirements,
        'company_info': company_info,
        'clean_word_count': len(clean_text.split()) if clean_text else 0
    }

def main():
    """Main processing function"""
    print(f"Loading CSV: {INPUT_CSV}")

    # Load data
    if SAMPLE_SIZE:
        df = pd.read_csv(INPUT_CSV, nrows=SAMPLE_SIZE)
        print(f"✓ Loaded sample of {len(df):,} job postings")
    else:
        df = pd.read_csv(INPUT_CSV)
        print(f"✓ Loaded {len(df):,} job postings")

    print(f"\nCleaning job descriptions...")
    print(f"{'='*80}")

    # Process each row
    cleaned_data = []
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"Processed {idx:,} / {len(df):,} rows...")

        cleaned = clean_job_description(row)
        cleaned_data.append(cleaned)

    # Create DataFrame with cleaned data
    cleaned_df = pd.DataFrame(cleaned_data)

    # Combine with original data
    df_output = pd.concat([df, cleaned_df], axis=1)

    # Save results
    print(f"\nSaving cleaned data to: {OUTPUT_CSV}")
    df_output.to_csv(OUTPUT_CSV, index=False)

    # Print summary
    print(f"\n{'='*80}")
    print("CLEANING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total jobs processed: {len(df_output):,}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"File size: {Path(OUTPUT_CSV).stat().st_size / 1024 / 1024:.2f} MB")

    print(f"\nCleaning Statistics:")
    print(f"  - Jobs with extracted responsibilities: {(df_output['job_responsibilities'] != '').sum():,}")
    print(f"  - Jobs with extracted requirements: {(df_output['job_requirements'] != '').sum():,}")
    print(f"  - Jobs with company info: {(df_output['company_info'] != '').sum():,}")

    print(f"\nWord Count Statistics (cleaned descriptions):")
    print(df_output['clean_word_count'].describe())

    # Show examples
    print(f"\n{'='*80}")
    print("SAMPLE CLEANED JOB DESCRIPTIONS")
    print(f"{'='*80}\n")

    sample = df_output[df_output['job_description_clean'] != ''].sample(min(3, len(df_output)))

    for idx, row in sample.iterrows():
        print(f"Job Title: {row['job_title']}")
        print(f"Original length: {len(str(row['job_description']))} chars")
        print(f"Cleaned length: {len(row['job_description_clean'])} chars")
        print(f"\nCleaned Description (first 300 chars):")
        print(row['job_description_clean'][:300])
        if row['job_responsibilities']:
            print(f"\nResponsibilities (first 200 chars):")
            print(row['job_responsibilities'][:200])
        if row['job_requirements']:
            print(f"\nRequirements (first 200 chars):")
            print(row['job_requirements'][:200])
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    try:
        import bs4
        main()
    except ImportError:
        print("Error: BeautifulSoup4 is required.")
        print("Install it with: pip install beautifulsoup4")
