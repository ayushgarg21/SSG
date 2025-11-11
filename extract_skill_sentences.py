"""
Skill Sentence Extractor
Extracts complete sentences containing skills that were identified by the skill extractor.
Uses the start/end positions from the skill extraction output.
"""

import pandas as pd
import re
from pathlib import Path
from bs4 import BeautifulSoup
import html

# Configuration
INPUT_SKILLS_CSV = "./jobtech_skills_extracted.csv"
OUTPUT_CSV = "./jobtech_skills_with_context.csv"

def clean_html(text):
    """Remove HTML tags and decode HTML entities"""
    if pd.isna(text) or text == '':
        return ''

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator=' ')

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

def extract_sentence(text, start_pos, end_pos):
    """
    Extract 10 words before and 10 words after the skill.

    Args:
        text: The full text
        start_pos: Start position of the skill
        end_pos: End position of the skill

    Returns:
        A string with 10 words before + skill + 10 words after
    """
    if pd.isna(text) or text == '' or start_pos < 0 or end_pos > len(text):
        return ''

    # Clean HTML first
    clean_text = clean_html(text)

    # Extract the skill text from original position
    skill_text = text[start_pos:end_pos] if start_pos < len(text) and end_pos <= len(text) else ''

    if not skill_text:
        return ''

    # Find the skill in cleaned text
    skill_clean = clean_html(skill_text)

    # Try to find the position in cleaned text
    try:
        clean_start = clean_text.lower().find(skill_clean.lower())
        if clean_start == -1:
            # Skill not found in cleaned text, return context around original position
            # This can happen due to HTML cleaning
            context_start = max(0, start_pos - 100)
            context_end = min(len(text), end_pos + 100)
            return clean_html(text[context_start:context_end])
    except:
        return ''

    # Split text into words using whitespace and common punctuation as delimiters
    # But keep the skill position intact
    words_pattern = r'\S+'  # Match sequences of non-whitespace characters

    # Find all words with their positions
    words_with_positions = []
    for match in re.finditer(words_pattern, clean_text):
        words_with_positions.append({
            'word': match.group(),
            'start': match.start(),
            'end': match.end()
        })

    # Find which word contains or is closest to the skill
    skill_end_pos = clean_start + len(skill_clean)
    skill_word_index = -1

    for i, word_info in enumerate(words_with_positions):
        # Check if this word overlaps with the skill position
        if (word_info['start'] <= clean_start < word_info['end']) or \
           (word_info['start'] < skill_end_pos <= word_info['end']) or \
           (clean_start <= word_info['start'] and skill_end_pos >= word_info['end']):
            skill_word_index = i
            break

    if skill_word_index == -1:
        # Couldn't find the skill word, return empty
        return ''

    # Extract 10 words before and 10 words after
    start_index = max(0, skill_word_index - 10)
    end_index = min(len(words_with_positions), skill_word_index + 11)  # +11 to include skill + 10 after

    # Get the words
    selected_words = [w['word'] for w in words_with_positions[start_index:end_index]]

    # Join the words
    context = ' '.join(selected_words)

    return context

def process_skills_file():
    """Process the skills extraction output and add sentence context"""
    print(f"Loading skills data from: {INPUT_SKILLS_CSV}")

    if not Path(INPUT_SKILLS_CSV).exists():
        print(f"Error: Input file not found: {INPUT_SKILLS_CSV}")
        print("Please run batch_skill_extraction.py first!")
        return

    df = pd.read_csv(INPUT_SKILLS_CSV)
    print(f"✓ Loaded {len(df):,} skill instances")
    print(f"  - From {df['job_id'].nunique():,} unique jobs")
    print(f"  - {df['skill'].nunique():,} unique skills")

    # Extract context (10 words before + skill + 10 words after)
    print(f"\nExtracting skill context (10 words before/after)...")

    df['skill_context'] = df.apply(
        lambda row: extract_sentence(
            row['job_description'],
            int(row['start']) if pd.notna(row['start']) else -1,
            int(row['end']) if pd.notna(row['end']) else -1
        ),
        axis=1
    )

    # Also clean the full job description for reference
    print("Cleaning job descriptions...")
    df['job_description_clean'] = df['job_description'].apply(clean_html)

    # Calculate context statistics
    df['context_length'] = df['skill_context'].str.len()
    df['context_word_count'] = df['skill_context'].str.split().str.len()

    # Reorder columns for better readability
    output_columns = [
        'job_id',
        'job_title',
        'skill_id',
        'skill',
        'skill_type',
        'skill_category',
        'is_core_skill',
        'skill_count',
        'weight',
        'skill_context',  # 10 words before + skill + 10 words after
        'context_length',
        'context_word_count',
        'start',
        'end',
        'job_description_clean',  # Full cleaned description for reference
        'job_description'  # Original with HTML
    ]

    df_output = df[output_columns].copy()

    # Save results
    print(f"\nSaving results to: {OUTPUT_CSV}")
    df_output.to_csv(OUTPUT_CSV, index=False)

    # Print summary
    print(f"\n{'='*80}")
    print("SKILL CONTEXT EXTRACTION COMPLETE!")
    print(f"{'='*80}")
    print(f"Total skill instances: {len(df_output):,}")
    print(f"With context extracted: {(df_output['skill_context'] != '').sum():,}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"File size: {Path(OUTPUT_CSV).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"\nContext Word Count Statistics:")
    print(df_output['context_word_count'].describe())
    print(f"\nContext Length (characters) Statistics:")
    print(df_output['context_length'].describe())
    print(f"\nSample extracted contexts (10 words before + skill + 10 words after):")
    print(f"{'='*80}")

    # Show some examples
    samples = df_output[df_output['skill_context'] != ''].sample(min(5, len(df_output)))
    for idx, row in samples.iterrows():
        print(f"\nSkill: {row['skill']} ({row['skill_type']})")
        print(f"Context: {row['skill_context']}")
        print(f"Word count: {row['context_word_count']}")
        print("-" * 80)

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    try:
        # Check if BeautifulSoup is available
        import bs4
        process_skills_file()
    except ImportError:
        print("Error: BeautifulSoup4 is required for HTML cleaning.")
        print("Install it with: pip install beautifulsoup4")
