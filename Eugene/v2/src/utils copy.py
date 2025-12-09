import re
from bs4 import BeautifulSoup
import pysbd


# HTML -> plain text (convert <br/> to \n then strip)
def html_to_text(html: str) -> str:
    if not isinstance(html, str):
        return ""
    # Normalize <br> to \n for consistent paragraph splitting
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    # Remove remaining tags but keep their text
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    # Collapse multiple blank lines to two newlines marker
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = text.strip()
    return text


# Sentence segmentation using pysbd
_seg = pysbd.Segmenter(language="en", clean=True)


def split_sentences(text: str):
    if not text:
        return []
    return _seg.segment(text)


# Heuristic: group sentences into paragraphs using blank lines and heading cues
HEADER_PATTERNS = [
    re.compile(r"^\s*(ABOUT|ABOUT US|WHO WE ARE|COMPANY|COMPANY PROFILE|COMPANY OVERVIEW|OUR MISSION|WHAT WE DO)[:\-]?$", re.IGNORECASE),
]


BULLET_MARKER = re.compile(r"^\s*[-*•\d\)]\s+")




def group_paragraphs_from_text(text: str, min_sentences=1):
    """
    Returns a list of paragraph texts. Heuristics:
    - Split on double newlines
    - If no double newlines, split by groups of sentences, merging based on header detection and bullets
    """
    if not text:
        return []

    # Primary fast split: double newline separates paragraphs
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    paragraphs = []

    # If blocks already look like paragraphs, return them after cleanup
    if len(blocks) > 1:
        for b in blocks:
            # rejoin lines within blocks if they are broken by single newlines
            lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
            para = " ".join(lines)
            paragraphs.append(para)
        return paragraphs

    # Fallback: sentence segmentation + heuristic grouping
    sents = split_sentences(text)
    cur = []
    for s in sents:
        # If sentence looks like a header, flush current
        if any(p.search(s.strip()) for p in HEADER_PATTERNS):
            if cur:
                paragraphs.append(" ".join(cur).strip())
                cur = []
            paragraphs.append(s.strip())
            continue

        # Bullet starts -> start new paragraph
        if BULLET_MARKER.search(s.strip()):
            if cur:
                paragraphs.append(" ".join(cur).strip())
                cur = []

        cur.append(s)

    # Flush remaining
    if cur:
        paragraphs.append(" ".join(cur).strip())

    # Filter out very short paragraphs
    filtered = [p for p in paragraphs if len(p.split()) >= min_sentences]
    return filtered


""" PARTS WHERE WE SEGEMNT SETENCES INTO PARAGRAPHS BY COMPARING. SEMANTIC MEANING BETWEEN ADJACENT SENTENCES """
import pysbd
from sentence_transformers import SentenceTransformer, util
import torch

# Detect best device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using MPS (Apple GPU) acceleration")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using CUDA (NVIDIA GPU) acceleration")
else:
    DEVICE = "cpu"
    print("Using CPU (no GPU acceleration)")

# Load model once globally for reuse
_semantic_model = None

def get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    return _semantic_model

def semantic_paragraph_segmentation(text, similarity_threshold=0.55):
    # Step 1 — sentence split
    seg = pysbd.Segmenter(language="en", clean=True)
    sentences = seg.segment(text)

    if not sentences:
        return []

    # Step 2 — embed the sentences using cached model
    model = get_semantic_model()
    embeddings = model.encode(sentences, convert_to_tensor=True, device=DEVICE)

    # Step 3 — create semantic paragraphs
    paragraphs = []
    current = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeddings[i-1], embeddings[i]).item()

        # Topic shift → new paragraph
        if sim < similarity_threshold:
            paragraphs.append(" ".join(current))
            current = []
        current.append(sentences[i])

    # Add the last paragraph
    if current:
        paragraphs.append(" ".join(current))

    return paragraphs

# paragraphs = semantic_paragraph_segmentation(job_description_text)
# for p in paragraphs:
#     print("----")
#     print(p)
## THIS IS ANOTHER CJHUNKER
from unstructured.partition.text import partition_text

def unstructured_parse(text):
    """Parse text using unstructured library"""
    if not text or len(text.strip()) == 0:
        return []

    try:
        # Write to temp file to avoid filename length issues
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(text)
            temp_path = f.name

        try:
            from unstructured.partition.text import partition_text
            elements = partition_text(filename=temp_path)
            parsed = []

            for el in elements:
                parsed.append({
                    "type": el.__class__.__name__,
                    "text": el.text.strip()
                })

            return parsed
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    except Exception as e:
        # Fallback: just return the text as one block
        print(f"Warning: unstructured_parse failed ({e}), using fallback")
        return [{"type": "NarrativeText", "text": text}]
    
job_description_text = """Position Summary:<br/><br/>As a Manufacturing Equipment Engineer 1, you will provide technical leadership in sustaining smart manufacturing equipment within a high-volume consumables production environment. You will define and carry out maintenance activities across optical, mechanical, electrical, and fluidics systems, with a focus on improving equipment uptime and reliability. Applying engineering principles, you will troubleshoot and resolve equipment-related issues. You may also be required to train Technicians to enhance their understanding of job scopes and responsibilities, ensuring their performance meets expectations. Additionally, you may be involved in guiding and mentoring Technicians to support their ongoing development.<br/><br/>Position Responsibilities:<br/><br/>Complete required training (e.g. cGMP, safety and work instructions, etc.) within a stipulated time frame.<br/>Execute tasks strictly following cGMP, Quality, Safety and Work instruction requirements.<br/>Ensure employees follow safety, quality requirement, and all applicable company policies at all times.<br/>Ensure proper housekeeping and maintain cleanliness (6S) of the working area.<br/>Timely report/ escalate any concerns (work-related or personal).<br/>Meet daily production schedule and performance expectation.<br/>Able to take on additional tasks or responsibilities when required.<br/>Competent to perform preventive maintenance activities with some supervision.<br/>Implement repair procedures to ensure correct operation of equipment and systems with some supervision.<br/>Identify and update critical spare parts in SAP to ensure continuous operation<br/>Improvement work:<br/>- Participate and lead improvement initiatives within the same functional area under broad direction.<br/>- Facilitate an organization's systems and processes relating to continuous improvement (e.g. 6S) and coach team members toward continuous improvement.<br/>- Manage the work instructions of each Equipment maintenance process to ensure consistent performance of the equipment.<br/>Incident investigation and reporting:<br/>- Lead an investigation team and able to submit a formal report under some supervision.<br/>- Provide guidance to team members.<br/>- Adapt different techniques and concepts in technical writing for effective engagement with individuals and/or teams.<br/>Lead and mentor a group of team members to meet production KPI and corporate goals &amp; targets:<br/>- Assist and provide technical support to the shift to meet daily production target and shift KPIs with some guidance.<br/>- Co-ordinate &amp; prioritize activities based on production, equipment and supply to meet the objectives.<br/>Learning and Development:<br/>- Contributie to the skill development of the shift.<br/>- Able to provide guidance to Technician on PLC and automation equipment troubleshooting.<br/>- Develop On the job (OJT) training programmes for Technician.<br/>Demonstrates proficiency in performing repair and preventive maintenance on basic smart manufacturing equipment and in developing effective solutions for recurring equipment problems.<br/>Implement improvements to the existing equipment maintenance processes with direct supervision.<br/>Identify opportunities to automate manual processes to improve product quality, cycle time, and cost.<br/>Data Collection and Analysis:<br/>Demonstrates the ability to collect and interpret simple operational data to identify basic trends that support equipment efficiency and routine operations.<br/>Be a role model for Technicians.<br/><br/>Position Requirements:<br/><br/>Bachelor&rsquo;s Degree in Electrical, Electronics, Mechanical or Mechatronics Engineering or related field of study with 0-2 years of relevant of working experience in manufacturing/production environment. Candidates with Diploma or NTC/NiTEC qualifications may also be considered with minimum 6 years of relevant working experience.<br/>Experience in GMP controlled manufacturing / production environment.<br/>Advanced understanding on Electrical / Electronics / Mechanical / Pneumatic<br/>Ensure compliance with cGMP, Quality, work safety instructions &amp; practices at all times.<br/>Experience in troubleshooting PLC (e.g. Beckhoff or Allen Bradley PLC).<br/>Complete tasks in a safe and efficient manner.<br/>High level of discipline and integrity.<br/>Able to undertake multiple tasks.<br/>Able to undergo job rotation/ cross training.<br/>Able to work in noisy environment with appropriate personal protection equipment (PPE).<br/>Able to work in chemical environment with appropriate personal protection equipment (PPE).<br/>Must be able to identify different colors for work purposes. In Illumina, we handle components/chemicals with different colors, and it is necessary to be able to differentiate and identify the correct components/chemicals (using visual) during operation.<br/>Possess a positive attitude and sense of urgency.<br/>Meticulous, keen attention to details and organized.<br/>(For Shift Engineer) The candidate must be able to work 12 hours (from 0800 to 2015 for the day shift and from 2000 to 0815 for the night shift) rotating shifts (Monthly rotations from day to night and vice versa) with staggered break times in accordance with the roster.<br/>Be agreeable to work normal shift when required.<br/>Able to safely lift and handle weights up to 15 kg, with occasional requirements to manage heavier loads<br/><br/>All listed requirements are deemed as essential functions to this position; however, business conditions may require reasonable accommodations for additional tasks and responsibilities."""
  
parsed = unstructured_parse(job_description_text)

for item in parsed:
    print(f"[{item['type']}] {item['text']}")

# Example outputs:

# [Title] Job Overview
# [NarrativeText] An Arborist executes tree care management programmes...
# [Heading] Key Responsibilities:
# [BulletedText] Assist in preparation of the Tree Assessment Report...
# [Heading] Qualifications:
# [NarrativeText] English speaking Certified Arborist ISA...

def hybrid_paragraph_segmentation(text):
    structured = unstructured_parse(text)
    paragraphs = []

    for block in structured:
        if block["type"] in ["NarrativeText", "BulletedText"]:
            # refine using semantic segmentation
            refined = semantic_paragraph_segmentation(block["text"], similarity_threshold=0.50)
            paragraphs.extend(refined)
        else:
            # keep headings as they are
            paragraphs.append(block["text"])

    return paragraphs
