import os
import re
import json
import fitz
from pathlib import Path


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PDF_FOLDER = BASE_DIR / "documents"
OUTPUT_FOLDER = BASE_DIR / "output"


CHUNK_SIZE = 250
OVERLAP = 50


os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------------------------------------------------
# Basic text cleaning
# ---------------------------------------------------------

def clean_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------
# Remove repeated PDF headers and footers
# ---------------------------------------------------------

def remove_header_footer(text):
    lines = text.splitlines()

    cleaned = []

    for line in lines:

        line = line.strip()

        if not line:
            cleaned.append("")
            continue

        # Repeated hospital header
        if line.startswith("Green Valley Community Hospital"):
            continue

        # Educational footer
        if "Fictional document created for educational use only" in line:
            continue

        # Page footer
        if re.search(r"Page\s+\d+", line, re.IGNORECASE):
            continue

        if "Policy ID:" in line and "Page" in line:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ---------------------------------------------------------
# Detect whether a line looks like a section heading
# ---------------------------------------------------------

def looks_like_heading(line, font_size=None, is_bold=False):
    line = line.strip()

    if not line:
        return False

    # Q&A lines are content, not section headings
    if line.startswith("Q:") or line.startswith("A:"):
        return False

    # Very long lines are probably paragraphs
    if len(line) > 90:
        return False

    # Sentences are usually paragraphs
    if line.endswith("."):
        return False

    # Common table rows should not become headings
    if "|" in line:
        return False

    # Obvious metadata
    if line.startswith("Policy ID"):
        return False

    if line.startswith("Effective Date"):
        return False

    if line.startswith("Document Status"):
        return False

    # Typography based detection
    if is_bold and len(line.split()) <= 12:
        return True

    if font_size is not None and font_size >= 12 and len(line.split()) <= 12:
        return True

    # Known heading style fallback
    known_headings = {
        "Purpose",
        "Scope",
        "Policy Statement",
        "General Visiting Hours",
        "Visitor Registration and Identification",
        "Visitor Passes",
        "Maximum Visitors and Visitor Limits",
        "Children Visiting",
        "ICU Access and Restricted Areas",
        "Quiet Hours and Visitor Conduct",
        "Lost Visitor Passes",
        "Visiting Exceptions",
        "Security Desk",
        "Discharge Request",
        "Discharge Preparation Steps",
        "Final Billing and Insurance Coordination",
        "Document Collection and Discharge Summary",
        "Property Check and Room Clearance",
        "Patient Confirmation and Administrative Signatures",
        "Transport Arrangements",
        "Discharge Delays",
        "Family Responsibilities",
        "Post Discharge Administrative Contact",
        "Typical Discharge Timeline",
        "Staff Identification Cards",
        "Administrative Office Hours",
        "Attendance",
        "Leave Requests",
        "Leave Request Steps",
        "Shift Change Requests",
        "Staff Records and Department Transfers",
        "Internal Communication",
        "Workplace Conduct",
        "Request Submission and Approval Process",
        "Administrative Support",
        "Meeting Schedules",
        "Meeting Agendas",
        "What an Agenda Should Contain",
        "Agenda Submission",
        "Attendance",
        "Minutes and Meeting Records",
        "Action Items and Follow Up",
        "Meeting Responsibilities",
        "Special Meetings",
        "Meeting Cancellation",
        "Patient Administrative Records",
        "Requesting Records",
        "Record Request Steps",
        "Identity Verification",
        "Authorized Representatives",
        "Request Submission and Processing Timeline",
        "Digital and Printed Records",
        "Record Correction Requests",
        "Missing Information and Document Collection",
        "Record Privacy",
        "Record Retention",
        "Appointment Booking",
        "New Appointments",
        "Returning Appointments",
        "Appointment Confirmation",
        "Outpatient Registration",
        "Rescheduling",
        "How to Reschedule",
        "Cancellation",
        "Late Arrival",
        "Waiting Area",
        "Appointment Records",
        "Missed Appointments"
    }

    if line in known_headings:
        return True

    return False


# ---------------------------------------------------------
# Extract PDF while preserving page and heading information
# ---------------------------------------------------------

def extract_sections(pdf_path):

    doc = fitz.open(pdf_path)

    sections = []

    current_section = "Document Introduction"
    current_content = []
    current_pages = []

    for page_number, page in enumerate(doc, start=1):

        page_dict = page.get_text("dict")

        for block in page_dict["blocks"]:

            if "lines" not in block:
                continue

            for line in block["lines"]:

                spans = line["spans"]

                if not spans:
                    continue

                text = "".join(
                    span["text"]
                    for span in spans
                ).strip()

                if not text:
                    continue

                max_font_size = max(
                    span["size"]
                    for span in spans
                )

                is_bold = any(
                    "bold" in span["font"].lower()
                    for span in spans
                )

                # Remove obvious header/footer text
                if "Fictional document created for educational use only" in text:
                    continue

                if text.startswith("Green Valley Community Hospital"):
                    continue

                if re.search(r"Page\s+\d+", text):
                    continue

                # Detect heading
                if looks_like_heading(
                    text,
                    max_font_size,
                    is_bold
                ):

                    # Save previous section
                    if current_content:

                        section_text = clean_text(
                            "\n".join(current_content)
                        )

                        if section_text:
                            sections.append({
                                "section": current_section,
                                "text": section_text,
                                "pages": sorted(set(current_pages))
                            })

                    current_section = text
                    current_content = []
                    current_pages = []

                else:

                    current_content.append(text)
                    current_pages.append(page_number)

    # Save final section
    if current_content:

        section_text = clean_text(
            "\n".join(current_content)
        )

        if section_text:
            sections.append({
                "section": current_section,
                "text": section_text,
                "pages": sorted(set(current_pages))
            })

    doc.close()

    return sections


# ---------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------

def split_into_paragraphs(text):

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    paragraphs = [
        clean_text(p)
        for p in paragraphs
        if clean_text(p)
    ]

    return paragraphs


# ---------------------------------------------------------
# Word based chunking
# ---------------------------------------------------------

def create_chunks(
    section_text,
    section_name,
    pages
):

    paragraphs = split_into_paragraphs(
        section_text
    )

    chunks = []

    current_words = []

    for paragraph in paragraphs:

        words = paragraph.split()

        # If paragraph itself is larger than chunk size
        if len(words) > CHUNK_SIZE:

            if current_words:

                chunks.append(
                    " ".join(current_words)
                )

                current_words = []

            start = 0

            while start < len(words):

                end = min(
                    start + CHUNK_SIZE,
                    len(words)
                )

                chunk = " ".join(
                    words[start:end]
                )

                chunks.append(chunk)

                start += CHUNK_SIZE - OVERLAP

        else:

            # Would adding paragraph exceed target?
            if (
                len(current_words) + len(words)
                > CHUNK_SIZE
            ):

                if current_words:

                    chunks.append(
                        " ".join(current_words)
                    )

                # Create overlap
                overlap_words = current_words[
                    -OVERLAP:
                ]

                current_words = (
                    overlap_words + words
                )

            else:

                current_words.extend(words)

    if current_words:

        chunks.append(
            " ".join(current_words)
        )

    # Add metadata
    final_chunks = []

    for i, chunk in enumerate(chunks):

        final_chunks.append({
            "section": section_name,
            "text": chunk,
            "pages": pages,
            "word_count": len(chunk.split())
        })

    return final_chunks


# ---------------------------------------------------------
# Process one PDF
# ---------------------------------------------------------

def process_pdf(pdf_path):

    filename = os.path.basename(pdf_path)

    print(f"\nProcessing: {filename}")

    sections = extract_sections(pdf_path)

    all_chunks = []

    for section in sections:

        chunks = create_chunks(
            section["text"],
            section["section"],
            section["pages"]
        )

        all_chunks.extend(chunks)

    # Extract policy ID
    policy_id = None

    doc = fitz.open(pdf_path)

    full_text = ""

    for page in doc:
        full_text += page.get_text()

    doc.close()

    match = re.search(
        r"Policy ID:\s*([A-Z0-9\-]+)",
        full_text
    )

    if match:
        policy_id = match.group(1)

    # Add metadata
    for index, chunk in enumerate(all_chunks):

        chunk["chunk_id"] = (
            f"{policy_id or 'DOC'}_{index:03d}"
        )

        chunk["document"] = filename
        chunk["policy_id"] = policy_id

    return all_chunks


# ---------------------------------------------------------
# Process all PDFs
# ---------------------------------------------------------

def main():

    all_chunks = []

    pdf_files = sorted(
        file
        for file in os.listdir(PDF_FOLDER)
        if file.lower().endswith(".pdf")
    )

    print(
        f"Found {len(pdf_files)} PDF files."
    )

    for filename in pdf_files:

        path = os.path.join(
            PDF_FOLDER,
            filename
        )

        chunks = process_pdf(path)

        all_chunks.extend(chunks)

        print(
            f"Created {len(chunks)} chunks"
        )

    output_file = os.path.join(
        OUTPUT_FOLDER,
        "chunks.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 50)
    print("CHUNKING COMPLETE")
    print("=" * 50)

    print(
        f"Total chunks: {len(all_chunks)}"
    )

    print(
        f"Saved to: {output_file}"
    )


if __name__ == "__main__":
    main()