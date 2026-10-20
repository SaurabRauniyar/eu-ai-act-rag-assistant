"""
Extract text from the EU AI Act PDF using PyMuPDF.

Input:
    data/raw/eu_ai_act_2026.pdf

Output:
    data/processed/eu_ai_act_2026.txt
"""

from pathlib import Path
import pymupdf


PDF_PATH = Path("data/raw/eu_ai_act_2026.pdf")
OUTPUT_PATH = Path("data/processed/eu_ai_act_2026.txt")


# Open PDF
document = pymupdf.open(PDF_PATH)

print(f"Total pages: {len(document)}")

all_text = []


# Extract one page at a time
for page_number, page in enumerate(document, start=1):

    text = page.get_text()

    if text:
        # Keep page number for future RAG citations
        page_text = f"\n--- PAGE {page_number} ---\n{text}"
        all_text.append(page_text)

        print(f"Page {page_number}: {len(text)} characters")

    else:
        print(f"Page {page_number}: No text found")


final_text = "\n".join(all_text)

OUTPUT_PATH.write_text(final_text, encoding="utf-8")

print("\nExtraction completed.")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Total characters: {len(final_text)}")