"""
Clean text extracted from the EU AI Act PDF.

Input:
    data/processed/eu_ai_act_2026.txt

Output:
    data/processed/eu_ai_act_2026_clean.txt
"""

from pathlib import Path
import re


# Input and output files
INPUT_PATH = Path("data/processed/eu_ai_act_2026.txt")
OUTPUT_PATH = Path("data/processed/eu_ai_act_2026_clean.txt")


# Read extracted text
text = INPUT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------
# 1. Remove invisible soft hyphens
# Example: trust­worthy -> trustworthy
# ---------------------------------------------------------
# If a soft hyphen occurs at a line break, join the word
# Example: trust­\nworthy -> trustworthy
text = text.replace("\u00ad\n", "")

# Remove any remaining soft hyphens
text = text.replace("\u00ad", "")


# ---------------------------------------------------------
# 2. Fix PDF word-splitting errors we observed
# ---------------------------------------------------------
corrections = {
    "leve l": "level",
    "t he": "the",
    "Un ion": "Union",
    "obligati ons": "obligations",
    "general-p urpose": "general-purpose",
    "enterpris es": "enterprises",

    # More errors found during our quality check
    "high- impact": "high-impact",
    "high- risk": "high-risk",
    "general- purpose": "general-purpose",
    "Recommendati on": "Recommendation",
    "201 3": "2013",
}


for wrong, correct in corrections.items():
    text = text.replace(wrong, correct)


# ---------------------------------------------------------
# 3. Remove unnecessary spaces before punctuation
# Example: "system ," -> "system,"
# ---------------------------------------------------------
text = re.sub(r"\s+([,.;:])", r"\1", text)


# ---------------------------------------------------------
# 4. Reduce excessive blank lines
# Keep paragraph structure, but avoid many empty lines
# ---------------------------------------------------------
text = re.sub(r"\n{3,}", "\n\n", text)


# ---------------------------------------------------------
# 5. Save cleaned text
# ---------------------------------------------------------
OUTPUT_PATH.write_text(text, encoding="utf-8")


print("Text cleaning completed.")
print(f"Saved to: {OUTPUT_PATH}")