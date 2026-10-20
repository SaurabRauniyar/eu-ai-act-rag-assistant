import pymupdf


PDF_PATH = "data/raw/eu_ai_act_2026.pdf"

document = pymupdf.open(PDF_PATH)

# Page 2 (Python starts counting from 0)
page = document[1]

text = page.get_text()

print("===== PYMUPDF EXTRACTION =====")
print(text[:2500])