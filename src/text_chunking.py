"""
Split the cleaned EU AI Act into smaller chunks for RAG.

Input:
    data/processed/eu_ai_act_2026_clean.txt

Output:
    data/processed/eu_ai_act_chunks.json
"""

from pathlib import Path
import json
import re


# Input cleaned text
INPUT_PATH = Path("data/processed/eu_ai_act_2026_clean.txt")

# Output chunks
OUTPUT_PATH = Path("data/processed/eu_ai_act_chunks.json")

# ---------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------

MAX_CHARS = 5000
OVERLAP = 500


# Read the cleaned document
text = INPUT_PATH.read_text(encoding="utf-8")

print(f"Document characters: {len(text)}")

# Find Article headings
# Find possible Article headings
article_pattern = r"(?m)^Article\s+\d+[a-zA-Z]?\s*$"
candidate_matches = list(re.finditer(article_pattern, text))


# Validate the candidates
matches = []

for match in candidate_matches:

    # Look at the text immediately after "Article X"
    following_text = text[match.end():].lstrip()

    # Get the next line
    next_line = following_text.splitlines()[0].strip()

    # A citation such as:
    # Article 10
    # of Directive (EU) 2016/680
    # is NOT a real Article heading.
    if next_line.lower().startswith("of "):
        continue

    matches.append(match)


print(f"Article candidates found: {len(candidate_matches)}")
print(f"Valid Article headings: {len(matches)}")

print(f"Articles found: {len(matches)}")

# Show the first 10 Article headings
for match in matches[:10]:
    print(match.group())


# ---------------------------------------------------------
# Find all Annex headings
# ---------------------------------------------------------

annex_pattern = r"(?m)^ANNEX\s+[IVXLCDM]+\s*$"

annex_matches = list(re.finditer(annex_pattern, text))

print(f"\nAnnexes found: {len(annex_matches)}")

for match in annex_matches:
    print(match.group().strip())


# The first Annex marks the end of Article 113
if annex_matches:
    annex_start = annex_matches[0].start()
    print(f"\nANNEX I starts at character: {annex_start}")
else:
    annex_start = len(text)
    print("\nNo Annexes found")


# ---------------------------------------------------------
# Check the size of each Annex
# ---------------------------------------------------------

annex_sizes = []

for i, match in enumerate(annex_matches):

    start = match.start()

    # Next Annex marks the end of current Annex
    if i + 1 < len(annex_matches):
        end = annex_matches[i + 1].start()
    else:
        # ANNEX XIV continues until the end of the document
        end = len(text)

    annex_text = text[start:end].strip()

    annex_sizes.append({
        "annex": match.group().strip(),
        "characters": len(annex_text)
    })


print("\nAnnex sizes:")

for annex in annex_sizes:
    print(
        annex["annex"],
        "-",
        annex["characters"],
        "characters"
    )


# ---------------------------------------------------------
# Create one section for each Annex
# ---------------------------------------------------------

annexes = []

for i, match in enumerate(annex_matches):

    start = match.start()

    # Next Annex marks the end of current Annex
    if i + 1 < len(annex_matches):
        end = annex_matches[i + 1].start()
    else:
        # ANNEX XIV continues to the end of the document
        end = len(text)

    annex_text = text[start:end].strip()

    annex = {
        "annex": match.group().strip(),
        "text": annex_text,
        "characters": len(annex_text)
    }

    annexes.append(annex)


print(f"\nAnnex sections created: {len(annexes)}")


# Inspect the first Annex
print("\nFirst Annex:")
print(annexes[0]["annex"])
print("Characters:", annexes[0]["characters"])
print(annexes[0]["text"][:500])

# ---------------------------------------------------------
# Split Annexes into chunks
# ---------------------------------------------------------

annex_chunks = []

for annex in annexes:

    annex_name = annex["annex"]
    annex_text = annex["text"]

    # -----------------------------------------
    # Short Annex -> keep as one chunk
    # -----------------------------------------
    if len(annex_text) <= MAX_CHARS:

        annex_chunks.append({
            "annex": annex_name,
            "chunk_number": 1,
            "text": annex_text
        })

    # -----------------------------------------
    # Long Annex -> split into chunks
    # -----------------------------------------
    else:

        start = 0
        chunk_number = 1

        while start < len(annex_text):

            target_end = min(
                start + MAX_CHARS,
                len(annex_text)
            )

            # Find a safe line boundary
            if target_end < len(annex_text):

                safe_end = annex_text.rfind(
                    "\n",
                    start,
                    target_end
                )

                if safe_end > start:
                    end = safe_end
                else:
                    end = target_end

            else:
                end = target_end

            chunk_text = annex_text[start:end].strip()

            annex_chunks.append({
                "annex": annex_name,
                "chunk_number": chunk_number,
                "text": chunk_text
            })

            # Finished this Annex
            if end >= len(annex_text):
                break

            # Create approximately 500-character overlap
            next_start = max(start + 1, end - OVERLAP)

            # Move to a line boundary
            line_start = annex_text.find(
                "\n",
                next_start,
                end
            )

            if line_start != -1:
                start = line_start + 1
            else:
                start = next_start

            chunk_number += 1


print(f"\nAnnex chunks created: {len(annex_chunks)}")


# ---------------------------------------------------------
# Sanity check
# ---------------------------------------------------------

chunked_annexes = {
    chunk["annex"]
    for chunk in annex_chunks
}

missing_annexes = [
    annex["annex"]
    for annex in annexes
    if annex["annex"] not in chunked_annexes
]

print(f"Annexes represented in chunks: {len(chunked_annexes)}")
print(f"Missing Annexes: {missing_annexes}")


# Check the size of each Article
article_sizes = []

for i, match in enumerate(matches):

    start = match.start()

    # The next Article marks the end of the current Article
    if i + 1 < len(matches):
        end = matches[i + 1].start()
    else:
        end = annex_start  # Last Article goes up to ANNEX I

    article_text = text[start:end]

    article_sizes.append(len(article_text))


print(f"\nSmallest Article: {min(article_sizes)} characters")
print(f"Largest Article: {max(article_sizes)} characters")
print(f"Average Article: {sum(article_sizes) / len(article_sizes):.0f} characters")

print("\nLast 15 detected Article headings:")

for match in matches[-15:]:
    print(match.group())


print("\nAll detected Article headings:")

for i, match in enumerate(matches, start=1):
    print(i, match.group().strip())



print("\nArticle size distribution:")

print("Under 2,000:", sum(size < 2000 for size in article_sizes))
print("2,000–5,000:", sum(2000 <= size < 5000 for size in article_sizes))
print("5,000–10,000:", sum(5000 <= size < 10000 for size in article_sizes))
print("Over 10,000:", sum(size >= 10000 for size in article_sizes))


# ---------------------------------------------------------
# Create one section for each Article
# ---------------------------------------------------------

articles = []

for i, match in enumerate(matches):

    start = match.start()

    # End at the next Article
    if i + 1 < len(matches):
        end = matches[i + 1].start()
    else:
        # Article 113 ends when ANNEX I begins
        end = annex_start

    article_text = text[start:end].strip()

    article = {
        "article": match.group().strip(),
        "text": article_text,
        "characters": len(article_text)
    }

    articles.append(article)


print(f"\nArticle sections created: {len(articles)}")


# Look at the first Article
print("\nFirst Article:")
print(articles[0]["article"])
print("Characters:", articles[0]["characters"])
print(articles[0]["text"][:300])



# ---------------------------------------------------------
# Split Articles into final chunks
# ---------------------------------------------------------



chunks = []

for article in articles:

    article_name = article["article"]
    article_text = article["text"]

    # -----------------------------------------
    # Short Article -> keep as one chunk
    # -----------------------------------------
    if len(article_text) <= MAX_CHARS:

        chunks.append({
            "article": article_name,
            "chunk_number": 1,
            "text": article_text
        })

    # -----------------------------------------
    # Long Article -> split into chunks
    # -----------------------------------------
    else:

        start = 0
        chunk_number = 1

        while start < len(article_text):

            # Preferred maximum ending position
            target_end = min(
                start + MAX_CHARS,
                len(article_text)
            )

            # If more text remains, find a safe line break
            if target_end < len(article_text):

                safe_end = article_text.rfind(
                    "\n",
                    start,
                    target_end
                )

                if safe_end > start:
                    end = safe_end
                else:
                    end = target_end

            else:
                end = target_end

            # Create the chunk
            chunk_text = article_text[start:end].strip()

            chunks.append({
                "article": article_name,
                "chunk_number": chunk_number,
                "text": chunk_text
            })

            # Finished this Article
            if end >= len(article_text):
                break

            # -----------------------------------------
            # Create approximately 500-char overlap
            # -----------------------------------------

            next_start = max(start + 1, end - OVERLAP)

            # Move to a line boundary
            line_start = article_text.find(
                "\n",
                next_start,
                end
            )

            if line_start != -1:
                start = line_start + 1
            else:
                start = next_start

            chunk_number += 1


print(f"\nFinal chunks created: {len(chunks)}")


# ---------------------------------------------------------
# Sanity check
# ---------------------------------------------------------

chunked_articles = {
    chunk["article"]
    for chunk in chunks
}

missing_articles = [
    article["article"]
    for article in articles
    if article["article"] not in chunked_articles
]

print(f"Articles represented in chunks: {len(chunked_articles)}")
print(f"Missing Articles: {missing_articles}")


# Inspect first 5 chunks
print("\nFirst 5 chunks:")

for chunk in chunks[:5]:
    print("\n--------------------")
    print("Article:", chunk["article"])
    print("Chunk:", chunk["chunk_number"])
    print("Characters:", len(chunk["text"]))
    print(chunk["text"][:200])




print("\n--- Boundary check ---")

for chunk in chunks[:5]:

    if len(chunk["text"]) > 1000:

        print("\n", chunk["article"], "- Chunk", chunk["chunk_number"])

        print("\nSTART:")
        print(chunk["text"][:250])

        print("\nEND:")
        print(chunk["text"][-250:])




# ---------------------------------------------------------
# Combine Article and Annex chunks
# ---------------------------------------------------------

final_chunks = []


# Add Article chunks
for chunk in chunks:

    final_chunks.append({
        "section_type": "article",
        "section": chunk["article"],
        "chunk_number": chunk["chunk_number"],
        "text": chunk["text"]
    })


# Add Annex chunks
for chunk in annex_chunks:

    final_chunks.append({
        "section_type": "annex",
        "section": chunk["annex"],
        "chunk_number": chunk["chunk_number"],
        "text": chunk["text"]
    })


print(f"\nTotal final chunks: {len(final_chunks)}")


# ---------------------------------------------------------
# Save final chunks to JSON
# ---------------------------------------------------------

OUTPUT_PATH.write_text(
    json.dumps(
        final_chunks,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print(f"Chunks saved to: {OUTPUT_PATH}")