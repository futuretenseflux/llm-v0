import unicodedata
import re
import html

def clean_scientific_text(text: str) -> str:
    if not text:
        return ""

    # 1. Recursive HTML Unescaping
    # Handles &amp;lt; -> &lt; -> <
    last_text = None
    while text != last_text:
        last_text = text
        text = html.unescape(text)

    # 2. Unicode Normalization (NFC)
    # NFC is safe for math; it preserves symbols like ², ³, and ℒ
    # whereas NFKC would turn ² into 2.
    text = unicodedata.normalize("NFC", text)

    # 3. Ligature Decomposition (Manual)
    # Decomposes common typographic ligatures that occur in scientific PDFs
    ligatures = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
        "ﬅ": "st", "ﬆ": "st"
    }
    for lig, replacement in ligatures.items():
        text = text.replace(lig, replacement)

    # 4. Standardize Whitespace (Non-Destructive)
    # Replace tabs with 4 spaces (standard for code)
    text = text.replace("\t", "    ")
    
    # Replace exotic Unicode spaces (non-breaking, thin, etc.) with ASCII space
    # Does NOT collapse multiple spaces (vital for code/math alignment)
    # Range covers: Non-breaking space, En/Em spaces, Thin spaces, etc.
    text = re.sub(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]", " ", text)

    # 5. Remove "Invisible" Poison Characters
    # Strips Zero-Width Space (U+200B), BOM (U+FEFF), etc.
    # These often break tokenization in code snippets.
    invisible_chars = re.compile(r"[\u200B-\u200D\uFEFF]")
    text = invisible_chars.sub("", text)

    # 6. Line Ending Normalization
    # Unify Windows (\r\n) and old Mac (\r) to Unix (\n)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 7. Final Polish
    # Strip trailing whitespace from each line (noise reduction)
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text

# --- Example Usage ---
raw_data = "Let x&sup2; &plusmn; y&sup2; = z&sup2;.   \nFind the solution in &reals;.\n\tprint(\"Done\")"
cleaned = clean_scientific_text(raw_data)
print(cleaned)
# Output:
# Let x² ± y² = z².
# Find the solution in ℝ.
#     print("Done")