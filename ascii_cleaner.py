import json
import unicodedata
import re

def normalize_unicode(text):
    """
    Normalize Unicode characters to their closest ASCII representation.
    This uses NFKD normalization and then re-encodes as ASCII (ignoring errors).
    """
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def escape_problematic_chars(text):
    """
    Optionally, replace problematic Unicode punctuation with ASCII approximations.
    You can add more replacements as needed.
    """
    replacements = {
        "–": "-",     # en dash
        "—": "-",     # em dash
        "’": "'",     # right single quotation mark
        "‘": "'",     # left single quotation mark
        "“": '"',     # left double quotation mark
        "”": '"',     # right double quotation mark
        "é": "e",
        "≈": "~",
        "α": "alpha",
        "ε": "epsilon",
        "°": "deg",
        "²": "2",
        "→": "->",
        "×": "x",
        "Σ": "Sigma",
        "⊆": "subset",
        "∈": "in",
        "≥": ">=",
        "ç": "c",
        "Φ": "Phi",
        "β": "beta",
        "γ": "gamma",
        "∞": "infinity",
        "Δ": "Delta",
        "π": "pi",
        "Ü": "Ue",
        "ü": "ue",
        "∝": "propto",
        "≅": "cong",
        "∇": "nabla",
        "θ": "theta",
    }
    for uni, ascii_rep in replacements.items():
        text = text.replace(uni, ascii_rep)
    return text

def clean_line(line):
    """
    Try cleaning a JSONL line:
      1. Replace problematic Unicode characters.
      2. Normalize Unicode.
      3. Remove any stray control characters.
    """
    # Replace known problematic characters
    cleaned = escape_problematic_chars(line)
    # Normalize Unicode to decompose accented characters
    cleaned = unicodedata.normalize('NFKC', cleaned)
    # Optionally, remove non-printable characters (except standard whitespace)
    cleaned = re.sub(r'[^\x20-\x7E\n\r]', '', cleaned)
    return cleaned

def fix_and_reorder_json_line(line):
    """
    Try to load the JSON, clean it if necessary, and ensure 'instruction', 'context', 'response' keys.
    Returns a valid JSON string (with ensure_ascii=True) or None if it cannot be fixed.
    """
    try:
        data = json.loads(line.strip())
        fixed_data = {
            "instruction": data.get("instruction", ""),
            "context": data.get("context", ""),
            "response": data.get("response", "")
        }
        return json.dumps(fixed_data, ensure_ascii=True)
    except json.JSONDecodeError as e:
        print(f"Initial JSON error: {e}\nLine: {line.strip()}")
        cleaned_line = clean_line(line)
        try:
            data = json.loads(cleaned_line.strip())
            fixed_data = {
                "instruction": data.get("instruction", ""),
                "context": data.get("context", ""),
                "response": data.get("response", "")
            }
            return json.dumps(fixed_data, ensure_ascii=True)
        except json.JSONDecodeError as e2:
            print(f"Could not fix line after cleaning: {e2}\nCleaned line: {cleaned_line.strip()}")
            return None

def process_jsonl_file(input_filename, output_filename):
    fixed_lines = []
    with open(input_filename, "r", encoding="utf-8") as infile:
        for i, line in enumerate(infile, start=1):
            fixed_json = fix_and_reorder_json_line(line)
            if fixed_json is not None:
                fixed_lines.append(fixed_json)
            else:
                print(f"Skipping line {i} due to errors.")
    with open(output_filename, "w", encoding="utf-8") as outfile:
        for fixed_line in fixed_lines:
            outfile.write(fixed_line + "\n")
    print(f"Processing complete. Fixed {len(fixed_lines)} lines out of {i} total.")

# Example usage:
if __name__ == "__main__":
    input_file = "puolsky.jsonl"  # Replace with your input JSONL file
    output_file = "puolsky_clean.jsonl" # Replace with your desired output file
    process_jsonl_file(input_file, output_file)
