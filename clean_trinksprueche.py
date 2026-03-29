import json
import re
from difflib import SequenceMatcher
from pathlib import Path

BASE = Path("data")
RAW_FILES = sorted([p for p in BASE.glob("raw*.txt") if p.is_file()])

NOISE_PATTERNS = [
    r"^upvoten$",
    r"^downvoten$",
    r"^auszeichnen$",
    r"^teilen$",
    r"^anmelden$",
    r"^anzeige$",
    r"^marktplatz$",
    r"^gassengeschwaetz$",
    r"^rpg foren$",
    r"^zum seitenanfang springen$",
    r"^zum hauptinhalt springen$",
    r"^springe zu der stelle",
    r"^beitrag von ",
    r"^aw:",
    r"^vor \d+ jahren$",
    r"^\d+[,.]?\d*\s*t\.$",
    r"^\d+$",
    r"^benutzer$",
    r"^aufrufe$",
    r"^trinksprueche.*$",
    r"^lustige trinksprueche.*$",
    r"^schoene trinksprueche.*$",
    r"^top\s*\d+.*$",
    r"^studyflix vernetzt.*$",
    r"^hier finden sie trinksprueche.*$",
    r"^die \d+ besten trinksprueche.*$",
    r"^was ist ein trinkspruch\?$",
    r"^allgemein$",
    r"^formen$",
    r"^als rede$",
    r"^der lyrische trinkspruch$",
    r"^sammlung von trinkspruechen$",
    r"^phila?\d+$",
]

NOISE_REGEX = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]

DRINK_HINT_STEMS = {
    "alkohol",
    "bier",
    "wein",
    "schnaps",
    "promille",
    "prost",
    "prosit",
    "sauf",
    "saeuft",
    "saeuft",
    "trink",
    "trinke",
    "trinkt",
    "trinken",
    "becher",
    "glas",
    "durst",
    "hopfen",
    "malz",
    "vollrausch",
    "wirt",
    "korn",
    "pils",
    "rausch",
    "stoe",
}

TOAST_HINTS = {
    "zum wohl",
    "auf uns",
    "wohlsein",
    "stoesschen",
    "proesterchen",
    "moegen unsere kinder",
    "auf die liebe und das leben",
    "nicht lang schnacken",
}

DIALECT_WORDS = {"koa", "net", "kopp", "ma", "des"}

REJECT_SUBSTRINGS = {
    "willst du mehr lesen",
    "durchstoebere andere themen",
    "das bild zeigt",
    "veranschaulicht",
    "interkulturelle",
    "hier finden sie",
    "bearbeitet vor",
    "sammlung von trinkspruechen",
    "aufrufe",
    "benutzer",
    "zum hauptinhalt",
    "zum seitenanfang",
    "beitrag von",
    "die meisten internationalen trinksprueche",
    "bist du im ausland unterwegs",
    "perfekte trinkspruch",
    "anweisungen",
    "entsprechenden anweisungen",
    "kein wirklicher spruch",
    "dabei handelt es sich",
    "der wohl bekannteste",
    "je verrueckter der trinkspruch",
    "ein guter trinkspruch ist",
    "ein trinkspruch ist ein kurzer",
    "kurze knackige sprueche fuer glaeser gravuren",
    "freundschaft zusammenhalt",
    "feierliche anlaesse",
    "intelligent ironisch",
    "alte weisheiten vintage",
    "zugeschrieben",
    "aus bayern",
    "volksmund",
    "martin luther",
    "claude tillier",
    "christoph braekling",
    "genau mein humor",
    "das bild zeigt",
    "op uw gezondheid",
    "ich sah regenboegen in aller pracht",
    "ich trinke auf euren",
}

ENGLISH_WORDS = {
    "the",
    "and",
    "with",
    "for",
    "your",
    "ever",
    "heart",
    "full",
    "light",
    "nights",
    "forget",
    "well",
    "never",
    "never",
    "remember",
    "friends",
    "cheers",
    "here",
}

GERMAN_HINT_WORDS = {
    "und",
    "der",
    "die",
    "das",
    "auf",
    "ein",
    "eine",
    "ich",
    "du",
    "wir",
    "ist",
    "nicht",
    "zum",
    "wohl",
    "prost",
}


def ascii_fold(value: str) -> str:
    folded = value.lower()
    folded = (
        folded.replace("ae", "ae")
        .replace("oe", "oe")
        .replace("ue", "ue")
        .replace("ss", "ss")
    )
    folded = (
        folded.replace("a", "a")
        .replace("o", "o")
        .replace("u", "u")
        .replace("s", "s")
    )
    folded = (
        folded.replace("\u00e4", "ae")
        .replace("\u00f6", "oe")
        .replace("\u00fc", "ue")
        .replace("\u00df", "ss")
        .replace("\u2019", "'")
    )
    return folded


def repair_mojibake(text: str) -> str:
    try:
        repaired = text.encode("latin1", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        return text

    old_bad = text.count("\u00c3") + text.count("\u00e2\u20ac") + text.count("\ufffd")
    new_bad = repaired.count("\u00c3") + repaired.count("\u00e2\u20ac") + repaired.count("\ufffd")
    if new_bad < old_bad:
        return repaired
    return text


def normalize_document(text: str) -> str:
    text = repair_mojibake(text)
    replacements = {
        "\u201e": '"',
        "\u201c": '"',
        "\u201d": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u2032": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u2026": "...",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def normalize_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[\-\*\u2022\s]+", "", line)
    line = line.replace("\u2060", "")
    line = re.sub(r"\[(gel\u00f6scht|deleted)\]", "", line, flags=re.IGNORECASE)
    line = re.sub(
        r"\([^)]*(latein|\u00fcbersetzung|video|mein bruder|z\\.b\\.)[^)]*\)",
        "",
        line,
        flags=re.IGNORECASE,
    )

    # Remove symbols/emojis while keeping normal punctuation and letters.
    line = re.sub(r"[^\w\s\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df.,;:!?\"'\-/]", " ", line)

    line = re.sub(r"\s+", " ", line).strip()
    line = line.strip("\"' ")
    line = re.sub(r"\s+([.,;:!?])", r"\1", line)
    line = re.sub(r"([.,;:!?]){2,}", r"\1", line)

    # Minimal, deterministic normalization to Hochdeutsch when obvious.
    replacements = {
        " mal ne ": " mal eine ",
        " ne ": " eine ",
        " Nich lang schnacken": " Nicht lang schnacken",
        " Kopp in den Nacken": " Kopf in den Nacken",
        " Kopp in Nacken": " Kopf in den Nacken",
        " heut\u2019": " heute",
        " heut'": " heute",
    }
    line_padded = f" {line} "
    for src, dst in replacements.items():
        line_padded = line_padded.replace(src, dst)
    line = line_padded.strip()
    line = re.sub(r"\bsauf' ich\b", "sauf ich", line, flags=re.IGNORECASE)

    # Remove category prefixes copied from list pages.
    line = re.sub(
        r"^(?:\d+\s+)?(?:[A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df\-]+\s+){0,8}"
        r"Trinkspr\u00fcche(?:\s+[A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df\-]+){0,8}\s*[:\-]?\s*",
        "",
        line,
        flags=re.IGNORECASE,
    )

    # Remove leading language labels like "Englisch:" or "Griechenland:".
    line = re.sub(
        r"^[A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df]{3,20}:\s*",
        "",
        line,
    )

    return line


def normalized_for_matching(text: str) -> str:
    text = ascii_fold(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenized(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalized_for_matching(text))


def has_drink_hint(text: str) -> bool:
    for tok in tokenized(text):
        if any(tok.startswith(stem) for stem in DRINK_HINT_STEMS):
            return True
    return False


def is_noise(line: str) -> bool:
    if not line:
        return True

    low = normalized_for_matching(line)

    if any(rx.search(low) for rx in NOISE_REGEX):
        return True

    if re.match(r"^[a-z0-9_\-]{2,20}$", low) and " " not in low:
        if not has_drink_hint(low):
            return True

    if re.match(r"^(mai|aug|jan|feb|maerz|apr|jun|jul|sep|okt|nov|dez)\.?\s+\d{4}$", low):
        return True

    return False


def looks_like_toast(text: str) -> bool:
    if len(text.split()) < 3:
        return False

    low = normalized_for_matching(text)
    if has_drink_hint(low):
        return True
    if any(h in low for h in TOAST_HINTS):
        return True

    # Common toast forms without explicit alcohol word.
    if low.startswith("moegen ") and text.endswith("!"):
        return True

    return False


def is_rejected_content(text: str) -> bool:
    low = normalized_for_matching(text)

    if text.lstrip().startswith(","):
        return True

    if "trinkspruech" in low or "trinkspruch" in low:
        return True

    if any(snippet in low for snippet in REJECT_SUBSTRINGS):
        return True

    # Discard obvious fragments.
    if text.endswith((",", ":", "-")):
        return True

    words = low.split()
    english_hits = sum(1 for w in words if w in ENGLISH_WORDS)
    german_hits = sum(1 for w in words if w in GERMAN_HINT_WORDS)
    if english_hits >= 2 and german_hits <= 1:
        return True

    # Remove long unfinished fragments.
    if not text.endswith((".", "!", "?")) and len(words) > 7:
        return True

    return False


def should_append(previous: str, current: str) -> bool:
    if not previous:
        return False
    if previous.endswith((".", "!", "?")):
        return False
    if current and current[0].islower():
        return True
    if len(previous.split()) < 8:
        return True
    return False


def split_combined_entry(entry: str) -> list[str]:
    entry = entry.strip()
    if not entry:
        return []

    # Keep short poetic multiline content intact; split only hard separators.
    parts = re.split(r"\s*/\s*", entry)
    results = []
    for part in parts:
        subparts = re.split(r"\s+-\s+(?=[A-Z\u00c4\u00d6\u00dc])", part)
        for sp in subparts:
            cleaned = normalize_line(sp)
            if cleaned:
                results.append(cleaned)
    return results


def pick_better(a: str, b: str) -> str:
    def score(value: str) -> tuple[int, int, int, int]:
        words = normalized_for_matching(value).split()
        dialect_hits = sum(1 for w in words if w in DIALECT_WORDS)
        return (
            int(value.endswith((".", "!", "?"))),
            -dialect_hits,
            int(4 <= len(words) <= 30),
            len(value),
        )

    return max((a, b), key=score)


def semantic_duplicate(a: str, b: str) -> bool:
    ka = normalized_for_matching(a)
    kb = normalized_for_matching(b)
    if ka == kb:
        return True

    ratio = SequenceMatcher(None, ka, kb).ratio()
    if ratio >= 0.93:
        return True

    sa = set(ka.split())
    sb = set(kb.split())
    if not sa or not sb:
        return False
    jaccard = len(sa & sb) / len(sa | sb)
    return jaccard >= 0.9


def extract_candidates(file_path: Path) -> list[dict[str, str]]:
    raw = normalize_document(file_path.read_text(encoding="utf-8", errors="replace"))
    rows = raw.splitlines()

    candidates: list[dict[str, str]] = []

    # Quoted snippets are often the actual toast text on scraped pages.
    for m in re.finditer(r'"([^"\n]{6,240})"', raw):
        original = m.group(1).strip()
        cleaned = normalize_line(original)
        if not is_noise(cleaned) and looks_like_toast(cleaned):
            candidates.append({"original": original, "cleaned": cleaned})

    pending = ""
    for raw_line in rows:
        line = normalize_line(raw_line)

        if not line or is_noise(line):
            if pending:
                for part in split_combined_entry(pending):
                    if looks_like_toast(part):
                        candidates.append({"original": pending, "cleaned": part})
                pending = ""
            continue

        # Remove very long explanatory prose unless it clearly is a toast.
        if len(line.split()) > 30 and not has_drink_hint(line):
            if pending:
                for part in split_combined_entry(pending):
                    if looks_like_toast(part):
                        candidates.append({"original": pending, "cleaned": part})
                pending = ""
            continue

        if pending and should_append(pending, line):
            pending = f"{pending} {line}".strip()
        else:
            if pending:
                for part in split_combined_entry(pending):
                    if looks_like_toast(part):
                        candidates.append({"original": pending, "cleaned": part})
            pending = line

    if pending:
        for part in split_combined_entry(pending):
            if looks_like_toast(part):
                candidates.append({"original": pending, "cleaned": part})

    return candidates


def main() -> None:
    all_candidates: list[dict[str, str]] = []
    for file_path in RAW_FILES:
        all_candidates.extend(extract_candidates(file_path))

    # Hard quality filters.
    filtered: list[dict[str, str]] = []
    for item in all_candidates:
        cleaned = normalize_line(item["cleaned"])
        if len(cleaned.split()) < 3:
            continue
        if is_rejected_content(cleaned):
            continue
        if is_noise(cleaned):
            continue
        if not looks_like_toast(cleaned):
            continue
        filtered.append({"original": item["original"], "cleaned": cleaned})

    # Remove exact cleaned duplicates while preserving one original sample.
    by_clean: dict[str, dict[str, str]] = {}
    for item in filtered:
        key = item["cleaned"]
        if key not in by_clean:
            by_clean[key] = item
        else:
            better = pick_better(by_clean[key]["cleaned"], item["cleaned"])
            if better == item["cleaned"]:
                by_clean[key] = item

    mapping = sorted(by_clean.values(), key=lambda x: x["cleaned"].lower())

    # Semantic dedup.
    final: list[str] = []
    for item in mapping:
        text = item["cleaned"]
        dup_idx = None
        for i, existing in enumerate(final):
            if semantic_duplicate(text, existing):
                dup_idx = i
                break
        if dup_idx is None:
            final.append(text)
        else:
            final[dup_idx] = pick_better(final[dup_idx], text)

    final_sorted = sorted(set(final), key=lambda s: s.lower())

    (BASE / "cleaned_with_original.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (BASE / "cleaned_deduplicated.json").write_text(
        json.dumps(final_sorted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"files={len(RAW_FILES)}")
    print(f"candidates={len(all_candidates)}")
    print(f"filtered={len(filtered)}")
    print(f"unique_clean={len(mapping)}")
    print(f"final={len(final_sorted)}")
    print("written=data/cleaned_with_original.json")
    print("written=data/cleaned_deduplicated.json")


if __name__ == "__main__":
    main()
