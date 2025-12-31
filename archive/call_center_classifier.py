import json
import difflib
from collections import Counter

# Basic keyword lists
automatable_keywords = [
    "entering", "inputting", "recording", "logging", "verifying", "checking",
    "reviewing", "processing", "filing", "categorizing", "transcribing", "responding",
    "replying", "acknowledging", "scanning", "updating records", "validating data"
]

augmentable_keywords = [
    "solving", "analyzing", "interpreting", "investigating", "diagnosing", "training",
    "demonstrating", "explaining", "advising", "recommending", "identifying solutions",
    "determining cause", "troubleshooting"
]

irreplaceable_keywords = [
    "resolving conflict", "empathizing", "motivating", "building relationships",
    "negotiating", "calming", "de-escalating", "soothing", "adapting to emotions",
    "handling sensitive topics", "consoling", "counseling"
]

# Keywords for filtering relevant jobs
call_center_keywords = [
    "customer service", "call center", "user support", "help desk", "customer support"
]

def fuzzy_match(text, keywords, threshold=0.85):
    """Return True if any keyword approximately matches the text."""
    for kw in keywords:
        if kw in text:
            return True
        # Simple fuzzy matching using difflib
        ratio = difflib.SequenceMatcher(None, kw, text).ratio()
        if ratio >= threshold:
            return True
    return False

def classify_activity(activity):
    """Classify a single activity string."""
    text = activity.lower()

    if fuzzy_match(text, automatable_keywords):
        return 'automatable'
    if fuzzy_match(text, augmentable_keywords):
        return 'augmentable'
    if fuzzy_match(text, irreplaceable_keywords):
        return 'irreplaceable'
    return 'unclassified'


def main():
    with open('All_Industries_detailed.json', 'r') as f:
        data = json.load(f)

    results = {}
    for job in data:
        raw_title = job.get('job_title', '')
        # Clean title by removing newlines and extra spaces
        title = " ".join(raw_title.split())
        lower_title = title.lower()
        if not any(kw in lower_title for kw in call_center_keywords):
            continue

        activities = job.get('work_activities', [])
        if not activities:
            continue

        counts = Counter()
        for act in activities:
            label = classify_activity(act)
            counts[label] += 1

        total = sum(counts.values())
        if total == 0:
            continue

        percentages = {k: round(counts[k] / total * 100, 2) for k in ['automatable', 'augmentable', 'irreplaceable']}
        results[title] = percentages

        print(f"\nJob Title: {title}")
        print(f"Total activities: {total}")
        for category, pct in percentages.items():
            print(f"  {category.capitalize()}: {pct}%")
        if counts.get('unclassified'):
            pct_unclassified = round(counts['unclassified'] / total * 100, 2)
            print(f"  Unclassified: {pct_unclassified}%")

    print("\nSummary dictionary:\n")
    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    main()
