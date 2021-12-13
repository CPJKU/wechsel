from xml.etree import ElementTree
from tqdm.auto import tqdm
import regex as re
from pathlib import Path
import pandas as pd


def parse_language_page(language_page_path):
    language_regex = re.Regex(r"\[ISO\s639:([^\]]+?)\]\]\]\s\[\[([^\]]+?)\]\]")

    mw_text = open(language_page_path).read()
    languages = []

    for match in re.finditer(language_regex, mw_text):
        codes = match.group(1).split("|")
        assert len(codes) == 2 and codes[0] == codes[1]
        code = codes[0]

        langname = match.group(2).lower()
        if langname.endswith("language"):
            langname = langname[: -len("language")].strip()

        if len(langname) > 0 and len(code) > 0:
            languages.append((code, langname))

    return pd.DataFrame(languages, columns=["code", "name"])


def parse_wiktionary_dump(dump_path, languages, out_directory):
    def tag(name):
        return "{http://www.mediawiki.org/xml/export-0.10/}" + name

    heading_regex = re.Regex(r"^==([^=]+?)==$", re.MULTILINE)
    subheading_regex = re.Regex(r"^===([^=]+?)===$", re.MULTILINE)
    translation_regex = re.Regex(r"\[\[(.+?)\]\]")

    bidict_files = {}
    bar = tqdm()

    for _, node in ElementTree.iterparse(dump_path):
        if node.tag != tag("page"):
            continue

        title = node.find(tag("title")).text
        text = node.find(tag("revision")).find(tag("text")).text

        # skip meta pages, very crude but seems to work
        # and there are not many conceivable false positives
        if ":" in title:
            continue

        language_headings = list(heading_regex.finditer(text))

        for i, heading in enumerate(language_headings):
            lang = heading.group(1).strip()
            start = heading.span()[1]
            end = (
                language_headings[i + 1].span()[0]
                if i + 1 < len(language_headings)
                else len(text)
            )

            if lang.lower() not in languages:
                continue

            subheadings = list(subheading_regex.finditer(text[start:end]))

            for match in translation_regex.finditer(text[start:end]):
                closest_subheading = ""
                for subheading in subheadings[::-1]:
                    if subheading.span()[1] < match.span()[0]:
                        closest_subheading = subheading.group(1)
                        break

                if closest_subheading.startswith(
                    "Etymology"
                ) or closest_subheading.startswith("Pronunciation"):
                    continue

                word = match.group(1)

                if ":" in word or "|" in word:
                    continue

                if lang not in bidict_files:
                    filename = lang.replace("-", "_").lower() + ".txt"
                    filepath = out_directory / filename

                    bidict_files[lang] = open(filepath, "w")

                bidict_files[lang].write(f"{word}\t{title}\n")
                break

        bar.update(1)
        node.clear()


if __name__ == "__main__":
    out_dir = Path("data").expanduser().resolve()
    out_dir.mkdir(exist_ok=True, parents=True)

    languages = parse_language_page("languages.mw")
    languages.to_csv("languages.csv", index=False)

    language_set = set(languages["name"]) - {"english"}
    parse_wiktionary_dump(
        "enwiktionary-20211201-pages-articles.xml", language_set, out_dir
    )
