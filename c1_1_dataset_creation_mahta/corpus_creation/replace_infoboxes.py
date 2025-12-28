import argparse
import bz2
import tqdm
import requests
import mwparserfromhell
import mwxml


MW_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TotalRecallRAG/0.1 (contact: email@example.edu)"


def wikicode_from_page_title(title):
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "rvlimit": 1,
        "titles": title,
        "format": "json",
        "formatversion": "2",
    }
    headers = {"User-Agent": USER_AGENT}
    req = requests.get(MW_API, headers=headers, params=params)
    res = req.json()
    revision = res["query"]["pages"][0]["revisions"][0]
    text = revision["slots"]["main"]["content"]
    return mwparserfromhell.parse(text)


def main(args): 
    with bz2.open(args.wikipedia_dump_path, "rb") as f:
        dump = mwxml.Dump.from_file(f)

        for idx, page in tqdm.tqdm(enumerate(dump), total=25000000):
            
            revision = next(iter(page)) # only need latest revision
            text = revision.text

            wikicode = mwparserfromhell.parse(text)
            templates = wikicode.filter_templates()
            # Need to process infoboxes in reversed order to handle nested templates
            for template in reversed(templates):
                
                name = template.name.strip().lower()
                if name.startswith("infobox"):
                    fields = [
                        f"{param.name.strip()}: {param.value.strip()}"
                        for param in template.params
                    ]
                    try:
                        # replace the template node in wikicode tree, this only replaces the specific node
                        wikicode.replace(template, "\n".join(fields) + "\n")
                    except ValueError:
                        try:
                            # replace the template string in wikicode tree, this replaces all matching occurrences
                            wikicode.replace(str(template), "\n".join(fields) + "\n")
                        except:
                            # revert to removing infobox in next processing step
                            continue
                
                if name.startswith("convert"):
                    params = list(template.params)
                    assert len(params) >= 2, "Convert template must have at least two parameters"

                    value = params[0].value.strip()
                    unit = params[1].value.strip()

                    replacement = f"{value} {unit}"
                    wikicode.replace(template, replacement)
                    
            
            parsed = wikicode.strip_code(keep_template_params=True)
            # mwparserfromhell is only used for infobox formating as WikiExtractor V2 does not handle infoboxes
            # the actual parsing will be done by WikiExtractor V2 once that is set up properly



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wikipedia_dump_path", 
        type=str, 
        required=True, 
        help="Path to Wikipedia dump in bz2 format"
    )

    args = parser.parse_args()

    main(args)