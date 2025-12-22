import argparse
import bz2
import tqdm
import requests
import mwparserfromhell
import mwxml


API_URL = "https://en.wikipedia.org/w/api.php"


def get_category_members(category):
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": "max",
        "format": "json"
    }
    while True:
        result = requests.get(API_URL, params=params).json()
        for item in result["query"]["categorymembers"]:
            if item["ns"] == 0:  # 0 = article namespace
                yield item
            # Recurse into subcategories
            if item["ns"] == 14:  # 14 = category namespace
                subcat = "Category:" + item["title"].split("Category:")[1]
                yield from get_category_members(subcat)
        if "continue" not in result:
            break
        params.update(result["continue"])


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
    headers = {"User-Agent": "My-Bot-Name/1.0"}
    req = requests.get(API_URL, headers=headers, params=params)
    res = req.json()
    revision = res["query"]["pages"][0]["revisions"][0]
    text = revision["slots"]["main"]["content"]
    return mwparserfromhell.parse(text)


def main(args): 
    with bz2.open(args.wikipedia_dump_path, "rb") as f:
        dump = mwxml.Dump.from_file(f)

        for idx, page in tqdm.tqdm(enumerate(dump)):
            
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
                            # debugging log
                            # with open("parse_error.log", "a") as err_file:
                            #     print(template, file=err_file)
                            #     print('---------------------------------------- PARSED ----------------------------------------', file=err_file)
                            #     print(wikicode, file=err_file)
                            #     print('---------------------------------------- ORIGINAL ----------------------------------------', file=err_file)
                            #     print(mwparserfromhell.parse(text), file=err_file)
                            #     raise
            
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