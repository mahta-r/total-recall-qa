import bz2
import tqdm
import requests
import mwparserfromhell
import mwxml




def process_infobox_template(template, single_line=True):
    
    fields = [
            f"{param.name.strip()} = {param.value.strip()}"
            for param in template.params
        ]
    parsed_infobox = "\n--- Infobox ---\n" + "\n".join(fields) + "\n--- Infobox ---\n" + "\n"
    infobox_code = mwparserfromhell.parse(str(parsed_infobox))
    nested_parsed_infobox = infobox_code.strip_code(keep_template_params=True)
    return nested_parsed_infobox

    # ===============================================
    
    # fields = []
    # if single_line:
    #     for param in template.params:
    #         k = param.name.strip()
    #         v = " ".join(str(param.value).split())  # collapse whitespace/newlines
    #         fields.append(f"{k}={v}")
    #     return "Infobox: " + "; ".join(fields) + "\n"
    # else:
    #     fields = [
    #         f"{param.name.strip()} = {param.value.strip()}"
    #         for param in template.params
    #     ]
    #     return "--- Infobox ---\n" + "\n".join(fields) + " --- Infobox ---\n" + "\n"


def process_convert_template(template):
    params = list(template.params)
    if len(params) < 1:
        print("Convert template must have at least two parameters:", template)
        return ""
    # assert len(params) >= 2, "Convert template must have at least two parameters"
    elif len(params) == 1:
        value = params[0].value.strip()
        return f"{value}"
    else:
        value = params[0].value.strip()
        unit = params[1].value.strip()
        return f"{value} {unit}"


def replace_custom_templates(text):
    wikicode = mwparserfromhell.parse(text)
    templates = wikicode.filter_templates()
    
    for template in reversed(templates):   
        name = template.name.strip().lower()
        
        if name.startswith("infobox"):
            replacement = process_infobox_template(template, single_line=False)
            try:
                # replace the template node in wikicode tree, this only replaces the specific node
                wikicode.replace(template, replacement)
            except ValueError:
                try:
                    # replace the template string in wikicode tree, this replaces all matching occurrences
                    wikicode.replace(str(template), replacement)
                except:
                    # revert to removing infobox in next processing step
                    pass

        if name.startswith("convert"):
            replacement = process_convert_template(template)
            wikicode.replace(template, replacement)
    
    return wikicode


MW_API = "https://en.wikipedia.org/w/api.php"

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
    req = requests.get(MW_API, headers=headers, params=params)
    res = req.json()
    revision = res["query"]["pages"][0]["revisions"][0]
    text = revision["slots"]["main"]["content"]
    return text, mwparserfromhell.parse(text)


# page_title = "South Caribbean Coast Autonomous Region"
# page_title = "Fischer group Fi23"
# page_title = "Abraham Lincoln"
# page_title = "Andre Agassi"
page_title = "List of Nobel Memorial Prize laureates in Economic Sciences"


######## Original Wikicode ########
text, wikicode = wikicode_from_page_title(page_title)
print(wikicode)
print("================================================================================================================================")

######## Customized Wikicode ########
# wikicode = replace_custom_templates(text)
# for table in wikicode.filter_tags(matches=lambda t: "table" in t.tag.lower()):
#     wikicode.remove(table)
# print(wikicode)
# print("================================================================================================================================")

######## Parsed Text ########
# parsed_page = wikicode.strip_code(keep_template_params=True)
# print(parsed_page)
# print("================================================================================================================================")
