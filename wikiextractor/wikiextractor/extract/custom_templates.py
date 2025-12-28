import mwparserfromhell



def process_infobox_template(template, single_line=True):

    fields = []
    for param in template.params:
        k = param.name.strip()
        if param.value.strip().startswith('--- Infobox:Start ---'):
            v = '\n' + param.value.strip()  # preserve newlines for nested infoboxes
        else:
            # v = param.value.strip()
            v = " ".join(str(param.value).split())  # collapse whitespace/newlines
        fields.append(f"{k}={v}")
    
    parsed_infobox = "\n--- Infobox:Start ---\n" + "\n".join(fields) + "\n--- Infobox:End ---\n" + "\n"
    infobox_code = mwparserfromhell.parse(str(parsed_infobox))
    nested_parsed_infobox = infobox_code.strip_code(keep_template_params=True)
    return nested_parsed_infobox

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
        # print("Convert template must have at least two parameters:", template)
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

    # for table in wikicode.filter_tags(matches=lambda t: "table" in t.tag.lower()):
    #     wikicode.remove(table)
    
    return str(wikicode)