from .io_utils import read_json_from_file
from .sparql_utils import get_structural_properties

NO_AGGREGATION_PROPS = [
    "P1705", # native label -- label for the items in their official language or their original language
    "P2561", # name -- name the subject is known by; preferably use a more specific subproperty if available
    "P1448", # official name -- official name of the subject in its official language(s)
    "P1814", # name in kana -- the reading of a Japanese name in kana
    "P2441", # literal translation -- direct or word-for-word translation of a name or phrase (qualifier for name, title, inscription, and quotation properties)
    "P898", # IPA transcription -- transcription in the International Phonetic Alphabet
    "P2184", # history of topic -- item about the historical development of a subject's topic, example: "history of Argentina" for "Argentina". To list key events of the topic, use "significant event" (P793)
    "P2633", # geography of topic -- item that deals with the geography of the subject. Sample: "Rio de Janeiro" uses this property with value "geography of Rio de Janeiro" (Q10288853). For the location of a subject, use "location" (P276).
    "P9241", # demographics of topic -- item that deals with demographics of the subject
    "P8744", # economy of topic -- item that deals with the economy of the subject
    "P163", # flag -- subject's flag
    "P237", # coat of arms -- subject's coat of arms
    "P418", # has seal, badge, or sigil -- links to the item for the subject's seal
    "P1451", # motto text -- short motivation sentence associated to item
    "P361", # part of -- object of which the subject is a part (if this subject is already part of object A which is a part of object B, then please only make the subject part of object A), inverse property of "has part" (P527, see also "has parts of the class" (P2670))
    "P155", # follows -- immediately prior item in a series of which the subject is a part, preferably use as qualifier of P179 [if the subject has replaced the preceding item, e.g. political offices, use "replaces" (P1365)]
    "P156", # followed by -- immediately following item in a series of which the subject is a part, preferably use as qualifier of P179 [if the subject has been replaced, e.g. political offices, use "replaced by" (P1366)]
    "P1343", # described by source -- work where this item is described
    "P1889", # different from -- item that is different from another item, with which it may be confused
    "P856", # official website -- URL of the official page of an item (current or former). Usage: If a listed URL no longer points to the official website, do not remove it, but see the "Hijacked or dead websites" section of the Talk page
    "P973", # described at URL -- item is described at the following URL
    "P1581", #official blog URL -- URL to the blog of this person or organization
    "P487", # Unicode character -- Unicode character representing the item (only if this is not a control character or a compatiblity character: in that case, use only P4213)
    "P5949", # Unicode range -- set of Unicode code points covered by this element
    "P1456", # list of monuments -- link to the list of heritage monuments in the place/area
    "P4565", # electoral district number -- number of the constituency/electoral district established by law. Only to be used if the number is established by law, regulation, or other legally-binding decision; not to be used for database identifier numbers which lack legal force, even if contained in a government database.
]


INTERNAL_WIKI_PROPS = set()
INTERNAL_WIKI_PROPS.update(["P31"]) # instance of
INTERNAL_WIKI_PROPS.update(get_structural_properties("Q19847637")) # identifier
INTERNAL_WIKI_PROPS.update(get_structural_properties("Q51118821")) # wikimedia property
INTERNAL_WIKI_PROPS.update(get_structural_properties("Q18614948")) # authority control

prop_type_mapping = read_json_from_file("c3_dataset_augmentation/mahta_code/resources/prop_type_mapping.json")
prop_operation_mapping = read_json_from_file("c3_dataset_augmentation/mahta_code/resources/prop_type_mapping.json")
