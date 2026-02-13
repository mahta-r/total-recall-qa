from wikidata.sparql_utils import get_structural_properties
from io_utils import read_json_from_file


UNUSABLE_PROPS = [
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
    "P131", # located in the administrative territorial entity -- the administrative territorial entity in which this item is located
    "P150", # contains the administrative territorial entity -- (list of) direct subdivisions of an administrative territorial entity
    "P279", # subclass of -- relation of type constraint
    "P8138", # located in the statistical territorial entity -- statistical territorial entity in which a place is located or is part of
    "P463", # member of -- organization, club or musical group to which the subject belongs
    "P1830", # owner of -- entities owned by the subject
    "P1344", # participant in -- event in which a person, organization or creative work was/is a participant
    # "P530", # diplomatic relation -- diplomatic relations of the country
    "P2341", # indigenous to -- place or ethnic group where a language, art genre, cultural tradition or expression, cooking style or food, or biological species or variety is found (or was originally found)
    # "P832", # public holiday -- official public holiday that occurs in this place in its honor, usually a non-working day
    "P485", # archives at -- the institution holding the subject's archives
    "P1313", # office held by head of government -- political office that is fulfilled by the head of the government of this item
    "P8402", # open data portal -- platform with publicly accessible data of an organization
    "P461", # opposite of -- item that is in some way the opposite of this item
    "P8956", # compatible with -- this work, product, object or standard can interact with another work, product, object or standard
    "P1552", # has characteristic -- inherent or distinguishing quality or feature of the entity. Use a more specific property when possible
    "P2852", # emergency phone number -- telephone number to contact emergency services
    "P527", # has part(s) -- part of this subject; inverse property of "part of" (P361). See also "has parts of the class" (P2670).
    "P421", # located in time zone -- time zone for this item
    "P1365", # replaces -- person, state or item replaced. Use "structure replaces" (P1398) for structures. Use "follows" (P155) if the previous item was not replaced or predecessor and successor are identical
    "P1376", # capital of -- country, state, department, canton or other administrative division of which the municipality is the governmental seat
    "P706", # located in/on physical feature -- located on the specified (geo)physical feature. Should not be used when the value is only political/administrative (P131) or a mountain range (P4552). Use P206 for things in/on bodies of water. 
    # "P138", # named after -- entity or event that inspired the subject's name, or namesake (in at least one language). Qualifier "applies to name" (P5168) can be used to indicate which one
]

NO_AGGREGATION_PROPS = [
    "P1383", # contains settlement -- settlement which an administrative division contains
    "P17", # country -- sovereign state that this item is in (not to be used for human beings)
    "P36", # capital -- seat of government of a country, province, state or other type of administrative territorial entity
    "P276", # location -- location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object
    "P30", # continent -- continent of which the subject is a part
]

NO_CONNECTING_PROPS = [
    "P47", # shares border with -- countries or administrative subdivisions, of equal level, that this item borders, either by land or water. A single common point is enough.
]


POINT_IN_TIME_QUALIFIER = 'P585'  # https://www.wikidata.org/wiki/Property:P585


INTERNAL_WIKI_PROPS = set()
INTERNAL_WIKI_PROPS.update(["P31"]) # instance of
INTERNAL_WIKI_PROPS.update(get_structural_properties("Q19847637")) # identifier
INTERNAL_WIKI_PROPS.update(get_structural_properties("Q51118821")) # wikimedia property
INTERNAL_WIKI_PROPS.update(get_structural_properties("Q18614948")) # authority control


VALID_PROP_DATATYPES = [
    "Quantity",
    "GlobeCoordinate",
    "Time",
    "WikibaseItem"
]

