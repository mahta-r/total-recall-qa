from collections import defaultdict

OPERATIONS = "OPERATIONS"
PROPERTIES = "PROPERTIES"


COMPATIBLE_PROP_OPS = [
  # ------------------------------------------------- ADDITIVE TOTALS & COUNTS -------------------------------------------------
  { 
    OPERATIONS: [
      "SUM", "SUM_TOP_K", "SUM_BOTTOM_K", "AVG", "AVG_TOP_K", "AVG_BOTTOM_K", "MEDIAN", "MIN", "MAX", "DIFFERENCE(MAX−MIN)", "RATIO(MAX/MIN)", "COUNT_LT_X", "COUNT_GT_X" 
    ],
    PROPERTIES: [
      # ----------------- Demographic Counts -----------------
      "P1082", # P1082 -- population -- number of people inhabiting the place; number of people of subject
      "P6343", # P6343 -- urban population -- number of people living within the territorial entity who live in its urban parts
      "P6344", # P6344 -- rural population -- number of people living within the territorial entity who live in its rural parts
      "P1539", # P1539 -- female population -- number of female people inhabiting the place; number of female people of subject
      "P1540", # P1540 -- male population -- number of male people inhabiting the place; number of male people of subject
      "P2573", # P2573 -- number of out-of-school children -- number of out-of-school children reported for a place
      "P1538", # P1538 -- number of households -- total number of households in this place, includes dwellings of all types
      "P4080", # P4080 -- number of houses -- number of houses in given territorial entity
      "P1831", # P1831 -- electorate -- number of persons qualified to vote during elections
      "P1128", # P1128 -- employees -- total number of employees of a company at a given "point in time" (P585). Most recent data would generally have preferred rank; data for previous years normal rank (not deprecated rank). Add data for recent years, don't overwrite
      "P1342", # P1342 -- number of seats -- total number of seats/members in an assembly (legislative house or similar)
      "P5982", # P5982 -- annual number of weddings -- number of marriages per year in a location
      
      # ----------------- Economic & Financial Totals ----------------- 
      "P2131", # P2131 -- nominal GDP -- market value of all officially recognized final goods and services produced within a country in a given period of time
      "P2403", # P2403 -- total assets -- value of assets held by a private or public entity
      "P2138", # P2138 -- total liabilities -- sum of all present obligations of the entity to transfer economic resources as a result of past events
      "P2402", # P2402 -- total expenditure -- amount of spending by this public or private entity, not to be confused with fiscal expenditure
      "P3087", # P3087 -- fiscal/tax revenue -- fiscal revenue of a public entity (not for private entities)
      
      # ----------------- Geographic & Environmental Totals ----------------- 
      "P2046", # P2046 -- area -- area occupied by an object
      "P5991", # P5991 -- carbon footprint -- total emissions caused by an individual, event, organisation, or product, expressed as carbon dioxide equivalent; has to be calculated via a scientificly proven methodology 
    ]
  },
  # ------------------------------------------------- RATES & INDICES & NORMALIZED MEASURES -------------------------------------------------
  {
    OPERATIONS: [
      "AVG", "AVG_TOP_K", "AVG_BOTTOM_K", "MEDIAN", "MIN", "MAX", "DIFFERENCE(MAX−MIN)", "RATIO(MAX/MIN)", "COUNT_LT_X", "COUNT_GT_X"
    ],
    PROPERTIES: [
      # ----------------- Demographic Rates -----------------
      "P10091", # P10091 -- death rate -- the total number of persons dead per 1,000 population per year
      "P8763",  # P8763 -- birth rate -- the total number of live births per 1,000 population
      "P4841",  # P4841 -- total fertility rate -- average number of children that would be born to a woman over her lifetime who lives in this territorial entity
      "P2250",  # P2250 -- life expectancy -- average life expectancy for this group or species
      
      # ----------------- Economic Normalized Measures -----------------
      "P2132",  # P2132 -- nominal GDP per capita -- country's total GDP divided by the population
      "P10622", # P10622 -- per capita income -- average income earned per person in a given area (city, region, country, etc.) in a specified year
      "P11899", # P11899 -- median household income -- median of the household income (cumulated salaries etc.) of a specific place, such as city, state, country
      "P8843",  # P8843 -- poverty incidence -- proportion of households with per capita income/expenditure less than the per capita poverty threshold to the total number of households
      
      # ----------------- Socioeconomic Indices -----------------
      "P1081",  # P1081 -- Human Development Index -- HDI value of a country
      "P11547", # P11547 -- Happy Planet Index score -- The Happy Planet Index (HPI) is an index of human well-being and environmental impact that was introduced by the New Economics Foundation in 2006
      "P8476",  # P8476 -- BTI Governance Index -- measures change of countries towards democracy and market economy
      "P8477",  # P8477 -- BTI Status Index -- measures change of countries towards democracy and market economy
    ]
  },
  # ------------------------------------------------- ABSOLUTE SCALAR MEASURES -------------------------------------------------
  {
    OPERATIONS: [
      "AVG", "MEDIAN", "MIN", "MAX", "DIFFERENCE(MAX−MIN)", "RATIO(MAX/MIN)", "COUNT_LT_X", "COUNT_GT_X"
    ],
    PROPERTIES: [
      # ----------------- Geographic Scalars ----------------- 
      "P2044", # P2044 -- elevation above sea level -- height of the item (geographical object) as measured relative to sea level
      "P2660", # P2660 -- topographic prominence -- height of a mountain or hill relative to the lowest contour line encircling it (on Earth, maximum 8,848 m)
      "P2659", # P2659 -- topographic isolation -- minimum distance to a point of higher elevation
      "P4511", # P4511 -- vertical depth -- vertical distance from a horizontal area to a point below. Compare with "horizontal depth" (P5524)
      
      # ----------------- Physical & Astronomical Scalars -----------------
      "P2049", # P2049 -- width -- width of an object
      "P2067", # P2067 -- mass -- mass (in colloquial usage also known as weight) of the item
      "P2120", # P2120 -- radius -- distance between the center and the surface of a circle or sphere
      "P2146", # P2146 -- orbital period -- the time taken for a given astronomic object to make one complete orbit about another object
      "P2233", # P2233 -- semi-major axis of an orbit -- semi-major axis of a stable orbit (Astronomy)
      "P2243", # P2243 -- apoapsis -- distance, at which a celestial body is the farthest to the object it orbits
      "P2244", # P2244 -- periapsis -- distance, at which a celestial body is the closest to the object it orbits
      
      # ----------------- Mathematical Scalars -----------------
      "P1164", # P1164 -- group cardinality -- number of elements in a finite group in mathematics
    ]
  },
  # ------------------------------------------------- BOUNDED OR NONLINEAR SCALARS -------------------------------------------------
  {
    OPERATIONS: [
      "AVG", "MEDIAN", "MIN", "MAX", "DIFFERENCE(MAX−MIN)", "COUNT_LT_X", "COUNT_GT_X"
    ],
    PROPERTIES: [
      # ----------------- Astronomical Scalars ----------------- 
      "P2213", # P2213 -- longitude of ascending node -- property of one of the orbital elements used to specify the orbit of an object in space
      "P2045", # P2045 -- orbital inclination -- orbital inclination of a stable orbit
      "P1096", # P1096 -- orbital eccentricity -- amount of the deviation of an orbit from a perfect circle
      "P1215", # P1215 -- apparent magnitude -- measurement of the brightness of an astronomic object, as seen from the Earth
      "P1457", # P1457 -- absolute magnitude -- absolute magnitude of an astronomic object
      
      # ----------------- Graph-Theoretic Scalars ----------------- 
      "P7391", # P7391 -- graph radius -- the minimum eccentricity of any vertex of a graph
      "P7462", # P7462 -- graph diameter -- maximum eccentricity of any vertex in the graph
      "P8986", # P8986 -- graph girth -- length of a shortest cycle contained in the graph
      
      # ----------------- Age-Based Demographic Measures ----------------- 
      "P4442", # P4442 -- mean age -- mean age in a given place
      "P2997", # P2997 -- age of majority -- threshold of adulthood as recognized or declared in law. Use qualifiers "statement is subject of" (P805) to link to item for articles with more detail. Use "start time" (P580) or "point in time" (P585) for historic data
      
      # ----------------- Architectural Features ----------------- 
      "P1139", # P1139 -- floors below ground -- total number of below ground floors of the building
      
      # ----------------- Physical & Environmental Measures ----------------- 
      "P2054", # P2054 -- density -- density of a substance with phase of matter and temperature as qualifiers
      "P2076", # P2076 -- temperature -- qualifier to indicate at what temperature something took place
      "P2927", # P2927 -- water as percent of area -- which percentage of the territory of this item inside coast line and international bounderies is water. Use "percent" (Q11229) as unit
      "P2884", # P2884 -- mains voltage -- voltage of residential mains electricity in a country or region
    ]
  },
  # ------------------------------------------------- GEOGRAPHIC COORDINATES -------------------------------------------------
  {
    OPERATIONS: [
      "PAIRWISE_MAX_DISTANCE", "PAIRWISE_MIN_DISTANCE", "COUNT_WITHIN_RADIUS", "COUNT_DIRECTIONAL"
    ],
    PROPERTIES: [
      # ----------------- Standard Coordinate Location -----------------
      "P625", # P625 -- coordinate location -- geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported
      
      # ----------------- Geographic Center Points -----------------
      "P5140", # P5140 -- coordinates of geographic center -- coordinates of the center of an area. Use qualifier "determination method" (P459) to indicate how

      # ----------------- Extreme Boundary Points ----------------- 
      "P1332", # P1332 -- coordinates of northernmost point -- northernmost point of a location. For an administrative entity this includes offshore islands
      "P1333", # P1333 -- coordinates of southernmost point -- southernmost point of a place. For administrative entities this includes offshore islands
      "P1334", # P1334 -- coordinates of easternmost point -- easternmost point of a location
      "P1335", # P1335 -- coordinates of westernmost point -- westernmost point of a location
    ]
  },
  # ------------------------------------------------- TEMPORAL PROPERTIES -------------------------------------------------
  {
    OPERATIONS: [
      "EARLIEST", "LATEST", "NTH_EARLIEST", "NTH_LATEST", "COUNT_BEFORE", "COUNT_AFTER", "TIME_BETWEEN_FIRST_LAST"
    ],
    PROPERTIES: [
      "P569",  # P569 -- date of birth -- date on which the subject was born
      "P571",  # P571 -- inception -- time when an entity begins to exist; for date of official opening use P1619
      "P575",  # P575 -- time of discovery or invention -- date or point in time when the item was discovered or invented
      "P577",  # P577 -- publication date -- date or point in time when a work or product was first published or released
      "P619",  # P619 -- UTC date of spacecraft launch -- date of spacecraft launch in UTC
      "P1249", # P1249 -- time of earliest written record -- first time a subject was mentioned in writing
    ]
  },
  {
    OPERATIONS: [
      "EARLIEST", "LATEST", "NTH_EARLIEST", "NTH_LATEST", "TIME_BETWEEN_FIRST_LAST"
    ],
    PROPERTIES: [
      "P580",  # P580 -- start time -- time an entity begins to exist or a statement starts being valid
      "P582",  # P582 -- end time -- moment when an entity ceases to exist and a statement stops being entirely valid or no longer be true
    ]
  },
  {
    OPERATIONS: [
      "COUNT_VALUE",
      # "COUNT_HAS_X", "COUNT_NOT_HAS_X", "COUNT_HAS_ANY_X", "COUNT_NOT_HAS_ANY_X"
    ],
    PROPERTIES: [
      "P17",   # P17 -- country -- sovereign state that this item is in (not to be used for human beings)
      "P30",   # P30 -- continent -- continent of which the subject is a part
      "P36",   # P36 -- capital -- seat of government of a country, province, state or other type of administrative territorial entity
      "P7959", # P7959 -- historic county -- traditional, geographical division of Great Britain and Ireland
      "P6885", # P6885 -- historical region -- geographic area which at some point in time had a cultural, ethnic, linguistic or political basis, regardless of present-day borders
      "P47",   # P47 -- shares border with -- countries or administrative subdivisions, of equal level, that this item borders, either by land or water. A single common point is enough.
      "P206",  # P206 -- located in or next to body of water -- body of water on or next to which a place is located
      "P1589", # P1589 -- lowest point -- point with lowest elevation in a region or path
      "P610",  # P610 -- highest point -- point with highest elevation in a region or path
      "P276",  # P276 -- location -- location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object
      "P159",  # P159 -- headquarters location -- city or town where an organization's headquarters is or has been situated. Use P276 qualifier for specific building
      "P937",  # P937 -- work location -- location where persons or organisations were actively participating in employment, business or other work
      "P106",  # P106 -- occupation -- occupation of a person. See also "field of work" (Property:P101), "position held" (Property:P39). Not for groups of people. There, use "field of work" (Property:P101), "industry" (Property:P452), "members have occupation" (Property:P3989).
      "P39",   # P39 -- position held -- subject currently or formerly holds the object position or public office
      "P102",  # P102 -- member of political party -- the political party of which a person is or has been a member or otherwise affiliated
      "P1906", # P1906 -- office held by head of state -- political office that is fulfilled by the head of state of this item
      "P122",  # P122 -- basic form of government -- subject's government
      "P7938", # P7938 -- associated electoral district -- constituencies/electoral districts in which a place is located or is part of. If a municipality/county is split into or part of several districts: add several values. Use only if distinct from administrative entities (P131) in predefined countries
      "P1879", # P1879 -- income classification (Philippines) -- classification grade of a Philippine local government unit based on income
      "P2564", # P2564 -- Köppen climate classification -- indicates the characteristic climate of a place
      "P735",  # P735 -- given name -- first name or another given name of this person; values used with the property should not link disambiguations nor family names
      "P21",   # P21 -- sex or gender -- sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)
      "P19",   # P19 -- place of birth -- most specific known birth location of a person, animal or fictional character
      "P140",  # P140 -- religion or worldview -- religion of a person, organization or religious building, or associated with this subject
      "P37",   # P37 -- official language -- language designated as official by this item
      "P2936", # P2936 -- language used -- language widely used (spoken or written) in this place or at this event or organisation
      "P1999", # P1999 -- UNESCO language status -- degree of endangerment of a language conferred by the UNESCO Atlas of World Languages in Danger
      "P3823", # P3823 -- Ethnologue language status -- language status identifier by Ethnologue.com using EGIDS scale
      "P282",  # P282 -- writing system -- alphabet, character set or other system of writing used by a language, word, or text, supported by a typeface
      "P5109", # P5109 -- has grammatical gender -- specific form of noun class system of a language. Use qualifier "statement is subject of" (P805) to link to language specific item if present
      "P1427", # P1427 -- start point -- starting place of this journey, flight, voyage, trek, migration etc.
      "P81",   # P81 -- connecting line -- railway line(s) subject is directly connected to
    ]
  }
]

PROP_OP_MAPPING = {}
OPERATION_FREQUENCY = defaultdict(int)
for mapping in COMPATIBLE_PROP_OPS:
    for prop in mapping[PROPERTIES]:
        PROP_OP_MAPPING[prop] = mapping[OPERATIONS]
    for op in mapping[OPERATIONS]:
        OPERATION_FREQUENCY[op] += len(mapping[PROPERTIES])
      

OPERATION_DESCRIPTIONS = {
  # ----------------- Quantity -----------------
  "SUM": "sum/total amount across all entities",
  "SUM_TOP_K": "the total amount for the top K entities (ranked by this value)",
  "SUM_BOTTOM_K": "the total amount for the bottom K entities (ranked by this value)",

  "AVG": "the average (mean) value across all entities",
  "AVG_TOP_K": "the average value among the top K entities (ranked by this value)",
  "AVG_BOTTOM_K": "the average value among the bottom K entities (ranked by this value)",

  "MEDIAN": "the median (middle) value when all values are ordered",
  "MAX": "the largest or highest value among all entities",
  "MIN": "the smallest or lowest value among all entities",

  "DIFFERENCE(MAX−MIN)": "the difference between the largest and smallest values",
  "RATIO(MAX/MIN)": "ratio of highest to lowest / how many times larger the maximum value is compared to the minimum value",

  "COUNT_LT_X": "how many entities have values below a given threshold",
  "COUNT_GT_X": "how many entities have values above a given threshold",

  # ----------------- Time -----------------
  "EARLIEST": "the earliest year among all entities",
  "LATEST": "the most recent year among all entities",
  
  "NTH_EARLIEST": "the year that ranks Nth earliest when all years are ordered",
  "NTH_LATEST": "the year that ranks Nth latest when all years are ordered",
  
  "COUNT_BEFORE": "how many entities have a date before a given point in time",
  "COUNT_AFTER": "how many entities have a date after a given point in time",

  "TIME_BETWEEN_FIRST_LAST": "the amount of time between the earliest and latest dates among all entities, in terms of number of years or months or days",

  # ----------------- Coordinates -----------------
  "PAIRWISE_MAX_DISTANCE": "the maximum geographic distance between any two entities in the set, measured along the Earth's surface as great-circle distance using latitude and longitude (Haversine formula, Earth radius 6,371 km)",
  "PAIRWISE_MIN_DISTANCE": "the minimum geographic distance between any two entities in the set, measured along the Earth's surface as great-circle distance using latitude and longitude (Haversine formula, Earth radius 6,371 km)",

  "COUNT_WITHIN_RADIUS": "how many entities lie within a given distance from a reference location, where distance is measured along the Earth's surface using great-circle distance (Haversine formula, Earth radius 6,371 km)",
  "COUNT_DIRECTIONAL": "how many entities lie north, south, east, west, or in a diagonal direction (NE, NW, SE, SW) relative to s reference point",

  # "EXTREME_DIRECTION": "the entity that lies farthest in a given direction (northmost, southmost, eastmost, or westmost)",

  # ----------------- WikibaseItem relations -----------------
  # "COUNT_VALUE": "how many entities have a specified entity as the value of this property"
  "COUNT_HAS_X": "how many entities have the specified entity X as a value of this property",
  "COUNT_NOT_HAS_X": "how many entities do not have the specified entity X as a value of this property",
  
  "COUNT_HAS_ANY_X": "how many entities have at least one value from a given set (X) of specified entities for this property",
  "COUNT_NOT_HAS_ANY_X": "how many entities have none of the values from a given set (X) of specified entities for this property",
}


CONSTRAINT_DESCRIPTIONS = {
  # ----------------- Numeric comparison constraints -----------------
  "GT":  "only entities whose value is strictly greater than a given reference value",
  "GTE": "only entities whose value is greater than or equal to a given reference value",
  "LT":  "only entities whose value is strictly less than a given reference value",
  "LTE": "only entities whose value is less than or equal to a given reference value",

  # ----------------- Directional (1D) spatial constraints -----------------
  "N": "only entities located north of a given reference location",
  "S": "only entities located south of a given reference location",
  "E": "only entities located east of a given reference location",
  "W": "only entities located west of a given reference location",

  # ----------------- Directional (2D) spatial constraints -----------------
  "NE": "only entities located northeast of a given reference location",
  "NW": "only entities located northwest of a given reference location",
  "SE": "only entities located southeast of a given reference location",
  "SW": "only entities located southwest of a given reference location",

  # ----------------- Temporal constraints -----------------
  "BEFORE": "only entities whose time value occurs before a given reference date",
  "AFTER":  "only entities whose time value occurs after a given reference date",

  # ----------------- WikibaseItem membership constraints -----------------
  "HAS":         "only entities that have a specific entity as a value of this property",
  "NOT_HAS":     "only entities that do not have a specific entity as a value of this property",
  "HAS_ANY":     "only entities that have at least one entity from a given set as a value of this property",
  "NOT_HAS_ANY": "only entities that have none of the entities from a given set as values of this property",
  "HAS_ALL":     "only entities that have all entities from a given set as values of this property",
  "NOT_HAS_ALL": "only entities that do not have all entities from a given set as values of this property",
}
