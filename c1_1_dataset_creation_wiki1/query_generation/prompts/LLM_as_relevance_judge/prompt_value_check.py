VALUE_INPUT_TEMPLATE = """
ENTITY: [{entity_name}]
PROPERTY: [{property_name}] is defined as {property_description}
STATEMENT: 
the [{property_name}] of [{entity_name}] is {property_value}

PASSAGE: {passage}
"""


VALUE_PROMPT = """
Your task is to identify if the value of a property mentioned in a passage matches a given value.
You are given an entity, the definiton of a property, and a statement about the value of that property for that entity. 
The statement will contain the value of the property, and potentially the unit and time of validness of that value. 
You're also given a passage that contains information about the value of the property for that entity. 
The value of the entity in the passage may be the same or different from the given value in the input statement.
Your answer should be YES if the value in the statement matches the value in the passage and NO if the values are different.
Important: if the statement mentions unit and/or time of validness, then your answer will only be yes if the unit and/or time given are also mentioned in the passage.
You should not explain your answer in any way, just a single YES/NO.


---------------------- Example 1 ----------------------

ENTITY: [Namaka]
PROPERTY: [orbital eccentricity] is defined as amount of the deviation of an orbit from a perfect circle
STATEMENT: 
the [orbital eccentricity] of [Namaka] is 0.249

PASSAGE: . Namaka follows a highly tilted elliptical orbit with an eccentricity of about 0.22 and an inclination of roughly with respect to both Haumea's equator and Hiiaka's orbital plane. Namaka is heavily perturbed by both the gravitational influence of Hiiaka and the variable gravitational field of Haumea's elongated shape, which results in a time-varying eccentricity and inclination as well as nodal and apsidal precession of Namaka's orbit. The ratio of Namaka's and Hiiaka's orbital periods is , which means Namaka and Hiiaka may be in (or is close to) a 8:3 mean-motion orbital resonance with each other, where Hiiaka completes

- - -
NO


---------------------- Example 2 ----------------------

ENTITY: [Pirin National Park]
PROPERTY: [employees] is defined as total number of employees of a company at a given "point in time" (P585). Most recent data would generally have preferred rank; data for previous years normal rank (not deprecated rank). Add data for recent years, don't overwrite
STATEMENT: 
the [employees] of Pirin National Park is 49.0
this statement is valid as of 2025-09-01T00:00:00Z

PASSAGE: Natura 2000. Pirin National Park is listed as an important bird and biodiversity area by BirdLife International. Pirin National Park is managed by a directorate subordinated to the Ministry of Environment and Water of Bulgaria based in the town of Bansko at the northern foothills of the mountain. As of 2004, the park administration had 92 employees. There are two visitor and information centres located in Bansko and Sandanski. The park is divided in six sectors: "Bayuvi Dupki" with office in Razlog, "Vihren" with office in Bansko, "Bezbog" with office in Dobrinishte, "Trite Reki" and "Kamenitsa", both with office in

- - -
NO


---------------------- Example 3 ----------------------


ENTITY: [Kent County]
PROPERTY: [water as percent of area] is defined as which percentage of the territory of this item inside coast line and international bounderies is water. Use "percent" (Q11229) as unit
STATEMENT: 
the [water as percent of area] of Kent County is 26.6 in terms of percent (unit)

PASSAGE: in the small town of Frederica. The suits, dubbed the "A7L," was first flown on the Apollo 7 mission in October 1967, and was the suit worn by Neil Armstrong and Buzz Aldrin on the Apollo 11 mission. The company still manufactures spacesuits to this day—the present-day Space Shuttle "soft" suit components (the arms and legs of the suit). Geography. According to the U.S. Census Bureau, the county has a total area of , of which is land and (26.6%) is water. Kent County, like all of Delaware's counties, is subdivided into Hundreds. There are several explanations given for how

- - -
YES

"""