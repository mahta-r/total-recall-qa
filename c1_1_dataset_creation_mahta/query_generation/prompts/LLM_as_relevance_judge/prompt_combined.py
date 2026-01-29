PROPERTY_CHECK_INPUT_TEMPLATE = """
ENTITY: [{entity_name}]
PROPERTY: [{property_name}] is defined as {property_description}
STATEMENT: 
the [{property_name}] of [{entity_name}] is [{property_value}]
{time_of_statement}

PASSAGE:
title: {passage_title}
section: {sections}
text: {passage}
"""


PROPERTY_CHECK_PROMPT = """
Your task is to identify if input passages contain any mention of the value of a given property for an entity. 
If the passage mentions the property's value, you must identify if the value mentioned matches a given value: same or different.
If the passage does not mention the property's value, you must identify if passage topic is related to the property: related or unrelated.

You are given an entity, the definiton of a property, and a statement about the value of that property for that entity. 
You are also given a passage that may contain information about the value of the property for that entity. 
In addition to the text content, passage will have the title of the document from which it was extracted.
Optionally, the passage may include the sections it appeared in (in the document) to provide additional context.

-------------------------------------- Steps to follow --------------------------------------

- - - - - - - - - - - - Step 1 - - - - - - - - - - - -

1) First, you will determine if the passage mentions the value of the property for the entity.
For this first task, you only care if that property is mentioned and its value is included in the text. You don't care what that value is.
Very important: the value of the property might be mentioned but not for the given entity, in that case the answer for task 1 is NO.
As this happens frequently, you need to make sure from the context in the sentence or rest of the passage that the value is indeed for the given entity.
Your answer for this first task will be strictly either "YES" or "NO", without any additional words.

- - - - - - - - - - - - Step 2-YES - - - - - - - - - - - -

2-YES) If the passage mentions the value of the property for the entity (answer to the first task is YES), 
you will then determine if the value mentioned in the passage matches the value given in the input statement.

{statement_explanation}

The statement can also potentially include the time of validness of that value.
Important: if the statement includes time of validness, then match holds only if the time of validness of value is also mentioned in the passage. 

For task 2-YES, your answer should be "SAME" or "DIFFERENT", without any additional words. 
Your answer should be SAME if the value in the statement matches the value in the passage (based on the matching criteria) and DIFFERENT if the values are different.

- - - - - - - - - - - - Step 2-NO - - - - - - - - - - - -

2-NO) If the passage does not mention the value of the property for the entity (answer to the first task is NO),
You will then determine if the topic of the passage is related to the property defined in the input or not.
Your criteria for deciding if the topic is related is to consider we want to add a sentence to this passage that mentions the value of the property:
If that sentence can fit naturally in the passage without making it inconsistent or illogical --> RELATED
If the sentence would be out-of-place in that passage --> UNRELATED 
For task 2-NO, your answer should be "RELATED" or "UNRELATED", without any additional words. 

- - - - - - - - - - - - Step 3 - - - - - - - - - - - -

3) Your final answer will be the combination of the answer to task 1 and task 2, separated by a single dash.
That means your final answer will be one of the following 4 options:

"YES-SAME"
"YES-DIFFERENT"
"NO-RELATED"
"NO-UNRELATED"

You should not explain your answer in any way, just one of the above options in a single line without any additional words/whitespace.


-------------------------------------- Examples --------------------------------------

- - - - - - - - - - - - Example 1 - - - - - - - - - - - -

ENTITY: [Namaka]
PROPERTY: [orbital eccentricity] is defined as amount of the deviation of an orbit from a perfect circle
STATEMENT: 
the [orbital eccentricity] of [Namaka] is 0.249

PASSAGE: 
title: Namaka (moon)
section:
text: Namaka follows a highly tilted elliptical orbit with an eccentricity of about 0.22 and an inclination of roughly with respect to both Haumea's equator and Hiiaka's orbital plane. Namaka is heavily perturbed by both the gravitational influence of Hiiaka and the variable gravitational field of Haumea's elongated shape, which results in a time-varying eccentricity and inclination as well as nodal and apsidal precession of Namaka's orbit. The ratio of Namaka's and Hiiaka's orbital periods is , which means Namaka and Hiiaka may be in (or is close to) a 8:3 mean-motion orbital resonance with each other, where Hiiaka completes

- - -
YES-DIFFERENT


- - - - - - - - - - - - Example 2 - - - - - - - - - - - -

ENTITY: [Dalian]
PROPERTY: [population] is defined as number of people inhabiting the place; number of people of subject
STATEMENT: 
the [population] of [Dalian] is [7450785.0]
this statement is valid as of 2020.

PASSAGE:
title: Dalian
section: Demographics
text: The population of Dalian according to the 2010 census totaled 6.69 million. The total registered population on household at year end 2014 was 5.943 million, with a net increase of 29,000 over the previous year.

- - -
YES-DIFFERENT


- - - - - - - - - - - - Example 3 - - - - - - - - - - - -

ENTITY: [Brezovica]
PROPERTY: [coordinate location] is defined as geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported
STATEMENT: 
the [coordinate location] of [Brezovica] is [(45.715556, 15.920583)]


PASSAGE:
title: Brezovica, Zagreb
section: 
text: Brezovica, Zagreb Infobox: official_name: Brezovica other_name:  native_name:  nickname:  settlement_type: City District motto:  image_skyline: Dvorac Brezovica južni ulaz.jpg imagesize:  image_caption: Brezovica castle view image_flag:  flag_size:  image_seal:  seal_size:  image_map: Map of Brezovica District (Zagreb).svg mapsize:  map_caption: Brezovica as a part of Zagreb pushpin_map: Croatia pushpin_label_position: bottom pushpin_mapsize:  pushpin_map_caption:  subdivision_type: Country subdivision_name:  subdivision_type1: County subdivision_name1: City of Zagreb government_footnotes:  government_type:  leader_title:  leader_name:  established_title:  established_date:  unit_pref: Imperial area_footnotes:  area_total_km2: 4.5 area_land_km2:  population_as_of: 2021 population_footnotes:  population_note:  population_total: 642 population_urban:  population_density_km2: auto population_metro:  timezone: CET utc_offset: +1 timezone_DST: CEST utc_offset_DST: +2 coordinates: 45.7242 N 15.9077 E region:HR-21_type:city(10884) title elevation_footnotes:  elevation_m:  elevation_ft:  postal_code_type:  postal_code:  area_code:  website:  footnotes: 

- - -
YES-SAME


- - - - - - - - - - - - Example 4 - - - - - - - - - - - -

ENTITY: [Gaza Governorate]
PROPERTY: [area] is defined as area occupied by an object
STATEMENT: 
the [area] of [Gaza Governorate] is [70.0 (unit: square kilometre)]


PASSAGE:
title: Gaza Governorate
section: 
text: Gaza Governorate Governorate of Palestine 31.52ºN, 34.45ºE Infobox: native_name: ar محافظة غزة type: Governorate image_map: Gaza in Palestine.svg mapframe: yes mapframe-zoom: 11 subdivision_type: Country subdivision_name:  coordinates:  area_total_km2: 70 population_total: 652,597 population_as_of: 2017 population_footnotes:  population_density_km2: auto iso_code: PS-GZA The Gaza Governorate (محافظة غزة), also alternatively known as Gaza City Governorate, is one of the 16 Governorates of Palestine, located in the north central Gaza Strip. Gaza is claimed by the State of Palestine, but the land is under the partial control of Hamas, while around a third of the governorate, its border with Israel, airspace, and maritime territory, are all controlled by the IDF.

- - -
YES-SAME

- - - - - - - - - - - - Example 5 - - - - - - - - - - - -

ENTITY: [Harju County]
PROPERTY: [shares border with] is defined as countries or administrative subdivisions, of equal level, that this item borders, either by land or water. A single common point is enough.
STATEMENT: 
the [shares border with] of [Harju County] is [Lääne County, Järva County, Lääne-Viru County, Rapla County]


PASSAGE:
title: Harju County
section: 
text: Harju County County of Estonia Harju County (Harju maakond or "Harjumaa"), is one of the fifteen counties of Estonia. It is situated in northern Estonia, on the southern coast of the Gulf of Finland, and borders Lääne-Viru County to the east, Järva County to the southeast, Rapla County to the south, and Lääne County to the southwest. The capital and largest city of Estonia, Tallinn, is situated in Harju County. Harju is the largest county in Estonia in terms of population, as almost half (45%) of Estonia's population lives in Harju County.

- - -
YES-SAME

- - - - - - - - - - - - Example 6 - - - - - - - - - - - -

ENTITY: [Shenyang]
PROPERTY: [population] is defined as number of people inhabiting the place; number of people of subject
STATEMENT: 
the [population] of [Shenyang] is [9070093.0]
this statement is valid as of 2020.

PASSAGE:
title: Shenyang
section: Administrative divisions
text: It previously had only an area of 39 km2 and a population of 764,419. In May 2002, the Shenyang city government annexed a large area of suburban land from the neighbouring Yuhong District to establish a new state-level development zone—the Shenyang Economic and Technological Development Zone (沈阳经济技术开发区), and transferred its administration to Tiexi District to form the Tiexi New District (铁西新区), thus giving Tiexi District the current "necked" shape on the map. The new Tiexi District now has a population of 907,091 (2014), a total area of 286 km2, and enjoys the same administrative rank as a municipality (Administrative Committee of Shenyang).

- - -
NO-RELATED

- - - - - - - - - - - - Example 7 - - - - - - - - - - - -

ENTITY: [Simmering]
PROPERTY: [elevation above sea level] is defined as height of the item (geographical object) as measured relative to sea level
STATEMENT: 
the [elevation above sea level] of [Simmering] is [156.0 (unit: metre)]


PASSAGE:
title: Simmering (Vienna)
section: Geography
text: The district lies in the southern part of Vienna. It borders the Danube and Danube Canal to the east and the "East railway" to the west. Of all the districts, Simmering is the lowest district in terms of elevation.

- - -
NO-RELATED

- - - - - - - - - - - - Example 8 - - - - - - - - - - - -

ENTITY: [Kunar Province]
PROPERTY: [official language] is defined as language designated as official by this item
STATEMENT: 
the [official language] of [Kunar Province] is [Dari, Pashto]


PASSAGE:
title: Kunar Province
section: History
text: The major city of Chitral (in modern Pakistan) was the base of a Mehtar (King), who ruled under the Maharajah of Kashmir According to a US Army paper, the Pashtuns of Kunar and the Kafirs of Kunar/Nuristan eventually joined in the 20th century. Fundamentalist religion came to the region in the 1950s but the heavy unification happened during the Soviet–Afghan War (1979–88). Some of the first anti-government forces (lashkar) rose in the Kunar region. Kerala, a town near Asadabad, was the site of the 1979 Kerala massacre, where the male population of a village was allegedly murdered by the People's Democratic Party of Afghanistan and its Soviet advisors.

- - -
NO-UNRELATED

"""


DATATYPE_EXPLANATION = {
    "Quantity": """
        The type of the given property is numerical. 
        The statement will contain the numerical value of the property, and potentially the unit of the numerical value.
        Values match if both the numerical value and unit (if given) are the same in the passage. If unit is not mentioned in passage, only the numerical value needs to match.
        The values don't need to be exactly the same for a match, a small difference is acceptable.
        The formatting of the numerical value in the passage can vary, for example "1,000" is equivalent to "1000" and "1e3".
    """,
    "Time": """
        The type of the given property is dates. 
        The statement will contain the date value of the property, and the calendar type of the date.
        Values match if both the date value and calendar type (if given) are the same in the passage. If calendar is not mentioned in passage, only the date value needs to match.
        if given date only mentions year, matches if year is the same. 
        if given date only mentions month and year, matches if both month and year are the same. 
        if given date mentions day, month, year, matches if all three are the same.
        The formatting of the date in the passage can vary, for example "January 1, 2000" is equivalent to "2000-01-01" and "2000/01/01".
    """,
    "GlobeCoordinate": """
        The type of the given property is coordinate locations. 
        The statement will contain the (latitude,longitude) values of the property.
        Values match if both latitude and longitude values are the same in the passage. 
        The values don't need to be exactly the same for a match, a small difference is acceptable (for example up to 0.1 degrees).
        The formatting of the coordinate values in the passage can vary, for example "40.7128 N, 74.0060 W" is equivalent to "(40.7128, -74.0060)" and "40.7128°N 74.0060°W".
    """,
    "WikibaseItem": """
        The type of the given property is a relation to another entity.
        The statement will contain the name of the entity.
        Values match if the name of the entity is mentioned as value of property in the passage.
        The formatting of the entity name in the passage can vary, for example "New York City" is equivalent to "new york" or "NYC".
    """
}



# DATATYPE_PROMPT = """
# The property can have 1 of 4 types: Quantity, Time, GlobeCoordinate, WikibaseItem.
# Depending on the property type, the statement will contain different info, and the condition for matching will be different:
# - Quantity: the numerical value of the property, and potentially the unit of the numerical value. 
# matches if both the numerical value and unit (if given) are the same in the passage. if unit is not mentioned in passage, only the numerical value needs to match.
# - Time: the date value of the property, and the calendar type of the date.
# matches if both the date value and calendar type (if given) are the same in the passage. if calendar is not mentioned in passage, only the date value needs to match.
# if given date only mentions year, matches if year is the same. 
# if given date only mentions month and year, matches if both month and year are the same. 
# if given date mentions day, month, year, matches if all three are the same.
# - GlobeCoordinate: the (latitude,longitude) values of the property.
# matches if both latitude and longitude values are the same in the passage.
# - WikibaseItem: the name of the entity 
# matches if the name of the entity is mentioned as value of property in the passage.
# """