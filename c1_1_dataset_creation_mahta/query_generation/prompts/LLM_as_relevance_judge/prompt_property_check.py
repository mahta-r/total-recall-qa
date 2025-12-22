PROPERTY_INPUT_TEMPLATE = """
The following passage is about [{entity_name}] 
[{property_name}] is defined as {property_description}
Does this passage mention the value of [{property_name}] of [{entity_name}]:

{passage}
"""


PROPERTY_PROMPT = """
Your task is to identify if input passages contain any sentences mentioning a given property of an entity. 
You are given the entity, the definition of the property and the passage text.
You only care if that property is mentioned and/or its value is included in the text. You don't care what that value is.
Your answer will be strictly either YES or NO in a single line without any additional words.
You should not explain your answer in any way, just a single YES/NO.


---------------------- Example 1 ----------------------

The following passage is about [Hiʻiaka (moon)]
[density] is defined as density of a substance with phase of matter and temperature as qualifiers
Does this passage mention the value of [density] of [Hiʻiaka (moon)]:

The mass of Hiʻiaka is estimated to be 1.213+0.322−0.311×1019 kg, using precise relative astrometry from the Hubble Space Telescope and taking perturbations into account.[5]: 6  Hiʻiaka's diameter and mass indicate it has a very low density between 0.46 g/cm3[b] and 0.69 g/cm3, which suggests Hiʻiaka's interior consists of highly porous water ice with a rock mass fraction between 50% and 70%

- - -
YES

---------------------- Example 2 ----------------------

The following passage is about [Eastern Alps]
[area] is defined as area occupied by an object
Does this passage mention the value of [area] of [Eastern Alps]:

Eastern Alps The Eastern Alps are usually defined as the area east of a line from Lake Constance and the Alpine Rhine valley, up to the Splügen Pass at the Alpine divide, and down the Liro River to Lake Como in the south. The peaks and mountain passes are lower than the Western Alps, while the range itself is broader and less arched. Geography. Overview. The Eastern Alps include the eastern parts of Switzerland (mainly Graubünden), all of Liechtenstein, and most of Austria from Vorarlberg to the east, as well as parts of extreme Southern Germany (Upper Bavaria), northwestern Italy

- - -
NO

---------------------- Example 3 ----------------------

The following passage is about [Sussex County]
[water as percent of area] is defined as which percentage of the territory of this item inside coast line and international bounderies is water. Use "percent" (Q11229) as unit
Does this passage mention the value of [water as percent of area] of [Sussex County]:

numerous bodies of water in the area where they were able to harvest fish, oysters, and other shellfish in the fall and winter. In the warmer months the women planted and cultivated crops, and processed the food. The men hunted deer and other small mammals, as larger game was not present in the area. European discovery. There is no agreement on which European group was the first to settle in Sussex County. Historians believe that, in the early years of exploration from 1593 to 1630, Swedish explorers were likely the first Europeans to see the Delaware River and the lands

- - -
NO

---------------------- Example 4 ----------------------

The following passage is about [Sussex County]
[water as percent of area] is defined as which percentage of the territory of this item inside coast line and international bounderies is water. Use "percent" (Q11229) as unit
Does this passage mention the value of [water as percent of area] of [Sussex County]:

county has a total area of , of which is land and (21.7%) is water. It is the largest county in Delaware by area. The county's land area comprises 48.0 percent of the state's land area. It is the second-highest percentage of territory of a state of any county in the United States. Sussex County, like Delaware's two other counties, is subdivided into Hundreds. There are several explanations given for how the Hundreds were determined: as an area containing 100 families, an area containing 100 people, or an area that could raise 100 soldiers. Sussex County is apportioned into eleven

- - -
YES

---------------------- Example 5 ----------------------

The following passage is about [Conway group Co2]
[group cardinality] is defined as number of elements in a finite group in mathematics
Does this passage mention the value of [group cardinality] of [Conway group Co2]:

Conway group Co2 In the area of modern algebra known as group theory, the Conway group "Co2" is a sporadic simple group of order 42,305,421,312,000 History and properties. "Co2" is one of the 26 sporadic groups and was discovered by as the group of automorphisms of the Leech lattice Λ fixing a lattice vector of type 2. It is thus a subgroup of Co0. It is isomorphic to a subgroup of Co1. The direct product 2×Co2 is maximal in Co0. The Schur multiplier and the outer automorphism group are both trivial. Representations. Co2 acts as a rank 3 permutation group on

- - -
YES

"""

