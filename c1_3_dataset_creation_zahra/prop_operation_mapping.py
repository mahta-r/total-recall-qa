from collections import defaultdict

OPERATIONS = "OPERATIONS"
PROPERTIES = "PROPERTIES"
DATATYPE = "DATATYPE"



COMPATIBLE_PROP_OPS = [
  {
    DATATYPE: "Quantity",
    OPERATIONS: [
      "AVG", "MEDIAN", "MIN", "MAX",
    ],
    PROPERTIES: [
      "Volume", 
      "Screen Size", 
      "Screen Size (Inches)", 
      "Size",
      "Monitor Size", 
      "Width", 
      "Width (Inches)",  
      "Height",  
      "Depth", 
      "Depth (Inches)", 
      "Blade Length", 
      "Hole Spacing (mm)", 
      "Filter Thread Size", 
      "Fixed Focal Length", 
      "Battery Life", 
      "Battery Runtime", 
      "Burn Time (hours)", 
      "Cord Length", 
      "Lumens", 
      "Brightness", 
      "Maximum RAM", 
      "Capacity", 
      "Capacity_Liters", 
      "Load Capacity", 
      "Power", 
      "Power (Watts)", 
      "Audio Output Power (Watts)", 
      "Wattage", 
      "Water Consumption", 
      "Thread Count", 
      "Maximum Refresh Rate", 
      "Response Time", 
      "Frame Rate", 
      "Camera Weight", 
      "Weight", 
      "Capsule Count", 
      "Page Count", 
      "Noise Level", 
      "Noise Level (dBA)", 
      "HDMI Ports", 
      "USB Ports", 
      "M.2 Slots", 
      "SATA Ports", 
      "Number of Memory Slots", 
      "Installation Holes", 
      "Number of Programs", 
      "Thread Count", 
      "Shelf Life", 
      "Shelf Life / Expiry", 
      "Warranty"
    ]
  },
  {
    DATATYPE: "Quantity",
    OPERATIONS: [
      "AVG", "MEDIAN", "MIN", "MAX", "DIFFERENCE(MAX-MIN)",
    ],
    PROPERTIES: [
      "Price",
      "Average Rating",
    ]
  }, 
  {
    DATATYPE: "Quantity",
    OPERATIONS: [
        "AVG", "MEDIAN", "MIN", "MAX", "MOST_COMMON",
    ],
    PROPERTIES: [
      "Optical Zoom", 
      "Processor Speed", 
      "Spin Speed", 
      "Rotational Speed", 
      "Flow Rate (GPM)", 
      "Storage Capacity", 
      "Internal Storage", 
      "RAM", 
      "Capacity (Place Settings)", 
      "Food Processor Capacity", 
      "Blender Jar Capacity", 
      "Mixer Capacity", 
      "Dust Capacity", 
      "Cache Size", 
      "Refresh Rate", 
      "Front Camera Resolution", 
      "Rear Camera Resolution", 
      "Camera Resolution (Rear)", 
      "Number of Pieces", 
      "Seed Count per Pack", 
      "Energy Rating", 
    ]
  },
  {
    DATATYPE: "Quantity",
    OPERATIONS: [
      "MIN", "MAX", "MOST_COMMON",
    ],
    PROPERTIES: [
      "Webcam Resolution", 
      "Viewing Angle", 
      "Form Factor", 
      "Maximum Aperture", 
      "Minimum Aperture", 
      "Processor Generation", 
      "Voltage", 
      "Resolution",
    ]
  },
  {
    DATATYPE: "Date",
    OPERATIONS: [
      "EARLIEST", "LATEST", "TIME_BETWEEN_FIRST_LAST",
    ],
    PROPERTIES: [
      "Production Year", 
      "Publication Year", 
    ]
  },
  {
    DATATYPE: "OrderedString",
    OPERATIONS: [
      "PERCENTAGE",
    ],
    PROPERTIES: [
      "Type", 
      "Size", 
      "Resolution", 
      "Set Size", 
      "Tank Capacity",
      "Contrast Ratio", 
      "Display Resolution", 
      "Quantity", 
      "Brewing Time", 
      "Pack Size", 
    ]
  },
  {
    DATATYPE: "String",
    OPERATIONS: [
      "PERCENTAGE",
    ],
    PROPERTIES: [
      "Height",
      "Depth",
      "Battery Life",
      "Cord Length",
      "Capacity",
      "Warranty",
      "Energy Rating",
      "USB Ports",
      "Water Consumption",
      "Bulb Type",
      "Noise Level",
      "Weight",
      "Laptop Type",


      # --- OrderedString properties ---
      "Type",                     # product type/category
      "Size",                     # size category (S, M, L, etc.)
      "Resolution",               # display resolution tier
      "Display Resolution",       # screen resolution tier

      # --- High-frequency String properties ---
      "Brand",                    # manufacturer/brand
      "Color",                    # product color
      "Material",                 # construction material
      "Pattern",                  # visual pattern
      "Country of Origin",        # manufacturing country
      "Connectivity",             # connection type(s)
      "Occasion",                 # use occasion
      "Style",                    # design style
      "Finish",                   # surface finish
      "Shape",                    # product shape
      "Lens Type",                # camera/sunglass lens type
      "Handle Material",          # handle material
      "Display Type",             # screen technology
      "Power Source",             # power source type
      "Battery Type",             # battery type
      "Gender",                   # target gender
      "Fit",                      # clothing fit
      "Length",                   # clothing length
      "Closure Type",             # closure mechanism
      "Smart Features",           # smart home features
      "Speed Settings",           # appliance speed options
      "Care Instructions",        # care/cleaning instructions
      "Dishwasher Safe",          # yes/no
      "Microwave Safe",           # yes/no
      "Operating System",         # OS type
      "Intended Use",             # primary use case
      "Use Case",                 # use scenario
      "Number of Lines",          

      # --- Lower-frequency String properties ---
      "Flavor",                   # food/beverage flavor
      "Flavor Profile",          # taste profile
      "Beverage Type",            # type of alcoholic beverage
      "Sub Type",                 # beverage sub-type
      "Alcohol Content (ABV)",    # ABV range category
      "Sweetness Level",          # sweet/dry scale
      "Carbonation",              # carbonation level
      "Serving Temperature",      # serving temp category
      "Aging",                    # aging method
      "Vintage",                  # vintage decade
      "Packaging Type",           # packaging format
      "Ingredients Base",         # base ingredient
      "Dietary",                  # dietary category
      "Roast",                    # coffee roast level
      "Bean Type",                # coffee bean type
      "Grind Size",               # coffee grind size
      "Caffeine Content",         # caffeine level
      "Origin",                   # product origin
      "Organic",                  # organic certification
      "Extraction Method",        # oil extraction method
      "Grade",                    # oil grade
      "Refinement Level",         # oil refinement
      "Smoke Point",              # cooking oil smoke point
      "Fat Content",              # fat content category
      "Flavor / Blend",           # tea flavor/blend
      "Caffeine Level",           # tea caffeine
      "Brewing Temperature",      # tea brew temp
      "Certification",            # tea certification
      "Sweetness",                # tea sweetness
      "Color / Appearance",       # tea appearance
      "Packaging Size",           # tea package size
      "Intended Use / Function",  # tea intended use
      "Form",                     # food form (dry/wet/etc.)
      "Life Stage",               # pet food life stage
      "Special Diet",             # pet food special diet
      "Main Ingredient",          # primary ingredient
      "Protein Content",          # protein level
      "Grain-Free",               # grain free flag
      "Grain Free",               # grain free flag
      "Primary Protein Source",   # protein source
      "Breed Size",               # dog breed size
      "Calorie Density",          # calorie level
      "Package Size",             # food package size
      "For Age Group",            # target age group
      "Ingredients",              # ingredient list
      "Nutritional Additives",    # additives
      "Limited Ingredient Diet",  # limited ingredient
      "Prescription Only",        # prescription flag
      "GMO Free",                 # GMO free flag
      "Texture",                  # food/rug texture
      "Sugar Content",            # sugar level
      "Product Type",             # candy product type
      "Candy Variety",            # candy variety
      "Chocolate Type",           # chocolate type
      "Container",                # container type
      "Scent",                    # candle scent
      "Wick Type",                # candle wick type
      "Packaging",                # candle packaging
      "Season",                   # seasonal use
      "Special Occasion",         # special occasion
      "Department",               # bedding department
      "Set Includes",             # bedding set contents
      "Weave Type",               # fabric weave
      "Texture/Finish",           # fabric texture
      "Hypoallergenic",           # hypoallergenic flag
      "Fabric",                   # dress fabric
      "Neckline",                 # dress neckline
      "Sleeve Style",             # dress sleeve style
      "Shoulder Style",           # dress shoulder
      "Silhouette",               # sunglasses silhouette
      "Frame Color",              # sunglasses frame color
      "Frame Material",           # sunglasses frame material
      "Lens Color",               # lens color
      "Waistband Style",          # pants waistband
      "Leg Style",                # pants leg style
      "Rise",                     # pants rise
      "Stretch",                  # pants stretch
      "Pocket Style",             # pants pocket style
      "Fabric Weight",            # pants fabric weight
      "Age Group",                # pants age group
      "Waist Type",               # shorts waist
      "Feature",                  # shorts feature
      "Active Style",             # shorts style
      "Pockets",                  # shorts pockets
      "Waist Style",              # skirts waist
      "Specialty Style",          # skirts specialty
      "Lining",                   # skirts lining
      "Cushioning",               # socks cushioning
      "Heel Type",                # socks heel
      "Toe Type",                 # socks toe
      "Sustainability",           # sustainable materials
      "Hood",                     # coat hood
      "Insulation Type",          # coat insulation
      "Breathability",            # coat breathability
      "Wind Protection",          # coat wind protection
      "Water Resistance",         # coat/watch water resistance
      "Water Resistant",          # backpack/headphone water resistant
      "Water_Resistance",         # suitcase water resistance
      "Waterproof Rating",        # flashlight waterproof
      "Laptop Compartment",       # backpack laptop compartment
      "RFID Protection",          # wallet RFID
      "Card Slots Count",         # wallet card slots
      "Expandable",               # suitcase expandable
      "Mobility",                 # suitcase mobility
      "Lock_Type",                # suitcase lock type
      "Handle_Type",              # suitcase handle type
      "Exterior_Features",        # suitcase exterior
      "Interior_Features",        # suitcase interior
      "Attachment Method",        # curtain attachment
      "Light Allowance",          # curtain light
      "Thermal/Insulation",       # curtain thermal
      "Noise Reduction",          # curtain noise
      "Decorative Features",      # curtain decor
      "Location",                 # curtain location
      "Handle",                   # cutting board handle
      "Grip Type",                # cutting board grip
      "Edge Style",               # cutting board edge
      "Surface Finish",           # cutting board surface
      "Reversible",               # cutting board reversible
      "Eco Friendly",             # plate/rug eco
      "Eco-Friendly",             # faucet eco
      "Stackable",                # plates stackable
      "Oven Safe",                # bowl/plate oven safe
      "Oven Safe Temperature",    # skillet oven safe temp
      "Bowl Type",                # bowl type
      "Graphic Theme",            # mug theme
      "Graphic Type",             # mug graphic type
      "Handle Type",              # mug handle type
      "Insulation",               # mug insulation
      "Personalization",          # mug personalization
      "Lid Included",             # mug/skillet lid
      "Pan Material",             # skillet material
      "Coating Type",             # skillet coating
      "Handle Style",             # skillet handle style
      "Weight Class",             # skillet weight class
      "Compatibility",            # skillet/phone/webcam compatibility
      "Knife Type",               # kitchen knife type
      "Edge Type",                # kitchen knife edge
      "Sharpness Level",          # kitchen knife sharpness
      "Construction",             # knife/rug construction
      "Pile Height",              # rug pile height
      "Usage",                    # rug usage
      "Placement",                # vase placement
      "Handmade",                 # vase handmade
      "Knob Shape",               # cabinet knob shape
      "Pull Type",                # cabinet pull type
      "Installation Type",        # cabinet/dishwasher installation
      "Spout Type",               # faucet spout
      "Valve Type",               # faucet valve
      "Mount Type",               # faucet/webcam mount
      "Water Type",               # faucet water type
      "Number of Handles",        # faucet handles
      "Locking Mechanism",        # lock mechanism
      "Security Level",           # lock security level
      "Backset",                  # lock backset
      "Door Compatibility",       # lock door type
      "Power Type",               # juicer/lock/screwdriver power
      "Feed Chute Size",          # juicer chute
      "Cleaning Type",            # juicer cleaning
      "Juice Control",            # juicer control
      "Safety Features",          # food processor/juicer safety
      "Design",                   # blender/mixer design
      "Included Accessories",     # blender/food processor/guitar/mixer accessories
      "Beater Functions",         # food processor beater
      "Beater Material",          # food processor beater material
      "Blender Functions",        # food processor blender
      "Blender Blade Material",   # food processor blade
      "Blade Material",           # blender blade"
      "Blender Jar Material",     # food processor jar
      "Dough Hook Material",      # food processor dough hook
      "Dough Hook Size",          # food processor dough hook size
      "Mixer Functions",          # food processor mixer
      "Control Type",             # dishwasher/rice cooker control
      "Drying System",            # dishwasher drying
      "Tub Material",             # dishwasher/washer tub
      "Wash Cycles",              # dishwasher cycles
      "Finish / Color",           # dishwasher finish
      "Drum Material",            # dryer drum
      "Dryer Type",               # dryer type
      "Has Smart Features",       # dryer smart
      "WiFi Connectivity",        # dryer wifi
      "Functions",                # rice cooker functions
      "Wash Programs",            # washer programs
      "Inverter Technology",      # washer inverter
      "Slot Type",                # toaster slot type
      "Toast Darkness Settings",  # toaster darkness
      "Bagel Function",           # toaster bagel
      "Cancel Function",          # toaster cancel
      "Defrost Function",         # toaster defrost
      "Reheat Function",          # toaster reheat
      "Extra Features",           # toaster extras
      "Crumb Tray",               # toaster crumb tray
      "Motor Type",               # vacuum motor
      "Bag Type",                 # vacuum bag
      "Suction Power",            # vacuum suction
      "Navigation Type",          # vacuum navigation
      "Filtration System",        # vacuum filtration
      "Charging Method",          # vacuum charging
      "Attachments Included",     # vacuum attachments
      "Surface Compatibility",    # vacuum surface
      "Design Style",             # humidifier design
      "Mist Temperature",         # humidifier mist temp
      "Humidity Control",         # humidifier control
      "Filter Type",              # humidifier filter
      "Essential Oil Support",    # humidifier essential oil
      "Portability",              # humidifier portability
      "Auto Shut-Off",            # humidifier auto shutoff
      "Coverage Area",            # humidifier coverage
      "Run Time",                 # humidifier runtime
      "Screwdriver Type",         # screwdriver type
      "Tip Material",             # screwdriver tip
      "Handle Color",             # screwdriver handle color
      "Shaft Material",           # screwdriver shaft
      "Adjustable Torque",        # screwdriver torque
      "Ergonomic Design",         # screwdriver ergonomic
      "Magnetic Tip",             # screwdriver magnetic
      "Insulated",                # screwdriver insulated
      "Guitar Type",              # guitar type
      "Number of Strings",        # guitar strings
      "Body Material",            # guitar/microscope body
      "Neck Material",            # guitar neck
      "Fingerboard Material",     # guitar fingerboard
      "Bridge Type",              # guitar bridge
      "Pickup Type",              # guitar pickup
      "Scale Length",             # guitar scale
      "String Type",              # guitar string type
      "Electronics",              # guitar electronics
      #"Ukulele Type",             # ukulele type
      "Microscope Type",          # microscope type
      "Magnification Range",      # microscope magnification
      "Eyepiece Type",            # microscope eyepiece
      "Objective Lens Type",      # microscope objective
      "Stage Type",               # microscope stage
      "Focus Mechanism",          # microscope focus
      "Illumination Source",      # microscope illumination
      "Light Type",               # microscope light
      "Head Type",                # microscope head
      "Power Supply",             # microscope power
      "Application",              # microscope application
      "Digital Support",          # microscope digital
      "Sensor Type",              # camera sensor type
      "Sensor Size",              # camera sensor size
      "ISO Range",                # camera ISO
      "Flash",                    # camera flash
      "Video Recording",          # camera video
      "Image Stabilization",      # camera/lens stabilization
      "Weather Sealing",          # camera/lens weather seal
      "Focus Type",               # lens focus type
      "Lens Format Coverage",     # lens format
      "Specialty",                # lens specialty
      "Zoom Focal Length",        # lens zoom
      "Lens Mount",               # camera/lens mount
      "Lens Quantity",            # phone lens count
      "Polar Pattern",            # microphone polar pattern
      "Frequency Response",       # microphone frequency
      "Sensitivity",              # microphone sensitivity
      "Power Requirement",        # microphone power
      "Diaphragm Size",           # microphone diaphragm
      "Use",                      # microphone use
      "Mounting Type",            # microphone/TV mount
      "Noise Cancellation",       # headphone noise cancel
      "Foldable",                 # headphone foldable
      "Wireless Type",            # headphone wireless
      "Connectivity Ports",       # tablet ports
      "Network Connectivity",     # tablet network
      "Processor Type",           # tablet processor
      "Security Features",        # tablet security
      "Audio Features",           # tablet audio
      "Expandable Storage",       # tablet expandable
      "Stylus Support",           # tablet stylus
      "Build Material",           # laptop/tablet material
      "Keyboard Type",            # laptop keyboard
      "Touchscreen",              # laptop touchscreen
      "Screen Type",              # laptop screen
      "Ports",                    # laptop ports
      "Graphics Brand",          # laptop graphics
      "Graphics Model",          # laptop graphics
      "Graphics Type",           # laptop graphics
      "RAM Type",                 # laptop RAM type
      "Storage Type",             # laptop storage type
      "Graphics Processor",       # desktop GPU
      "Colors",                   # desktop colors
      "Made In",                  # desktop made in
      "User Experience",          # desktop UX
      "Monitor Resolution",       # desktop resolution
      "Number of Cores",          # desktop/laptop cores
      "Processor Brand",          # desktop/laptop CPU brand
      "Panel Type",               # monitor panel
      "Aspect Ratio",             # monitor aspect ratio
      "Color Gamut",              # monitor color gamut
      "HD Format",                # monitor HD format
      "Adjustable Stand",         # monitor stand
      "Built-in Speakers",        # monitor speakers
      "VESA Mount Support",       # monitor VESA
      "Webcam",                   # monitor webcam
      "Energy Efficiency Rating", # monitor/dishwasher efficiency
      "HDR Support",              # monitor/TV HDR
      "Drive Type",               # hard drive type
      "Interface",                # hard drive interface
      "Recording Technology",     # hard drive recording
      "Usage Type",               # hard drive usage
      "Shock Resistance",         # hard drive shock
      "Encryption Support",       # hard drive encryption
      "Operating System Compatibility", # hard drive OS
      "Socket Type",              # motherboard socket
      "Chipset",                  # motherboard chipset
      "Supported RAM Type",       # motherboard RAM type
      "PCIe Slots",               # motherboard PCIe
      "Audio Channels",           # motherboard audio
      "Ethernet Port",            # motherboard ethernet
      "WiFi Support",             # motherboard wifi
      "Onboard Graphics Support", # motherboard graphics
      "RGB Lighting",             # motherboard RGB
      "SIM Slots",                # phone SIM
      "Broadband Generation",     # phone broadband
      "Supported Formats",        # MP3 formats
      "EQ Presets",               # MP3 EQ
      "Voice Recorder",           # MP3 recorder
      "FM Radio",                 # MP3 radio
      "Expandable Memory",        # MP3 memory
      "Smart TV Platform",        # TV platform
      "Screen Form",              # TV screen form
      "Audio Technology",         # TV audio tech
      "HDTV Format",              # TV HDTV format
      "Sony Technology Line",     # TV Sony tech
      "Panasonic Technology Line",# TV Panasonic tech
      "Telephone Type",           # telephone type
      "Handset Count",            # telephone handsets
      "Audio Quality",            # telephone audio
      "Wireless Range",           # telephone range
      "Call Features",            # telephone features
      "Speakerphone",             # telephone speakerphone
      "Mounting Option",          # telephone mounting
      "Watch Type",               # watch type
      "Movement",                 # watch movement
      "Mechanical Movement Type", # watch mechanical
      "Case Material",            # watch case
      "Case Shape",               # watch case shape
      "Band Material",            # watch band
      "Display Technology",       # watch display
      "Gender Target",            # watch gender
      "Additional Functions",     # watch functions
      "Autofocus",                # webcam autofocus
      "Privacy Shutter",          # webcam privacy
      "Low Light Correction",     # webcam low light
      "Video Resolution",         # webcam video res
      "Field of View (FOV)",      # webcam FOV
      "Software Features",        # webcam software
      "Microphone",               # headphone/webcam mic

      # --- Book-specific ---
      "Author",                   # book author
      "Publisher",                # book publisher
      "Genre",                    # book genre
      "Language",                 # book language
      "Format",                   # book format
      "Edition",                  # book edition
      "Fiction Form",             # book fiction form
      "Reference Type",           # book reference type
      "ISBN",                     # book ISBN

      # --- General manufacturer ---
      "Manufacturer",             # product manufacturer

      # --- Decaffeinated / Fair Trade ---
      "Decaffeinated",            # coffee decaf
      "Fair Trade",               # coffee fair trade
      "Flavor Notes",             # coffee flavor notes
      "Container Type",           # coffee container
      "Blend",                    # coffee blend

      # --- Seeds ---
      "Germination Time",         # seed germination
      "Sunlight Requirements",    # seed sunlight
      "Planting Season",          # seed planting season
    ]
  },
]


# Build flat lookup: property_label -> list of compatible operations
PROP_OP_MAPPING = {}
OPERATION_FREQUENCY = defaultdict(int)
for mapping in COMPATIBLE_PROP_OPS:
    if mapping[DATATYPE] not in PROP_OP_MAPPING:
        PROP_OP_MAPPING[mapping[DATATYPE]] = {}
    for prop in mapping[PROPERTIES]:
        PROP_OP_MAPPING[mapping[DATATYPE]][prop] = mapping[OPERATIONS]
    for op in mapping[OPERATIONS]:
        OPERATION_FREQUENCY[op] += len(mapping[PROPERTIES])


OPERATION_DESCRIPTIONS = {
  # ----------------- Quantity -----------------
  "SUM": "sum/total amount across all products (entities)",
  "SUM_TOP_K": "the total amount for the top K products (entities) (ranked by this value)",
  "SUM_BOTTOM_K": "the total amount for the bottom K products (entities) (ranked by this value)",

  "AVG": "the average (mean) value across all products (entities)",
  "AVG_TOP_K": "the average value among the top K products (entities) (ranked by this value)",
  "AVG_BOTTOM_K": "the average value among the bottom K products (entities) (ranked by this value)",

  "MEDIAN": "the median (middle) value when all values are ordered",
  "MOST_COMMON": "the value that appears most frequently among all products (entities)",
  "MAX": "the largest or highest value among all products (entities)",
  "MIN": "the smallest or lowest value among all products (entities)",

  "DIFFERENCE(MAX-MIN)": "the difference between the largest and smallest values",
  "RATIO(MAX/MIN)": "ratio of highest to lowest / how many times larger the maximum value is compared to the minimum value",

  # ----------------- Time -----------------
  "EARLIEST": "the earliest year among all products (entities)",
  "LATEST": "the most recent year among all products (entities)",

  "NTH_EARLIEST": "the year that ranks Nth earliest when all years are ordered",
  "NTH_LATEST": "the year that ranks Nth latest when all years are ordered",

  "TIME_BETWEEN_FIRST_LAST": "the number of years between the earliest and latest dates among all products (entities)",

  # ----------------- String / OrderedString -----------------
  "PERCENTAGE": "what percentage of the filtered products (entities) have a specific value for this property",

  # ----------------- Property-less -----------------
  "COUNT": "the total number of remaining products (entities) after all constraints are applied",
}


CONSTRAINT_DESCRIPTIONS = {
  # ----------------- Numeric comparison constraints -----------------
  "GT":  "only products (entities) whose value is strictly greater than a given reference value",
  "GTE": "only products (entities) whose value is greater than or equal to a given reference value",
  "LT":  "only products (entities) whose value is strictly less than a given reference value",
  "LTE": "only products (entities) whose value is less than or equal to a given reference value",

  # ----------------- OrderedString constraints -----------------
  "MORE_THAN": "only products (entities) whose ordered value is higher than a given reference",
  "LESS_THAN": "only products (entities) whose ordered value is lower than a given reference",

  # ----------------- Temporal constraints -----------------
  "BEFORE": "only products (entities) whose time value occurs before a given reference date",
  "AFTER":  "only products (entities) whose time value occurs after a given reference date",

  # ----------------- String membership constraints -----------------
  "IS":         "only products (entities) that have a specific value for this property",
  "IS_NOT":     "only products (entities) that do not have a specific value for this property",
  "IS_ANY":     "only products (entities) whose value is one of a given set of values",
  "IS_NOT_ANY": "only products (entities) whose value is not any of a given set of values",
}
