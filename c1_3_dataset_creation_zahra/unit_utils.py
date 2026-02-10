


# Each entry defines: (src_unit, dst_unit): factor
# such that value_in_dst_unit = factor * value_in_src_unit
UNIT_FACTORS = {
    # --------------------
    # Volume
    # --------------------
    ("Milliliter", "Liter"): 0.001,
    ("Liter", "Milliliter"): 1000.0,

    ("Liter", "Quart"): 1.05668821,
    ("Quart", "Liter"): 0.946352946,

    ("Liter", "Cup"): 4.22675284,
    ("Cup", "Liter"): 0.2365882365,

    ("Milliliter", "Quart"): 0.00105668821,
    ("Quart", "Milliliter"): 946.352946,

    ("Milliliter", "Cup"): 0.00422675284,
    ("Cup", "Milliliter"): 236.5882365,

    ("Quart", "Cup"): 4.0,
    ("Cup", "Quart"): 0.25,

    # --------------------
    # Weight
    # --------------------
    ("Gram", "Kilogram"): 0.001,
    ("Kilogram", "Gram"): 1000.0,

    ("Gram", "Ounce"): 0.0352739619,
    ("Ounce", "Gram"): 28.349523125,

    ("Gram", "Pound"): 0.00220462262,
    ("Pound", "Gram"): 453.59237,

    ("Kilogram", "Ounce"): 35.2739619,
    ("Ounce", "Kilogram"): 0.028349523125,

    ("Kilogram", "Pound"): 2.20462262,
    ("Pound", "Kilogram"): 0.45359237,

    ("Ounce", "Pound"): 0.0625,
    ("Pound", "Ounce"): 16.0,

    # --------------------
    # Data
    # --------------------
    ("Megabyte", "Gigabyte"): 1 / 1024,
    ("Gigabyte", "Megabyte"): 1024.0,

    ("Gigabyte", "Terabyte"): 1 / 1024,
    ("Terabyte", "Gigabyte"): 1024.0,

    ("Megabyte", "Terabyte"): 1 / (1024 * 1024),
    ("Terabyte", "Megabyte"): 1024 * 1024,

    # --------------------
    # Time
    # --------------------
    ("Hour", "Day"): 1 / 24,
    ("Day", "Hour"): 24.0,

    # --------------------
    # Coffee pods (identity)
    # --------------------
    ("Pod", "Pod"): 1.0,
    ("K-Cup", "K-Cup"): 1.0,
    ("Pod", "K-Cup"): 1.0,
    ("K-Cup", "Pod"): 1.0,
}



UNIT_CANONICAL = {
    # volume
    "ml": "Milliliter",
    "l": "Liter",
    "liter": "Liter",
    "liters": "Liter",
    "litr": "Liter",
    "cup": "Cup",
    "cups": "Cup",
    "qt": "Quart",
    "quart": "Quart",
    "gallons/cycle": "Gallon per Cycle",

    # mass / weight
    "kg": "Kilogram",
    "g": "Gram",
    "lb": "Pound",
    "lbs": "Pound",
    "oz": "Ounce",

    # length / size
    "inch": "Inch",
    "mm": "Millimeter",
    "m": "Meter",
    "degrees": "Degree",

    # time
    "hour": "Hour",
    "hours": "Hour",
    "days": "Day",
    "months": "Month",
    "year": "Year",
    "years": "Year",
    "min": "Minute",
    "ms": "Millisecond",

    # frequency / rate
    "hz": "Hertz",
    "fps": "Frames per Second",
    "frames per second": "Frames per Second",
    "rpm": "Revolutions per Minute",
    "ghz": "Gigahertz",

    # data / storage
    "mb": "Megabyte",
    "gb": "Gigabyte",
    "tb": "Terabyte",

    # power / electrical
    "w": "Watt",
    "v": "Volt",
    "nits": "Nit",

    # sound
    "db": "Decibel",
    "dba": "Decibel A-weighted",

    # resolution / optics
    "mp": "Megapixel",
    "p": "Pixel",
    "x": "Optical Zoom Factor",
    "x optical zoom": "Optical Zoom Factor",

    # count / discrete
    "piece": "Piece",
    "pods": "Pod",
    "k-cups": "K-Cup",
    "seeds": "Seed",
    "star": "Star",

    # processor / hardware specific
    "th gen": "Generation",

    # currency
    "$": "US Dollar",
}
