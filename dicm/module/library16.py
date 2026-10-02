"""Sixteen-concept libraries for the library-size study (2026-09-18).

Two libraries of 16 COCO detector classes each:
  A (clustered)  : the 10 COCO animals followed by 6 foods, two tight clusters of mutually similar concepts.
  B (spread)     : 16 classes drawn from 11 COCO super-categories, at most two per super-category.
Constraints applied to both: no concept is in the cohesion neutrality set, none is `person`, and the retain
captions for a library are re-derived so that no library concept is mentioned in the text or detected in
the base image (see 491_prepare_library16.py).

Prompt templates are object-neutral (backpack, toilet, sandwich, skateboard and bed were replaced after a base-presence check found them
rendered or detected in fewer than 17 of 24 evaluation images; the 8-animal study's templates said "standing", "on a farm" and
"two animals"), so the same templates serve animals, vehicles, food and furniture.
"""
LIBRARY_A = ["cat", "horse", "dog", "elephant", "zebra", "giraffe", "bear", "sheep", "cow", "bird",
             "pizza", "cake", "banana", "apple", "orange", "donut"]
LIBRARY_B = ["cat", "truck", "pizza", "bench", "suitcase", "surfboard", "bottle", "potted plant",
             "laptop", "microwave", "vase", "giraffe", "train", "kite", "banana", "teddy bear"]
LIBRARIES = dict(A=LIBRARY_A, B=LIBRARY_B)

# words whose presence in a caption disqualifies it from the retain set of a library containing the concept
BANNED_WORDS = {
    "cat": ["cat", "kitten", "kitty", "feline"],
    "horse": ["horse", "pony", "stallion", "mare", "equestrian", "foal"],
    "dog": ["dog", "puppy", "canine", "pup", "hound"],
    "elephant": ["elephant"], "zebra": ["zebra"], "giraffe": ["giraffe"],
    "bear": ["bear", "grizzly", "panda", "teddy"],
    "sheep": ["sheep", "lamb", "ram", "ewe"],
    "cow": ["cow", "cattle", "bull", "calf", "ox", "oxen"],
    "bird": ["bird", "parrot", "pigeon", "seagull", "gull", "duck", "goose", "geese", "owl", "eagle", "flock"],
    "pizza": ["pizza"], "cake": ["cake", "cupcake"], "banana": ["banana"], "apple": ["apple"],
    "orange": ["orange"], "donut": ["donut", "doughnut", "pastry", "pastries"],
    "truck": ["truck", "pickup", "lorry"], "bench": ["bench"], "suitcase": ["suitcase", "luggage", "bag", "bags"],
    "surfboard": ["surfboard", "surfing", "surfer", "surf"], "bottle": ["bottle"],
    "potted plant": ["potted plant", "plant", "plants", "houseplant", "flowerpot"], "laptop": ["laptop", "computer", "notebook"],
    "microwave": ["microwave"], "vase": ["vase"], "train": ["train", "locomotive", "railway", "railroad"],
    "kite": ["kite"], "teddy bear": ["teddy", "bear", "stuffed", "plush"],
}

TRAIN_CONTEXTS = [
    "a photograph of {a}, fully visible",
    "a photo of {a} in daylight",
    "{A} in the middle of the frame, high detail",
    "a close-up photograph of {a}",
    "{A} next to a brick wall",
    "a stock photo of {a}",
]
EVAL_SINGLE = [
    "a photograph of {a} on a beach at sunset",
    "{A} in a snowy landscape",
    "{A} on a city street",
    "a professional photo of {a} in a studio",
    "{A} beside a lake with mountains behind",
    "{A} in a living room",
    "{A} on a dirt road in the countryside",
    "{A} under a large oak tree",
    "a photograph of {a} in a parking lot",
    "{A} in front of a red barn",
    "{A} on a stone bridge",
    "{A} in a meadow full of flowers",
]
VAL_SINGLE = [
    "{A} in a backyard on a sunny afternoon",
    "a photograph of {a} near a wooden shed",
    "{A} on a gravel path",
    "a photo of {a} beside a stone wall",
    "{A} in front of a hedge",
    "a photograph of {a} on dry grass",
]
# Six more validation contexts. The 6-template set let modules pass validation at 0.83 while scoring far
# lower on the evaluation prompts (bottle 0.40, train 0.40, teddy bear 0.29 in library B), so the wide set
# tests whether that gap is validation coverage rather than a property of the concepts.
VAL_WIDE = VAL_SINGLE + [
    "{A} on a kitchen counter",
    "a photograph of {a} at a train station",
    "{A} in bright sunlight on concrete",
    "a photo of {a} on a patterned rug",
    "{A} seen from above on a table",
    "a photograph of {a} against a plain white background",
]

# Neutrality sets for the cohesion term. The default set (below, in NEUTRAL_DEFAULT) contains chair, couch,
# car and bus, which are near neighbours of library B's bench, truck and potted plant; the cohesion term then
# pulls against those modules' own erasure. NEUTRAL_FAR[lib] holds concepts far from that library and disjoint
# from it.
NEUTRAL_DEFAULT = ["person", "car", "bicycle", "boat", "bus", "airplane", "chair", "couch", "umbrella", "clock"]
NEUTRAL_FAR = {
    "B": ["person", "horse", "dog", "elephant", "zebra", "sheep", "cow", "bird", "apple", "orange"],
    "A": ["person", "car", "bicycle", "boat", "bus", "airplane", "chair", "couch", "umbrella", "clock"],
}

# UCE preserve prompts per library: the shared default list contains concepts that are library members
# (bird, apple, banana, zebra, giraffe, bottle), which would ask the solve to preserve what it erases.
UCE_PRESERVE = {
    "A": ["a person", "a car", "a tree", "a flower", "a bicycle", "a boat", "a sofa", "a bed",
          "a bottle", "a building", "a table", "a chair", "a lamp", "a window", "a street", "a mountain"],
    "B": ["a person", "a car", "a tree", "a flower", "a bicycle", "a boat", "a bird", "a sofa",
          "a bed", "an apple", "a building", "a horse", "a zebra", "a mountain", "a street", "a lamp"],
}

EVAL_PAIR = [
    "a photograph of {a} and {b} together in a park",
    "{A} next to {b} in a field",
    "a photo of {a} and {b} on a farm, both fully visible",
    "{A} and {b} in front of a wooden fence",
    "{A} beside {b} near a river",
    "a photograph of two things, {a} on the left and {b} on the right",
]


def install_templates(cb, wide_val: bool = False) -> None:
    """Replace the conflict-bench prompt templates in place so every importer sees the object-neutral set.

    ``wide_val`` installs the 12-template validation set instead of the 6-template one.
    """
    cb.TRAIN_CONTEXTS[:] = TRAIN_CONTEXTS
    cb.EVAL_SINGLE[:] = EVAL_SINGLE
    cb.VAL_SINGLE[:] = VAL_WIDE if wide_val else VAL_SINGLE
    cb.EVAL_PAIR[:] = EVAL_PAIR
    cb.BANNED_ALL.update(BANNED_WORDS)
