import collections
import random

template_response = {
    "restaurant": "By the way, are you hungry? What would you like for dinner?",
    'hotel': 'By the way, have you lived in a hotel? Did you eat breakfast?',
    'movie': 'By the way, do you want to find movies by genre and optionally director?',
    'song': 'By the way, what songs do you like? Do you want to search for a song?',
    'transportation': 'By the way, what is your favorite transportation?',
    'attraction': 'By the way, do you want to browse attractions in a given city?'
}

def init_template():
    template_response = {
        "restaurant": "By the way, are you hungry? What would you like for dinner?",
        "hotel": "By the way, have you lived in a hotel? Did you eat breakfast?",
        "movie": "By the way, do you want to find movies by genre and optionally director?",
        "song": "By the way, what songs do you like? Do you want to search for a song?",
        "transportation": "By the way, what is your favorite transportation?",
        "attraction": "By the way, do you want to browse attractions in a given city?",
    }

    transition_words = [
        "Oops.",
        "Pardon me.",
        "Excuse me.",
        "Sorry for interrupting.",
        "I beg your pardon.",
        "Sorry.",
        "Sorry for bothering.",
    ]
    people = {
        "he": [
            "boyfriend",
            "husband",
            "dad",
            "boss",
            "colleague",
            "student",
            "teacher",
            "classmate",
            "friend",
            "petit amis",
            "son",
            "nephew",
            "cousin",
            "brother",
            "grandpa",
        ],
        "she": [
            "girlfriend",
            "wife",
            "dad",
            "boss",
            "colleague",
            "student",
            "teacher",
            "classmate",
            "friend",
            "petit amis",
            "daughter",
            "niece",
            "cousin",
            "sister",
            "grandma",
        ],
        "they": [
            "family",
            "parents",
            "children",
            "kids",
            "colleagues",
            "grandparents",
            "friends",
            "colleagues",
        ],
    }

    requirements = [
        "could you wait a minute",
        "could you wait for a minute",
        "could you hold on for a minute",
        "could you hold on a sec",
        "could you wait for a while",
        "hang on a second",
        "just a minute",
        "just a sec",
        "just a moment",
        "hold on",
        "hang on",
    ]
    reasons = [
        "called",
        "texted",
        "just called",
        "just texted",
        "suddenly called",
    ]

    first_transition = []
    for a in transition_words:
        for k, l in people.items():
            for b in l:
                for c in reasons:
                    for d in requirements:
                        first_transition.append(
                            f"{a} My {b} {c} me. ".capitalize() +  f"{d.capitalize()}?"
                        )

    auxilliary_verb = [
        "can you",
        "could you",
        "will you",
        "would you",
        "could you please",
        "will you please",
        "can you please",
        "would you please",
        "would you like to",
    ]

    verb = [
        "recommend",
        "introduce",
        "suggest",
        "tell",
    ]

    preceding_words = {
        "general": [
            "new",
            "cool",
            "special",
            "unique",
            "good",
            "great",
            "nice",
            "recently heard",
            "brilliant"
        ],
        "restaurant": ["delicious", "tasty", "recipe for"],
        "hotel": ["comfortable", "cozy", "cheap", "high quality", "high end", "certified"],
        "movie": ["highly rated", "interesting", "premiered", "old", "classic"],
        "song": ["popular"],
        "transportation": ["fast", "cheap", "reliable", "safe", "reliable"],
        "attraction": ["famous", "well-knowned"]
    }

    noun = {
        "hotel": [
            "hotel",
            "b & b",
            "hostel",
            "inn",
            "guesthouse",
            "motel",
            "resort",
            "lobby",
            "cinema",
        ],
        "restaurant": [
            "beverage",
            "bistro",
            "dessert",
            "ice cream",
            "cuisine",
            "soda",
            "fish",
            "lobster",
            "ham",
            "pasta",
            "seafooder",
            "egg",
            "risotto",
            "rice",
            "liquor",
            "bread",
            "brunch",
            "oyster",
            "bite",
            "spaghetti",
            "pudding",
            "food",
            "burger",
            "fried foods",
            "pizzeria",
            "chicken",
            "vegetable",
            "chew",
            "barbecue",
            "beef",
            "steak",
            "bbq",
            "cafe",
            "toast",
            "banquet",
            "sandwich",
            "curry",
            "cocktail",
            "pie",
            "bacon",
            "pizza",
            "stew",
            "tea",
            "salmon",
            "meat",
            "cake",
            "salad",
            "wine",
            "waffle",
            "omelet",
            "sushi",
            "meal",
            "soup",
        ],
        "movie": [
            "movie",
            "film",
            "fantasy movie",
            "cartoon",
            "fiction",
            "thriller",
            "theater",
            "documentary",
            "story",
            "drama",
            "horror movie",
            "musical",
            "comedy",
            "soundtrack",
        ],
        "song": [
            "rhythm",
            "classical",
            "cd",
            "singer",
            "punk",
            "choir",
            "instrumental",
            "music",
            "chorus",
            "tune",
            "melody",
            "opera",
            "sing",
            "hum",
            "drum",
            "rhyme",
            "vocal",
            "recording",
            "rap",
            "piano",
            "studio",
            "verse",
            "lyrics",
            "jam",
            "concert",
            "radio",
            "guitar",
            "album",
            "pop",
            "song",
            "audio",
            "duo",
            "chord",
            "jazz",
            "orchestra",
            "band",
            "rock",
            "karaoke",
            "musician",
            "musical",
            "symphony",
            "lullaby",
            "top hits",
        ],
        "transportation": [
            "automobile",
            "vehicle",
            "taxi",
            "car",
            "plane",
            "ship",
            "subway",
            "bus",
            "transportation",
            "rail",
            "boat",
            "airplane",
            "flight",
            "railway",
            "train",
            "airline",
            "jet",
            "cab",
            "station",
            "railroad",
            "metro",
            "bicycle",
        ],
        "attraction": [
            "park",
            "hill",
            "zoo",
            "landmark",
            "beach",
            "view",
            "national park",
            "island",
            "museum",
            "tower",
            "landscape",
            "mountain",
            "ceremonial",
            "memorial",
            "landmark",
            "amusement park",
            "somewhere for honeymoon",
            "somewhere for trip",
            "somewhere for travel",
            "somewhere for camping",
            "lake",
            "hall",
            "temple",
            "somewhere for sightseeing",
            "theme park",
            "statue",
        ],
    }


    real_transition = collections.defaultdict(list)
    for aux in auxilliary_verb:
        for v in verb:
            for key, value in noun.items():
                for adj in preceding_words[key]:
                    for n in value:
                        article = "an" if any([adj.startswith(x) for x in "aeiou"]) else "a"
                        real_transition[key].append(
                            f"{aux} {v} {article} {adj} {n}?".capitalize()
                        )
                        real_transition[key].append(
                            f"{aux} {v} any {adj} {n}?".capitalize()
                        )
                for adj in preceding_words['general']:
                    for n in value:
                        article = "an" if any([adj.startswith(x) for x in "aeiou"]) else "a"
                        real_transition[key].append(
                            f"{aux} {v} {article} {adj} {n}?".capitalize()
                        )
                        real_transition[key].append(
                            f"{aux} {v} any {adj} {n}?".capitalize()
                        )
        return first_transition, real_transition


def get_transition(first_transition, real_transition, intent, turn = 0):
    # if turn == 0:
    #     return random.choice(first_transition) + " " + random.choice(real_transition[intent])
    # else:
    return "Forget about it. " + random.choice(real_transition[intent])

first_transition, real_transition = init_template()