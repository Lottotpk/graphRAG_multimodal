ABSTRACT_PROMPT = """
    Answer the question from the provided images with a few specific keywords (1-3 words) in these eight aspects.
    If there is nothing that the images can be represented in this cateogry, leave that section blank and do not make up answer.
    The examples are provided in each of the category. You may or may not use the keyword from these examples.
    0. Emotion / Mood: Describes the overall atmosphere or emotional tone of the image, for example
        - Energetic, Calm, Joyful, Dramatic, Melancholy, Serene, ...
        - Inspirational, Mysterious, Whimsical, Intense, Playful, Rebellious, ...
    1. Purpose / Context: Specifies the intended use case or application scenario, for example
        - Gym poster, Living-room wall art, Kids' room decoration
        - Corporate slide cover, Startup pitch deck, Meditation app hero
        - Fashion magazine cover, Wedding invitation, Product packaging
        - Music festival poster, NFT artwork, Book cover
    2. Style / Visual Attributes: Highlights the artistic or design style of the work, for example
        - Minimalist, Vintage, Futuristic, Pop-art, Emo, Grunge, Cyberpunk
        - Vaporwave, Surrealism, Bauhaus, Brutalist, Kawaii, Hand-drawn/Doodle
        - Advertising / Commercial ad, Corporate bland, Geometric, Dynamic motion
    3. Medium / Material: Indicates the creative medium or material representation, for example
        - Photography, Oil painting, Watercolor, Digital 2D render
        - Sketch, Collage, Graffiti, Infographic, Mixed media
    4. Color / Lightning Characteristics: Focuses on color atmosphere and lighting style, for example
        - Vivid, Bright, Pastel, Monochrome, Neon
        - Warm tones, Cold tones, High-contrast, Low-contrast
        - Dramatic lighting, Soft lighting, Golden hour, Chiaroscuro
    5. Cultural / Regional Elements: Incorporates cultural or regional aesthetics, for example
        - Japanese, Chinese, Nordic, Mediterranean, Middle Eastern
        - Traditional, Modernized, Indigenous, Folk-art
        - Festival-specific: Christmas, Diwali, Lunar New Year
    6. Target Audience / Perception: Tailors design to different demographic or user groups, for example
        - Kids, Teenagers, Adults, Seniors
        - Professional, Casual, Gamer, Traveler
        - Feminine, Masculine, Neutral
    7. Narrative / Symbolic Elements: Expresses abstract concepts or metaphorical meaning, for example
        - Symbolic: Power, Freedom, Love, Chaos, Harmony
        - Metaphorical: Rising sun for hope, A maze for confusion
        - Abstract shapes, Fractals, Mandalas
    Your answer must be in JSON format with key as string data type, and value as list of string (keywords), as the following JSON sample:
    ```
    {
        'Emotion / Mood' : [ANSWER HERE],
        'Purpose / Context': [ANSWER HERE],
        'Style / Visual Attributes': [ANSWER HERE],
        'Medium / Material': [ANSWER HERE],
        'Color / Lightning Characteristics': [ANSWER HERE],
        'Cultural / Regional Elements': [ANSWER HERE],
        'Target Audience / Perception': [ANSWER HERE],
        'Narrative / Symbolic Elements': [ANSWER HERE]
    }
    ```
"""