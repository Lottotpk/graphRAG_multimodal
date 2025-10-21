EMOTION_MOOD = """
    Emotion/Mood: The dominant visual atmosphere of an image. What feeling does this image represents? The list of keywords are:
    - Joyful: Evokes a feeling of great happiness and delight.
    - Vibrant: Full of energy, excitement, and life.
    - Hopeful: Instills a feeling of expectation and desire for a certain thing to happen.
    - Romantic: Suggests feelings of love, tenderness, and intimacy.
    - Nostalgic: A sentimental longing or wistful affection for the past.
    - Uplifting: Inspiring happiness, optimism, or hope.
    - Whimsical: Playfully quaint or fanciful, especially in an appealing and amusing way.
    - Melancholic: A feeling of pensive sadness, typically with no obvious cause.
    - Somber: Dark or dull in color or tone; gloomy.
    - Tense: Evoking a sense of suspense, anxiety, or unease.
    - Bleak: Lacking warmth, life, and cheer. Desolate, windswept landscapes or scenes of poverty and despair can be described as bleak.
    - Ominous: Giving the impression that something bad or unpleasant is going to happen; threatening.
    - Haunting: Poignantly or persistently evocative; difficult to ignore or forget.
    - Gloomy: Dark or poorly lit, especially so as to appear depressing or frightening.
    - Aggressive: Ready or likely to attack or confront.
    - Miserable: Wretchedly unhappy or uncomfortable.
    - Mysterious: Difficult or impossible to understand, explain, or identify.
    - Intriguing: Arousing one's curiosity or interest; fascinating.
    - Surreal: Having the qualities of a dream; bizarre.
    - Dramatic: Exciting or impressive.
    - Thought-provoking: Stimulating careful consideration or reflection.
    - Pensive: Engaged in, involving, or reflecting deep or serious thought.
    - Intense: Of extreme force, degree, or strength.
    - Subtle: So delicate or precise as to be difficult to analyze or describe.
    - Dynamic: Characterized by constant change, activity, or progress.
    - Raw: Realistic and unconcealed; stark.
    - Ethereal: Extremely delicate and light in a way that seems too perfect for this world.
"""

PURPOSE_CONTEXT = """
    Purpose/Context: The intended use case or application scenario for the image. You may need to provide more additional keywords on top of the following list. The example list of keywords are:
    - Movie/Event Poster: Promotional image for a film or public event.
    - Music Festival Poster: Visual branding for a music event.
    - Gym Poster: Motivational or promotional imagery for a fitness center.
    - Digital Ad Banner: An online advertisement for a website.
    - Social Media Post: Promotional content for platforms like Instagram or Facebook.
    - Email Newsletter Graphic: Visuals for a marketing email.
    - Trade Show Booth Graphic: Large-scale imagery for a promotional display.
    - Product Packaging: The design on a retail item's container.
    - Restaurant Menu: Images used to showcase food items.
    - E-commerce Catalog: Clean product shots for an online store.
    - Book Cover: The front-facing art for a novel or non-fiction work.
    - Fashion Magazine Cover: A high-impact portrait for a fashion publication.
    - Magazine Spread: An image illustrating an article.
    - Album Art: The visual identity for a musical album.
    - Blog Post Header: The representative image for an online article.
    - Video Thumbnail: A preview image to encourage clicks on a video.
    - Textbook Illustration: A diagram or photo explaining an academic concept.
    - Infographic: A visual representation of data.
    - Gallery Wall Art: A fine art piece for exhibition and sale.
    - NFT Artwork: A digital image registered as a unique collectible.
    - Living-room Wall Art: A print or photo meant to decorate a home.
    - Kids’ Room Decoration: Playful or themed art for a child's space.
    - Mobile Phone Wallpaper: An image formatted for a device background.
    - Custom T-shirt Graphic: A design intended to be printed on clothing.
    - Wedding Invitation: A decorative image for a formal announcement.
    - Greeting Card: An image for a specific occasion.
    - Family Photo Album: A personal image for private memory keeping.
    - Event Backdrop: A large-scale background for photo opportunities.
    - Instagram Story: A temporary, casual post.
    - Profile Picture: An avatar for a social media account.
    - Real Estate Listing: A photo to showcase a property for sale.
    - Architect's Portfolio: Images documenting a completed building.
    - Medical Diagram: An illustration for a medical journal or textbook.
    - Forensic Evidence Photo: A photo for legal documentation.
    - Others (specify the keyword): If the appropriate word is not belong to the list above.
"""

STYLE_VISUAL_ATTR = """
    Style/Visual Attributes: The artistic or desgin style of an image. The list of keywords are:
    - Minimalist: Characterized by extreme simplicity and sparseness. These images use a limited number of elements to create a clean and uncluttered look.
    - Maximalist: The opposite of minimalism, this style embraces abundance, complexity, and vibrant colors and patterns.
    - Geometric: Relies on shapes like circles, squares, and triangles to create a sense of order and structure.
    - Symmetrical: Elements are arranged in a way that creates a sense of balance and harmony.
    - Asymmetrical: Lacks symmetry but achieves balance through the careful arrangement of elements of varying visual weight.
    - Dynamic: Conveys a sense of movement and energy, often through the use of leading lines and diagonal compositions.
    - Monochromatic: Uses a single color in varying shades and tints to create a harmonious and unified image.
    - Vibrant: Characterized by bright, vivid, and saturated colors that create a sense of energy and excitement.
    - Muted: Uses a desaturated color palette to create a subtle and understated mood.
    - High-Contrast: Features a strong difference between the light and dark areas of the image, creating a dramatic and bold look.
    - Low-Contrast: Has a narrow range of tones, resulting in a more subtle and gentle image.
    - Ethereal: Possesses a delicate and light quality that seems otherworldly, often achieved through soft, diffused lighting.
    - Abstract: Does not attempt to represent external reality, instead using shapes, forms, colors, and textures to achieve its effect.
    - Surreal: Features dream-like and bizarre imagery that often defies logical explanation.
    - Retro: Evokes the styles and trends of the recent past, particularly the 1950s, 60s, and 70s.
    - Vintage: Has the appearance of belonging to a past era, often characterized by specific color palettes and textures.
    - Grunge: A style characterized by distressed textures, dark colors, and a raw, edgy aesthetic.
    - Photorealistic: Aims to reproduce a high-resolution photograph with such detail and precision that it appears to be a real photo.
    - Grainy: Has a textured appearance reminiscent of film grain, which can add a nostalgic or raw feel.
    - Crisp: Appears sharp, clear, and well-defined, with a high level of detail.
    - Glossy: Has a smooth, shiny, and reflective surface.
    - Matte: Features a non-reflective, dull finish that can create a more subdued and sophisticated look.
    - Textured: Has a tangible surface quality that can be either physical or visually implied.
    - Organic: Utilizes natural shapes, lines, and textures, often inspired by the forms found in nature.
    - Striking: Makes a strong and immediate impression due to its visual power or originality.
    - Bold: Features strong lines, shapes, and colors that command attention.
    - Graphic: Has a clear, simplified, and often two-dimensional quality, similar to a drawing or print.
    - Stylized: Depicts subjects in a non-realistic manner, emphasizing artistic style over literal representation.
    - Clean: Appears uncluttered, with sharp lines and a simple, well-organized composition.
    - Ornate: Richly decorated with complex patterns and details.
"""

MEDIUM_MATERIAL = """
    Medium/Material: The creative medium or material representaion. The list of keywords are:
    - Oil-based: Characterized by rich, deep colors and a slow drying time that allows for extensive blending. Oil paintings often have a visible texture from the brushstrokes.
    - Watercolor: Known for its transparency and luminosity, creating soft, blended washes of color where the white of the paper often shines through.
    - Acrylic: A versatile, water-based paint that can mimic the effects of both watercolor and oil.
    - Ink-drawn: Created using pens or brushes with ink, this style is characterized by strong lines, precision, and high contrast.
    - Charcoal: A dry medium that produces deep blacks and a range of grays.
    - Pastel: A medium with a chalk-like consistency, known for its vibrant, rich colors and soft, blendable texture.
    - Graphite: The medium found in pencils, used for drawing and sketching.
    - Printmade: An image created by transferring ink from a prepared surface (like a woodblock or metal plate) to paper.
    - Collage: An artwork made from an assemblage of different forms, thus creating a new whole. Collages can include paper, photographs, fabric, and other found objects, glued to a piece of paper or canvas.
    - Sculptural: Refers to three-dimensional artwork created by shaping or combining materials like clay, stone, metal, or wood.
    - Textile: Art that is created using fabrics, fibers, and yarn.
    - Vector: Graphics created using mathematical equations, resulting in clean lines, sharp edges, and the ability to be scaled to any size without losing quality. Often used for logos and illustrations.
    - Raster: Images made up of a grid of pixels, which is characteristic of digital paintings and photographs.
    - 3D-Rendered: Computer-generated imagery that creates three-dimensional objects and scenes with realistic lighting, shadows, and textures.
    - Pixelated: A deliberately low-resolution style where individual pixels are visible, often to evoke a retro, video-game aesthetic.
    - Digital Collage: The technique of combining multiple digital images to create a single, new composition using software like Photoshop.
    - Algorithmic/AI-Generated: Images created with the assistance of artificial intelligence algorithms, often based on text prompts.
    - Monochrome/Black-and-White: An image that contains only tones of a single color, most commonly black, white, and shades of gray.
    - Sepia: A photograph that has been toned to have a brownish hue, often used to give the image a vintage or antique feel.
    - Double Exposure: A technique where two different images are layered on top of each other to create a single, often surreal, composition.
    - Long Exposure: Created by using a long-duration shutter speed to capture stationary elements sharply while blurring moving elements, often used for light trails or smoothing water.
"""

COLOR_LIGHTNING_CHAR = """
    Color/Lightning Characteristics: The color palette and the lightning style. The list of keywords are:
    - Warm: Dominated by colors like red, orange, and yellow, creating a feeling of energy, comfort, or aggression.
    - Cool: Dominated by colors like blue, green, and purple, often creating a feeling of calm, serenity, or melancholy.
    - Vibrant: Characterized by bright, highly saturated colors that give the image a lively and intense appearance.
    - Muted: Uses low-saturation or desaturated colors, resulting in a soft, subtle, and often understated or sophisticated look.
    - Earth-toned: A palette dominated by natural colors like browns, greens, and ochres, evoking a grounded, rustic, or organic feel.
    - Pastel: Features pale, soft colors with high lightness and low saturation, creating a delicate, gentle, and often whimsical atmosphere.
    - Monochromatic: Uses a single color in various shades and tints, creating a unified and cohesive visual experience.
    - Complementary: Utilizes colors opposite each other on the color wheel (e.g., blue and orange) to create high visual contrast and tension.
    - Analogous: Uses colors that are next to each other on the color wheel (e.g., blue, blue-green, green) to create a harmonious and comfortable mood.
    - Desaturated: Colors have had their intensity reduced, giving the image a washed-out, vintage, or emotional feel.
    - Over-saturated: Colors are unnaturally vivid and intense, often to create a dramatic, hyper-real, or pop-art effect.
    - High-key: The majority of the image is bright and light-toned, with minimal shadows, creating an airy, optimistic, or clean mood.
    - Low-key: The majority of the image is dark and shadowed, with only selective areas lit, creating a dramatic, mysterious, or somber atmosphere.
    - Harsh: Characterized by strong, undiffused light that creates sharp, distinct shadows and high contrast.
    - Soft: Created by diffused light that minimizes shadows and transitions smoothly between light and dark areas, giving a gentle, flattering, or dreamy feel.
    - Dramatic: Uses extreme contrast or directional light (like spotlights) to emphasize specific elements and heighten the emotional impact.
    - Backlit: The main light source is positioned behind the subject, often creating a glowing outline or silhouette.
    - Flat: Achieved when the light source is directly in front of the subject, minimizing shadows and reducing the appearance of texture or depth.
    - Diffused: Light is scattered and spread out evenly, often by clouds or a softbox, eliminating harsh shadows and creating a uniform look.
    - Ambient: The natural, available light of a scene, often resulting in a realistic and environmental look.
    - Rim-lit: Light hits the subject from behind or the side, creating a bright line (a rim) around the edges.
    - Glowy: Characterized by a soft, warm light that seems to emanate from within the image, creating a magical or ethereal quality.
    - Silhouetted: The subject is seen as a dark shape against a brighter background, resulting from a strong backlight.
    - Chiaroscuro: An Italian term referring to the dramatic use of strong contrast between light and dark, often used in fine art to create a sense of volume and drama.
    - Spotlit: Only a small, specific area of the image is highlighted, drawing intense focus and leaving the rest in deep shadow.
"""

CULTURAL_REGIONAL = """
    Cultural/Regional Elements: Any elements that connect the image to the specific culture, region, religion, or tradition. The list of keywords are:
    - Traditional: Adhering to the long-established customs, beliefs, and artistic practices of a particular culture.
    - Indigenous: Originating from the native inhabitants of a specific region, often reflecting a deep connection to the land and ancestral heritage.
    - Folk Art: Pertaining to the art of the common people, often characterized by a naive style and a focus on utilitarian or decorative objects.
    - Tribal: Relating to the distinct artistic styles and motifs of a particular tribe or ethnic group.
    - Ceremonial: Associated with religious or social rituals and ceremonies, often featuring symbolic imagery and specific materials.
    - Vernacular: Reflecting the everyday life, architecture, and design of a particular locality or region.
    - Mythological: Depicting scenes, characters, or symbols from myths, legends, and folklore specific to a culture.
    - Historical: Representing a specific period in a region's history, often depicting significant events, figures, or daily life from that era.
    - Afrocentric (specify country, if possible): Emphasizing or celebrating African culture and the African diaspora.
    - Pan-African (specify country, if possible): Relating to a sense of unity and shared identity among all people of African descent.
    - Sub-Saharan (specify country, if possible): Pertaining to the diverse cultures and artistic traditions of the regions south of the Sahara Desert.
    - East Asian (specify country, if possible): Encompassing the artistic traditions of China, Japan, and Korea, often characterized by calligraphy, ink wash painting, and a focus on nature.
    - South Asian (specify country, if possible): Referring to the arts of the Indian subcontinent, including intricate patterns, vibrant colors, and religious iconography from Hinduism, Buddhism, and Islam.
    - Southeast Asian (specify country, if possible): Covering the diverse artistic expressions of countries like Thailand, Indonesia, and Vietnam, often featuring rich textiles, shadow puppetry, and ornate temple art.
    - Central Asian (specify country, if possible): Reflecting the nomadic and Silk Road cultures of the region, often seen in textiles, metalwork, and manuscript illumination.
    - Classical: Referring to the art of ancient Greece and Rome, characterized by harmony, balance, and idealized forms.
    - Medieval: Pertaining to the art of the Middle Ages in Europe, often religious in nature and including styles like Gothic and Romanesque.
    - Renaissance: Marking the revival of classical art and learning in Europe, with a focus on humanism, realism, and perspective.
    - Baroque: A highly ornate and dramatic style that emerged in the 17th century, characterized by grandeur and emotional intensity.
    - Nordic/Scandinavian: Reflecting the design sensibilities of Northern Europe, often characterized by minimalism, functionality, and a connection to nature.
    - Pre-Columbian: Referring to the art and cultures of the Americas before the arrival of Christopher Columbus, including those of the Aztec, Maya, and Inca civilizations.
    - Latin American (specify country, if possible): A broad term for the diverse artistic traditions of Central and South America, often characterized by vibrant colors, magical realism, and a blend of indigenous and colonial influences.
    - North American (specify country, if possible): Encompassing a wide range of styles, from Native American art to the various movements that have shaped the art of the United States and Canada.
    - Religion (specify which religion): Buddhist, Christianity, Islamic, etc.
    - Persian: Referring to the rich artistic heritage of Iran, known for its elaborate carpets, miniature paintings, and decorative tilework.
    - Ottoman: Pertaining to the art of the Ottoman Empire, which blended Islamic, Persian, and Byzantine influences.
    - Aboriginal: The art of the indigenous people of Australia, often featuring dot painting, symbolic imagery, and a deep connection to the "Dreamtime."
    - Polynesian: Encompassing the arts of the islands of the central and southern Pacific, including tattooing, wood carving, and textile arts like tapa cloth.
"""

TARGET_AUDIENCE_PERCEPTION = """
    Target Audience/Perception: The specific demographic or user group the image is tailored to. The list of keywords are:
    - Child-friendly: Designed for young children, often featuring bright colors, simple shapes, and playful themes.
    - Teen-oriented: Tailored to the interests and aesthetics of teenagers, which can include trends, social themes, and edgier styles.
    - Youthful: Aimed at young adults, but also appealing to a general mindset that values energy, modernity, and new ideas.
    - Mature: Intended for an older, more experienced audience, often focusing on themes of sophistication, experience, or nostalgia.
    - Family-oriented: Created to be appropriate and appealing for all ages, often depicting wholesome scenes and universal themes of togetherness.
    - Senior-focused: Specifically designed for the elderly, addressing their interests, needs, and life experiences.
    - Business-oriented: For a professional audience, typically used in business reports, presentations, and marketing materials. The tone is often formal and polished.
    - Academic: Intended for use in educational contexts, research papers, and scholarly publications. The focus is on clarity and conveying information.
    - Creative-focused: Appeals to artists, designers, musicians, and other creative professionals, often showcasing innovation, style, and artistic expression.
    - Tech-savvy: For an audience interested in technology, innovation, and digital culture. Imagery may be futuristic, sleek, or data-driven.
    - Enthusiast-focused: Tailored to hobbyists or individuals with a passion for a specific niche, such as gaming, classic cars, or gardening.
    - Athletic: Designed for athletes and sports fans, often depicting action, competition, and physical fitness.
    - Luxury: Aimed at a high-income demographic, emphasizing exclusivity, quality, and prestige.
    - Economical: Appeals to consumers looking for value, affordability, and practicality.
    - Mainstream: Designed to have broad appeal to the general public, using relatable and widely understood themes.
    - Niche (specify what kind of niche): Created for a small, specific group with specialized interests that are not catered to by mainstream media.
    - Alternative: Targets audiences who operate outside of mainstream culture, often embracing unconventional or rebellious ideas.
    - Aspirational: Designed to appeal to an audience's desire for a better life, often showcasing success, luxury, or personal growth.
    - Health-conscious: For individuals focused on physical and mental well-being, nutrition, and fitness.
    - Sustainable: Appeals to an audience concerned with environmentalism, ethical consumption, and nature.
    - Inspirational: Aims to evoke feelings of hope, encouragement, and ambition.
    - Comedic: Intended to be funny, lighthearted, and entertaining.
    - Provocative: Designed to challenge conventions, spark debate, and grab attention through controversial or surprising content.
    - [FIELD] Expert: Positioned to be perceived as credible, trustworthy, and knowledgeable on that specific FIELD, often used in technical or professional contexts. (FIELD is added freely by your thought)
    - Comforting: Creates a sense of safety, calm, and trust, often used in healthcare or financial services.
    - Relatable: Designed to feel friendly, familiar, and easy to connect with, fostering a sense of authenticity.
    - Sensational: Intended to shock, excite, or arouse strong curiosity, often used in news headlines or entertainment promotion.
    - Feminine: Tailored to aesthetics and interests traditionally associated with women.
    - Masculine: Tailored to aesthetics and interests traditionally associated with men.
    - Gender-neutral: Intentionally avoids gender-specific cues to appeal to all genders equally.
"""

NARRATIVE_SYMBOLIC = """
    Narrative/Symbolic Elements: Any abstract concepts, or metaphorical meanings represented. What does the image symbolize? The list of keywords are:
    - Power: The ability to influence or control the behavior of others. Visually represented by thrones, crowns, muscular figures, or a high vantage point.
    - Freedom: The state of being unrestrained. Often symbolized by open skies, birds in flight, broken chains, or vast, open landscapes.
    - Justice: The concept of moral rightness and fairness. Represented by scales, gavels, or balanced compositions.
    - Conflict: A struggle or clash between opposing forces. Depicted through battle scenes, arguments, or stark visual contrasts.
    - Oppression: The exercise of authority in a cruel or unjust manner. Symbolized by cages, chains, heavy burdens, or looming, shadowy figures.
    - Equality: The state of being equal, especially in status, rights, and opportunities. Can be shown through symmetrical compositions or figures at the same level.
    - Revolution: A forcible overthrow of a social order in favor of a new system. Represented by images of protest, toppled statues, or rising flags.
    - Community: A feeling of fellowship with others as a result of sharing common attitudes, interests, and goals. Depicted through images of groups, gatherings, or interconnected shapes.
    - Hierarchy: A system in which people are ranked one above the other according to status or authority. Visually shown through pyramids, ladders, or figures placed at different vertical levels.
    - Propaganda: Information, especially of a biased or misleading nature, used to promote a political cause or point of view.
    - Hope: A feeling of expectation and desire for a certain thing to happen. Metaphorically represented by a rising sun, a single light in darkness, a budding plant, or a rainbow.
    - Despair: The complete loss or absence of hope. Symbolized by darkness, desolate landscapes, or figures in postures of defeat.
    - Love: An intense feeling of deep affection. Can be romantic, platonic, or familial, often symbolized by hearts, entwined figures, or warm light.
    - Loneliness: A feeling of solitude or separation. Depicted by a single figure in a vast space, shadows, or empty rooms.
    - Grief: The sorrow experienced after a loss. Often conveyed through images of wilting flowers, rain, or objects left behind.
    - Growth: The process of changing or developing. Symbolized by butterflies, shedding skin, seasonal changes, or a journey from darkness to light.
    - Innocence: The state of being pure and free from guilt. Often represented by children, lambs, or the color white.
    - Corruption: The process of impairment or moral deterioration. Visually represented by rot, ruins, rust, or distorted figures.
    - Mortality: The state of being subject to death. Classically symbolized by skulls (memento mori), hourglasses, or extinguished candles.
    - Identity: The sense of self. Explored through themes of masks, mirrors, reflections, or fragmented portraits.
    - Chaos: A state of complete disorder and confusion. Depicted through frenetic, disorganized compositions, and clashing colors.
    - Order: The arrangement of people or things in a systematic way. Shown through symmetry, geometric patterns, and clean lines.
    - Creation: The act of bringing something into existence. Symbolized by light, birth, or hands shaping clay.
    - Destruction: The action of damaging something so that it no longer exists. Represented by fire, explosions, or crumbling structures.
    - Time: The indefinite continued progress of existence. Symbolized by clocks, hourglasses, rivers, or the changing seasons.
    - Destiny: The idea that events are predetermined. Represented by threads, pathways, or celestial alignments.
    - Truth: The quality of being in accordance with fact or reality. Metaphorically represented by light, a clear reflection, or an open eye.
    - Deception: A false idea or belief. Symbolized by masks, smoke and mirrors, or optical tricks.
    - Knowledge: Facts, information, and skills acquired through experience or education. Symbolized by books, keys, or light.
    - Spirituality: A connection to something larger than oneself. Represented by light from above, ascending figures, mandalas, or eyes looking upward.
    - Journey: A road, path, or river can symbolize the course of a life or a personal quest.
    - Limitation: Walls, fences, or cages can represent physical, social, or psychological barriers.
    - Opportunity: An open door or window often stands for a new chance or a potential future.
    - Heritage: Roots of a tree can symbolize ancestry, tradition, and a connection to the past.
    - Consciousness: The surface of water can represent the conscious mind, while the depths symbolize the unconscious.
"""

EXAMPLE_IMAGE_ANSWER = {
    "Emotion/Mood": [
        {
            "keyword": "Dramatic",
            "reason": "The powerful contrast between the dark landscape and the brilliant, explosive sky creates a theatrical visual tension." 
        },
        {
            "keyword": "Spiritual",
            "reason": "The heavens are depicted as an immense, divine force, with the church steeple reaching towards this cosmic energy."
        },
        {
            "keyword": "Turbulent",
            "reason": "The swirling, chaotic brushstrokes of the sky and clouds suggest inner turmoil and restless, powerful motion."
        }
    ],
    "Purpose/Context": [
        {
            "keyword": "Fine Art",
            "reason": "It is a non-commercial, expressive piece made for aesthetic contemplation, intended to convey the artist's personal vision."
        }
    ],
    "Style/Visual Attributes": [
        {
            "keyword": "Post-Impressionist",
            "reason": "It uses vivid color and bold strokes not just for light, but to express emotion and symbolism."
        },
        {
            "keyword": "Stylized",
            "reason": "Forms like the swirling sky and flame-like tree are intentionally exaggerated and not meant to be realistic."
        }
    ],
    "Medium/Material": [
        {
            "keyword": "Oil-based",
            "reason": "The rich color depth and the ability to hold thick, textured brushstrokes are characteristic of oil paint."
        }
    ],
    "Color/Lightning Characteristics": [
        {
            "keyword": "Cool",
            "reason": "The composition is dominated by a palette of deep blues, indigos, and greens in the sky and landscape."
        },
        {
            "keyword": "High-Contrast",
            "reason": "The bright, glowing yellows of the stars and moon stand out sharply against the dark blues of the night."
        },
        {
            "keyword": "Luminous",
            "reason": "The celestial bodies appear to generate their own brilliant light, casting a surreal glow over the scene."
        }
    ],
    "Cultural/Regional Elements": [
        {
            "keyword": "France",
            "reason": "The village's architecture and the artwork's style are products of late 19th-century provincial France and its art movements."
        }
    ],
    "Target Audience/Perception": [
        {
            "keyword": "Art-Enthusiast",
            "reason": "Appreciating its historical context and stylistic innovations appeals to an audience knowledgeable about art."
        },
        {
            "keyword": "Comtemplative",
            "reason": "The deep symbolism and emotional weight invite viewers to reflect on themes of life, death, and spirituality."
        }
    ],
    "Narrative/Symbolic Elements": [
        {
            "keyword": "Mortality",
            "reason": "The dark, imposing cypress tree in the foreground is a traditional European symbol of death and mourning."
        },
        {
            "keyword": "Transcendence",
            "reason": "The enormous, powerful sky dwarfs the quiet village, suggesting a reality or spiritual plane beyond human life."
        },
        {
            "keyword": "Hope",
            "reason": "The brilliantly shining stars serve as points of light and guidance in the overwhelming darkness of the night."
        }
    ],
}

CATEGORY = [EMOTION_MOOD, PURPOSE_CONTEXT, STYLE_VISUAL_ATTR, MEDIUM_MATERIAL, COLOR_LIGHTNING_CHAR, CULTURAL_REGIONAL, TARGET_AUDIENCE_PERCEPTION, NARRATIVE_SYMBOLIC]
CATEGORY_NAME = ["Emotion/Mood", "Purpose/Context", "Style/Visual Attributes", "Medium/Material", "Color/Lightning Characteristics", "Cultural/Regional Elements", "Target Audience/Perception", "Narrative/Symbolic Elements"]

def ABSTRACT_PROMPT(cate_num: range, img_sum: str) -> str:
    cate_prompt = ""
    example = "{\n"
    output = "{\n"
    for i, num in enumerate(cate_num):
        cate_exam = list(EXAMPLE_IMAGE_ANSWER.items())[num]
        cate_prompt += f"{i+1}" + ") " + CATEGORY[num] + '\n'
        example += "\t\"" + cate_exam[0] + '\": [\n'
        output += "\t\"" + CATEGORY_NAME[num] + """\" : [{"keyword": "", "reason": "", "confidence": <number between 0.0 and 1.0>}],\n"""
        for key in cate_exam[1]:
            example += "\t" + str(key) + ',\n'
        example += "\t],\n"
    example += "}"
    output += "}"
    return """
    Inputs (always provided)
    - Query image: The image input to analyze
    - Example answer: The correct answer of the example image (Starry Night)
    - Image summary: The image summary as an additional relationship
    - Categories: description aspect to consider
    - Example keywords: possibly keyword from each category; you can use keywords other than the example
    - Example explanation: definitions of each example, which can be used as references

    Example answer
    %s

    Image summary
    %s

    Categories
    %s
    
    Workflow (apply for each category in Categories)
    0) Understand the image summary
    - Explore details from text description as an additional context along with the Query image
    1) Analyze the Query image
    - Identify up to 3 relevant keywords according to the category based on the evidence from the Query image, such as the main characters' expression, the place, and the Query image vibe.
    2) Justify each keyword
    - Make a compact reason for each keyword on why this keyword is chosen over other keywords that sound similar and why not the other keywords on the opposite. The clues are included in the reason. The length is not more than 20 words.
    3) Evaluate grounding
    - Analyzing grounding behind the reasons: whether each reason from keyword can be linked to the visual evidence in the Query image.
    - No making up explanation or implication: For example, "Gallery Wall Art" is for arts not the photography, so the reason behind it has to be synchronus and grounding.
    - Mentally simulate the situation using the image for the "Purpose/Context" category whether the keyword intuitively make sense in that specific context or not.
    4) Summarize & Score
    - Write keywords and its reason after evaluation
    - Output a confidence score (0.0–1.0), where 1.0 = highly relevant, 0.0 = irrelevant reflecting: the relevance between keyword and reason and Query image, the mental simulation, and the grounding candidates observed from the Query image.
    5) Selection
    - Strictly remove all keywords with confidence score lower than 0.7 even though that category remains zero keywords.
    - Category with zero keywords can simply use [] (the empty list in response)

    Output Requirements
    - Return strictly valid JSON only (no prose, no markdown, no comments).
    - Use exactly these top-level keys and types for each category in Categories:
    %s

    Validation Checklist (must pass before you output)
    - JSON is strictly valid: no trailing commas; correct double quotes; confidence is a number; correct the comma delimiter; beware of unterminated string; enclose the double quotes; enclose the brackets.
    - Each "keyword" contains 1-2 words, 1 reason, and 1 confidence number.
    - Each "reason" is a single crispy sentence with no more than 20 words.
    - Avoid generic keyword; recalculate confidence of the keyword with highly repeated usages across every image
    - At most 3 keywords per "category", no more than this; can use less than this
    - Reasons and evidence can be found from the Query image; No making up the answer.
    - Using the example only as a reference, generate the output for the image.
    - All the keywords with confidence score lower than 0.7 are excluded from the output.
""" % (example, img_sum, cate_prompt, output)

SYSTEM_PROMPT: str = """
    You are an expert image analyst. 
    Your task: classify images into structured JSON with keywords, reasons, and confidence on categories based on the image and its summary. 
    Follow the workflow strictly and output JSON only.
"""
