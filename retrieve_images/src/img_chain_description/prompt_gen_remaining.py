def ABSTRACT_PROMPT(img_sum: str) -> str:
    return """
    Inputs (always provided)
    - Query image: The image input to analyze
    - Image summary: The image summary as an additional relationship

    Image summary
    %s

    Workflow
    0) Understand the image summary
    - Explore details from text description as an additional context along with the Query image
    1) Capture entities
    - Identify objects presented in the Query image.
    - Add details of those objects. If it is a person, describe the details of that person in details.
    2) Analyse relations
    - Consider the Subject Placement, Object Placement, Inter-Subject & Inter-Object relations, Background Elements, Gaze & Interaction, Composition & Lightning of an image.

    The output format should be in JSON format, remember, just plain JSON text, no markdown.
    {
        "entities": "(the format should be \'entity: details\' separated by \'\\n\')",
        "relations": "(the format should be \'relation description\' separated by \'\\n\')",
    }

    Validation Checklist (must pass before you output)
    - JSON is strictly valid: no trailing commas; correct double quotes; correct the comma delimiter; beware of unterminated string; enclose the double quotes; enclose the brackets.
    - Each entity is straight-forward words of object; no adjective or number. They are separated clearly with \'\\n\'.
    - Each entity has its details attached to the entity.
    - Each relation truly shows connections between objects; no number. They are separated clearly with \'\\n\'.
    - Relation must be a single phrase of relation description.
    - Entities and relations can be found from the Query image; No making up the answer
    - Strict to the output format; output only JSON format; no thinking part
""" % (img_sum)

SYSTEM_PROMPT: str = """
    You are an expert image analyst. 
    Your task: identify entities and their relations in the images into structured JSON based on the image and its summary. 
    Follow the workflow strictly and output JSON only.
"""
