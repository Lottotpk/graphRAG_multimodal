PROMPT: str = """Inputs (always provided)
    - Query image: The image input to summarize

    Workflow (apply for each category in Categories)
    1) Summarize the Query image
    - Identify the summary of the Query image. The summary is a compact paragraph that includes all entities, places, vibes, and events from the image

    Output Requirements
    - Return strictly valid JSON  only (no prose, no markdown, no comments).
    - Use exactly these top-level keys and types for each category in Categories:
    {
        "summary": ""
    }

    Validation Checklist (must pass before you output)
    - JSON is strictly valid: no trailing commas; correct double quotes; correct the comma delimiter; beware of unterminated string; enclose the double quotes; enclose the brackets.
    - The summary's word count is lower than 400 words.
    - Reasons and evidence can be found from the Query image; No making up the answer"""

SYSTEM_PROMPT: str = """
    You are an expert image analyst. 
    Your task: summarize images showing entities, relations, and abstract feeling.
    Follow the workflow strictly and output JSON only.
"""
