# Dream-Interpreter

## Overview
This project aims to interpret a given text dream using the common Freudian method. This is the flowchart of our process:

**Keyword Extraction -> Map Interpretations based on keywords -> Use LLM to summarize dream interpretations**

### Keyword extraction
This is the process of extracting meaningful keywords out of the dream text - keywords that hold the main story of the dream.

### Map Interpretations based on keywords
After extracting the keywords - each keyword is mapped to its Freudian interpretation

### Use LLM to summarize dream interpretations
A prompt is given to a pretrained LLM in order to summarize the fetched interpretations.

Full summary of the project can be found in [Dream-interpreter.ipynb](Dream-Interpreter.ipynb)
