from typing import Any, Tuple, List
import pandas as pd
from yaml_parser import load_config
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.base import Pipeline


def read_datasets(config: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function reads the datasets from the config file and returns them as dataframes
    :param config: The loaded configuration from the .yaml file
    :return: 2 dataframes: one for the dream text and one for the keywords
    """
    if '.tsv' in config['data']['dream_text_file_path']:
        dream_df = pd.read_csv(config['data']['dream_text_file_path'], sep='\t')

    else:
        dream_df = pd.read_csv(config['data']['dream_text_file_path'])

    if '.tsv' in config['data']['keywords_file_path']:
        keywords_df = pd.read_csv(config['data']['keywords_file_path'], sep='\t')

    else:
        keywords_df = pd.read_csv(config['data']['keywords_file_path'])

    return dream_df, keywords_df

def extract_keywords_from_text(dream_text: str, candidate_keywords: List[str], pipe: Pipeline, top_n_keywords=10,
                               max_new_tokens=128) -> List[str]:
    """
    This function extracts keywords from the dream text using the given pipeline
    (which contains a model and tokenizer)
    :param dream_text: Text description of the dream
    :param candidate_keywords: List of keywords to choose from
    :param pipe: pipeline that contains a model and a tokenizer
    :param top_n_keywords: How many keywords to extract?
    :param max_new_tokens: Maximum number of tokens to generate
    :return: A list of keywords
    """
    prompt = (f"Extract {top_n_keywords} keywords from the following dream using only this keywords list:\n "
              f"{','.join(candidate_keywords)}\nDream:\n{dream_text}\nKeywords:")
    output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]['generated_text']
    extracted_keywords = output.split('Keywords:')[-1].strip().split(',')[:top_n_keywords]
    return [kw.strip() for kw in extracted_keywords if kw.strip() in candidate_keywords]

def extract_and_save_keywords_from_dataframes(config_path:str='config.yaml') \
        -> None:
    """
    This function gets a config file path and uses its parameters to
    read the dataframes and extract keywords from the dream text
    :param config_path: The path to the .yaml config file
    :return:
    """
    config = load_config(config_path)
    dream_df, keywords_df = read_datasets(config)

    # model and tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    device=config['model']['device'])

    candidate_keywords = keywords_df[config['data']['keywords_column']].tolist()
    top_n_keywords = config['model']['num_keywords']

    keywords_col = []

    for dream_text in dream_df[config['data']['dream_text_column']]:
        keywords = extract_keywords_from_text(dream_text, candidate_keywords, pipe,
                                              top_n_keywords=top_n_keywords)
        keywords_col.append(keywords)

    dream_df[config['data']['keywords_column']] = keywords_col

    if '.tsv' in config['data']['dream_text_file_path']:
        dream_df.to_csv(config['data']['dream_text_file_path'], sep='\t', index=False)

    else:
        dream_df.to_csv(config['data']['dream_text_file_path'], index=False)
