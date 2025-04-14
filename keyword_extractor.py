from typing import Any, Tuple, List
import pandas as pd
from yaml_parser import load_config
import torch
from sentence_transformers import SentenceTransformer, util


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

def extract_keywords_from_text(dream_text: str, keywords_embeddings: torch.Tensor, candidate_keywords: List[str],
                               model: SentenceTransformer, top_n_keywords: int = 10) -> List[str]:
    """
    This function extracts keywords from the dream text using the given pipeline
    (which contains a model and tokenizer)
    :param dream_text: Text description of the dream
    :param keywords_embeddings: Embeddings of the keywords
    :param candidate_keywords: List of keywords to choose from
    :param model: SentenceTransformer model
    :param top_n_keywords: Number of keywords to extract
    :return: A list of keywords
    """
    dream_embeddings = model.encode(dream_text, convert_to_tensor=True, device=model.device.type)
    scores = util.cos_sim(dream_embeddings, keywords_embeddings)[0]
    top_indices = torch.topk(scores, k=top_n_keywords).indices
    return [candidate_keywords[i] for i in top_indices]


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
    canndidate_keywords = keywords_df[config['data']['keywords_column']].dropna().unique().tolist()

    # Load model
    model_name = config['model']['name']
    model = SentenceTransformer(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Encode keywords
    keywords_embeddings = model.encode(canndidate_keywords, convert_to_tensor=True, device=model.device.type)

    # Extract keywords from ech dream
    top_n = config['model']['num_keywords']
    results = []

    for dream in dream_df[config['data']['dream_text_column']]:
        keywords = extract_keywords_from_text(dream, keywords_embeddings, canndidate_keywords, model, top_n)
        results.append(", ".join(keywords))

    dream_df[config['data']['keywords_column']] = results

    # Save results
    if '.tsv' in config['data']['dream_text_file_path']:
        dream_df.to_csv(config['data']['dream_text_file_path'], sep='\t', index=False)

    else:
        dream_df.to_csv(config['data']['dream_text_file_path'], index=False)


if __name__ == '__main__':
    extract_and_save_keywords_from_dataframes()