from typing import Any, Tuple, List
import pandas as pd
import numpy as np


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

def extract_keywords_with_mmr(dream_text: str, keyword_embeddings: torch.Tensor, candidate_keywords: List[str], model:SentenceTransformer,
                              top_n_keywords: int = 10, diversity: float = 0.7) -> List[str]:
    """
    This function uses MMR (Maximal Marginal Relevance) to extract keywords from the dream text. It penalizes similar keywords to increase diversity.
    :param dream_text: Text description of the dream
    :param keyword_embeddings: Embeddings of the keywords
    :param candidate_keywords: list of keywords to choose from
    :param model: SentenceTransformer model
    :param top_n_keywords: number of keywords to extract
    :param diversity: diversity parameter to penalize similarity between keywords
    :return: List of keywords
    """


    if torch.cuda.is_available():
        dream_embeddings = model.encode(dream_text, convert_to_tensor=True).cuda()
        similarities = util.cos_sim(dream_embeddings, keyword_embeddings)[0].cpu().numpy()  # to convert to numpy

    else:
        dream_embeddings = model.encode(dream_text, convert_to_tensor=True).cpu()
        similarities = util.cos_sim(dream_embeddings, keyword_embeddings)[0].cpu().numpy()

    selected_keywords = []
    keyword_scores = similarities.copy()

    redundancy_scores = np.zeros(len(similarities))     # Initialize redundancy scores for all keywords
    index = None

    for _ in range(top_n_keywords):
        if not selected_keywords:
            index = similarities.argmax()

        else:       # Penalize similarity scores of overly similar keywords
            for selected_idx in selected_keywords:
                redundancy_scores = np.array([
                    np.mean(
                        [util.cos_sim(keyword_embeddings[index], keyword_embeddings[prev]).item() for prev in selected_keywords]
                    )
                    for index in range(len(candidate_keywords))
                ])

                adjusted_scores = similarities - (diversity * redundancy_scores)

                # Mask already-selected indices
                for idx in selected_keywords:
                    adjusted_scores[idx] = -np.inf  # assign very low score to already-selected keywords

                # select the index with the highest score
                index = adjusted_scores.argmax()

        selected_keywords.append(index)

    return [candidate_keywords[i] for i in selected_keywords]

def semantic_search_filter(dream_embeddings: torch.Tensor, keywords_embeddings: torch.Tensor, top_k: int = 100) \
        -> List[int]:
    """
    This function uses semantic search to create a subset of keywords that are relevant to the dream text.
    :param dream_embeddings: Embeddings of the dream text
    :param keywords_embeddings: Embeddings of the keywords
    :param top_k: number of keywords in the subset
    :return: List of indices of relevant keywords
    """
    results = util.semantic_search(dream_embeddings, keywords_embeddings, top_k=top_k)
    relevant_indices = [result['corpus_id'] for result in results[0]]
    return relevant_indices

def extract_keywords_with_semantic_search(dream_text: str, keywords_embeddings: torch.Tensor,
                                          candidate_keywords: List[str],
                                          model: SentenceTransformer, top_k_semantic: int=100, top_n_mmr=10,
                                          diversity=0.7) -> List[str]:
    """
    This function extracts keywords using semantic search and MMR. It first creates a subset of keywords
    using semantic search and then extracts keywords using MMR.
    :param dream_text: text description of the dream
    :param keywords_embeddings: Embeddings of keywords
    :param candidate_keywords: list of keywords to choose from
    :param model: sentence transformer model
    :param top_k_semantic: The size of the subset filtered by the semantic search.
    :param top_n_mmr: number of keywords to extract
    :param diversity: diversity parameter to penalize similarity between keywords
    :return: list of extracted keywords
    """
    # Encode dream text
    if torch.cuda.is_available():
        dream_embeddings = model.encode(dream_text, convert_to_tensor=True).cuda()

    else:
        dream_embeddings = model.encode(dream_text, convert_to_tensor=True).cpu()

    # step 1: semantic search to create a subset
    filtered_indices = semantic_search_filter(dream_embeddings, keywords_embeddings, top_k=top_k_semantic)
    filtered_embeddings = keywords_embeddings[filtered_indices]
    filtered_keywords = [candidate_keywords[i] for i in filtered_indices]

    # step 2: extract keywords using MMR
    return extract_keywords_with_mmr(dream_text=dream_text, keyword_embeddings=filtered_embeddings,
                                     candidate_keywords=filtered_keywords, model=model, top_n_keywords=top_n_mmr,
                                     diversity=diversity)

def extract_and_save_keywords_from_dataframes(dream_df, keywords_df, config_path:str='config.yaml') \
        -> pd.DataFrame:
    """
    This function gets a config file path and uses its parameters to
    read the dataframes and extract keywords from the dream text
    :param config_path: The path to the .yaml config file
    :return:
    """
    config = load_config(config_path)
    candidate_keywords = keywords_df[config['data']['keywords_column']].dropna().unique().tolist()

    # Load model
    model_name = config['model']['name']
    model = SentenceTransformer(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Encode keywords
    keywords_embeddings = model.encode(candidate_keywords, convert_to_tensor=True, device=model.device.type)

    # Extract keywords from ech dream
    top_n = config['model']['num_keywords']
    results = []

    for dream in dream_df[config['data']['dream_text_column']]:
        keywords = extract_keywords_from_text(dream, keywords_embeddings, candidate_keywords, model, top_n)
        results.append(";".join(keywords))

    dream_df[config['data']['keywords_column']] = results

    # Save results
    if '.tsv' in config['data']['dream_text_file_path']:
        dream_df.to_csv(config['data']['dream_text_file_path'], sep='\t', index=False)

    else:
        dream_df.to_csv(config['data']['dream_text_file_path'], index=False)

    return dream_df

def extract_and_save_keywords_with_semantic_search(config_path:str='config.yaml') \
        -> pd.DataFrame:
    """
    This function uses parameters from the config file
    to extract keywords from the dream text using semantic search.
    :param config_path: Path of the configuration file to fetch parameters from.
    :return: The dataset with the extracted keywords
    """
    config = load_config(config_path)
    dream_df, keywords_df = read_datasets(config)
    candidate_keywords = keywords_df[config['data']['keywords_column']].dropna().unique().tolist()

    # Load model
    model_name = config['model']['name']
    model = SentenceTransformer(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Encode keywords
    keywords_embeddings = model.encode(candidate_keywords, convert_to_tensor=True, device=model.device.type)

    # Extract keywords from each dream using semantic search
    top_k_semantic = config['model']['num_semantic']
    top_n_mmr = config['model']['num_keywords']
    diversity = config['model']['diversity']
    results = []

    for dream in dream_df[config['data']['dream_text_column']]:
        keywords = extract_keywords_with_semantic_search(dream, keywords_embeddings, candidate_keywords, model,
                                                         top_k_semantic, top_n_mmr, diversity)
        results.append(",".join(keywords))

    dream_df[config['data']['keywords_column']] = results

    # Save results
    if '.tsv' in config['data']['dream_text_file_path']:
        dream_df.to_csv(config['data']['dream_text_file_path'], sep='\t', index=False)

    else:
        dream_df.to_csv(config['data']['dream_text_file_path'], index=False)

    return dream_df

if __name__ == '__main__':
    extract_and_save_keywords_from_dataframes()