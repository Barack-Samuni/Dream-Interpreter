from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import ngrams
from scipy.special import softmax
from sentence_transformers import SentenceTransformer, util

from yaml_parser import load_config


def tokenize_columns(row:pd.DataFrame, dream_column: str='dream', interpretations_column='interpretation') \
        -> Tuple[List[str], List[str]]:
    """
    Splits and tokenizes the columns within a given DataFrame row into two separate lists.

    This function processes a specific row of a pandas DataFrame, extracting
    data from specific columns, tokenizing their values into lists of strings,
    and then returning two separate lists of tokens.

    :param row: A single row from a pandas DataFrame to be processed.
    :param dream_column: The name of the column containing the reference text.
    :param interpretations_column: The name of the column containing the interpretations.
    :return: A tuple containing two lists of tokens, where each list corresponds
             to the processed value of specific columns within the input row
             after tokenization.
    :return: A tuple containing two lists of tokens, reference and candidate.
    :rtype: Tuple[List[str], List[str]]
    """
    reference = word_tokenize(str(row[dream_column]).lower())
    candidate = word_tokenize(str(row[interpretations_column]).lower())
    return reference, candidate

def evaluate_bleu_on_df(df: pd.DataFrame, dream_column: str = 'dream', interpretation_column: str = 'interpretation',
                        output_column: str = 'BLEU') -> pd.DataFrame:
    """
    Evaluate dream interpretations using BLEU metric and add scores to DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame containing dreams and interpretations
        dream_column (str): Name of the column containing dreams
        interpretation_column (str): Name of the column containing interpretations
        output_column (str): Name of the column to store BLEU scores
    
    Returns:
        pd.DataFrame: DataFrame with added BLEU scores column
    """

    def calculate_bleu(row):
        reference, candidate = tokenize_columns(row, dream_column, interpretation_column)
        return sentence_bleu([reference], candidate)

    df[output_column] = df.apply(calculate_bleu, axis=1)
    return df


def evaluate_perplexity_on_df(df: pd.DataFrame, dream_column: str = 'dream',
                              interpretation_column: str = 'interpretation',
                              output_column: str = 'perplexity') -> pd.DataFrame:
    """
    Evaluate dream interpretations using perplexity metric and add scores to DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame containing dreams and interpretations
        dream_column (str): Name of the column containing dreams
        interpretation_column (str): Name of the column containing interpretations
        output_column (str): Name of the column to store perplexity scores
    
    Returns:
        pd.DataFrame: DataFrame with added perplexity scores column
    """

    def calculate_perplexity(row):
        reference, candidate = tokenize_columns(row, dream_column, interpretation_column)

        # Create vocabulary from both sequences
        vocab = list(set(reference + candidate))

        # Convert tokens to probability distribution
        ref_dist = np.zeros(len(vocab))
        for token in reference:
            ref_dist[vocab.index(token)] += 1
        ref_dist: np.ndarray = softmax(ref_dist)

        # Calculate cross entropy
        entropy = 0
        for token in candidate:
            if token in vocab:
                prob = ref_dist[vocab.index(token)]
                entropy -= np.log2(prob) if prob > 0 else 0

        # Calculate perplexity
        return 2 ** (entropy / len(candidate)) if len(candidate) > 0 else np.inf

    df[output_column] = df.apply(calculate_perplexity, axis=1)
    return df


def evaluate_rouge_on_df(df: pd.DataFrame, dream_column: str = 'dream', interpretation_column: str = 'interpretation',
                         output_column: str = 'ROUGE') -> pd.DataFrame:
    """
    Evaluate dream interpretations using the ROUGE metric and add scores to DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame containing dreams and interpretations
        dream_column (str): Name of the column containing dreams
        interpretation_column (str): Name of the column containing interpretations
        output_column (str): Name of the column to store ROUGE scores
    
    Returns:
        pd.DataFrame: DataFrame with added ROUGE scores column
    """

    def calculate_rouge(row):
        reference, candidate = tokenize_columns(row, dream_column, interpretation_column)

        # Calculate ROUGE-1 F1 score
        ref_ngrams = set(ngrams(reference, 1))
        cand_ngrams = set(ngrams(candidate, 1))

        if not ref_ngrams or not cand_ngrams:
            return 0.0

        # Calculate overlap
        overlap = len(ref_ngrams.intersection(cand_ngrams))

        # Calculate precision and recall
        precision = overlap / len(cand_ngrams) if len(cand_ngrams) > 0 else 0
        recall = overlap / len(ref_ngrams) if len(ref_ngrams) > 0 else 0

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f1

    df[output_column] = df.apply(calculate_rouge, axis=1)
    return df


def evaluate_bert_on_df(df: pd.DataFrame, config_path: str = 'config.yaml', dream_column: str = 'dream',
                        interpretation_column: str = 'interpretation', output_column: str = 'BERT') -> pd.DataFrame:
    """
    Evaluate dream interpretations using BERT-based similarity metric and add scores to DataFrame

    Args:
        df (pd.DataFrame): DataFrame containing dreams and interpretations
        config_path (str): Path to a configuration file
        dream_column (str): Name of the column containing dreams
        interpretation_column (str): Name of the column containing interpretations
        output_column (str): Name of the column to store BERT scores

    Returns:
        pd.DataFrame: DataFrame with added BERT scores column
    """
    # Load configuration and initialize model
    config = load_config(config_path)
    model = SentenceTransformer(config['model']['name'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def calculate_bert_score(row):
        # Get text from columns
        reference_text = str(row[dream_column])
        candidate_text = str(row[interpretation_column])

        # Calculate embeddings
        reference_embedding = model.encode(reference_text, convert_to_tensor=True)
        candidate_embedding = model.encode(candidate_text, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity = util.cos_sim(reference_embedding, candidate_embedding)
        return float(similarity[0][0])

    df[output_column] = df.apply(calculate_bert_score, axis=1)
    return df


def evaluate_dream_interpretations(df: pd.DataFrame, config_path: str = 'config.yaml',
                   dream_column: str = 'dream', interpretation_column: str = 'interpretation') -> pd.DataFrame:
    """
    Evaluate dream interpretations using multiple metrics (BLEU, perplexity, ROUGE, and BERT)

    Args:
        df (pd.DataFrame): DataFrame containing dreams and interpretations
        config_path (str): Path to a configuration file
        dream_column (str): Name of the column containing dreams
        interpretation_column (str): Name of the column containing interpretations

    Returns:
        pd.DataFrame: DataFrame with added evaluation metric scores
    """
    df = evaluate_bleu_on_df(df, dream_column, interpretation_column)
    df = evaluate_perplexity_on_df(df, dream_column, interpretation_column)
    df = evaluate_rouge_on_df(df, dream_column, interpretation_column)
    df = evaluate_bert_on_df(df, config_path, dream_column, interpretation_column)
    return df


