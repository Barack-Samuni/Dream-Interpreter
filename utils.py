
def release_all_gpu_memory(additional_objects=[]):
    import gc
    import torch

    # Delete model objects (make sure they're declared global or passed)
    globals_to_clear = ["model", "tokenizer", "text2text_generator"] + additional_objects
    print(globals_to_clear)
    gks = list(globals().keys())
    # print(gks)
    for name in globals_to_clear:
        if name in gks:
            print("clearing ", name)
            del globals()[name]

    gc.collect()

    if torch.cuda.is_available():
        print("clearing cuda cache")
        torch.cuda.empty_cache()
        print("clearing ipc cache")
        torch.cuda.ipc_collect()

    print("✅ All GPU memory cleared.")

def globals_snapshot():
    import pandas as pd
    gks = list(globals().keys())
    vars = []
    for k in gks:
        v = globals()[k]
        vars.append({"key": k, "var": str(v) , "type": str(type(v))})
    tps = pd.DataFrame(vars)
    return tps

def save_df_as_pretty_html(df, filename="output.html"):
    # Convert newlines to <br> for HTML
    df_html_ready = df.copy()
    for col in df_html_ready.columns:
        df_html_ready[col] = df_html_ready[col].astype(str).str.replace('\n', '<br>', regex=False)

    # Generate styled HTML
    html = df_html_ready.to_html(
        escape=False,  # Needed to render <br>
        index=False,
        border=0,
        classes="styled-table"
    )

    # Add CSS styling
    style = """
    <style>
    .styled-table {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 16px;
        font-family: Arial, sans-serif;
        width: 100%;
        table-layout: auto; /* ✅ Let browser fit naturally */
    }
    .styled-table th, .styled-table td {
        border: 1px solid #dddddd;
        padding: 10px;
        vertical-align: top;
        text-align: left;
        overflow-wrap: break-word; /* ✅ Break inside words */
        white-space: pre-wrap; /* ✅ Honor \\n linebreaks */
    }
    .styled-table td {
        max-width: 600px; /* ✅ Avoid huge dream fields expanding table */
    }
    .styled-table th {
        background-color: #f2f2f2;
    }
    </style>
    """

    # Write full HTML document
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"<!DOCTYPE html><html><head>{style}</head><body>{html}</body></html>")

    print(f"✅ HTML table saved to: {filename}")


def read_csvs(save_dir = "output"):
    import os
    import pandas as pd  
    dfs = []

    for f in os.listdir(save_dir):
        if f.endswith(".csv"):
            try:
                existing_df = pd.read_csv(os.path.join(save_dir, f))
                existing_df["filename"] = f
                dfs.append(existing_df)
            except Exception:
                continue

    dataset = pd.concat(dfs)
    return dataset


def plot_evaluations(dreams_interpretations_df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    import pandas as pd

    # Create figure with subplots for non-perplexity scores
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.grid()

    # Plot distributions without perplexity
    scores_without_perplexity = ['BLEU', 'ROUGE', 'BERT']
    for score in scores_without_perplexity:
        sns.kdeplot(data=dreams_interpretations_df[score], label=score, ax=ax1)
    ax1.set_title('Score Distributions (BLEU, ROUGE, BERT)')
    ax1.legend()

    # Calculate statistics for heatmap without perplexity
    stats_df = pd.DataFrame()
    for score in scores_without_perplexity:
        stats_df[score] = [
            dreams_interpretations_df[score].min(),
            dreams_interpretations_df[score].max(),
            dreams_interpretations_df[score].mean(),
            dreams_interpretations_df[score].median(),
            stats.mode(dreams_interpretations_df[score])[0]
        ]
    stats_df.index = ['Min', 'Max', 'Average', 'Median', 'Mode']

    # Plot heatmap
    sns.heatmap(stats_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
    ax2.set_title('Score Statistics (BLEU, ROUGE, BERT)')
    plt.tight_layout()

    # Create separate figure for perplexity
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot perplexity distribution
    sns.kdeplot(data=dreams_interpretations_df['perplexity'], ax=ax3)
    ax3.set_title('Perplexity Distribution')

    # Calculate perplexity statistics
    perplexity_stats = pd.DataFrame({
        'perplexity': [
            dreams_interpretations_df['perplexity'].min(),
            dreams_interpretations_df['perplexity'].max(),
            dreams_interpretations_df['perplexity'].mean(),
            dreams_interpretations_df['perplexity'].median(),
            stats.mode(dreams_interpretations_df['perplexity'])[0]
        ]
    })
    perplexity_stats.index = ['Min', 'Max', 'Average', 'Median', 'Mode']

    # Plot perplexity heatmap
    sns.heatmap(perplexity_stats, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
    ax4.set_title('Perplexity Statistics')
    plt.tight_layout()

    plt.show()