
def release_all_gpu_memory(additional_objects=[]):
    import gc
    import torch

    # Delete model objects (make sure they're declared global or passed)
    globals_to_clear = ["model", "tokenizer", "text2text_generator"] + additional_objects
    for name in globals_to_clear:
        if name in globals():
            print("clearing ", name)
            del globals()[name]

    gc.collect()

    if torch.cuda.is_available():
        print("clearing cuda cache")
        torch.cuda.empty_cache()
        print("clearing ipc cache")
        torch.cuda.ipc_collect()

    print("✅ All GPU memory cleared.")


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