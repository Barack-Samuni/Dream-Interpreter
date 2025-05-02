conda deactivate

python -m venv .venv
source .venv/bin/activate

pip install transformers datasets
pip install torch
pip install pandas pandasql
pip install matplotlib plotly
pip install ipykernel tabulate
pip install ipywidgets datasets

pip install accelerate bitsandbytes tqdm
pip install huggingface_hub