conda deactivate

python -m venv .venv
source .venv/bin/activate

pip install transformers
pip install torch
pip install pandas pandasql
pip install matplotlib
pip install ipykernel
pip install ipywidgets

pip install accelerate bitsandbytes tqdm
pip install huggingface_hub