# Lexicon Lucidus

Psycholinguistic analysis using PyTorch, transformer sentence embeddings, and a classifier trained with a conditional adversarial autoencoder (CAAE)-style objective.

This guide is for people running the repository locally as a consumer. Commands below assume you run them from the repository root (the folder containing `requirements.txt`).

## What It Does

The project expands labeled MBTI seed sentences, turns a sample of the generated text into 384-dimensional embeddings using `sentence-transformers/all-MiniLM-L6-v2`, trains a small neural model, and predicts an MBTI label for each sentence in `inputs.csv`. The prediction output also includes the trait descriptions in `mbti_traits.csv` for the predicted type.

The model is trained locally each time; this repository does not currently save or load trained model checkpoints. The first run downloads the transformer tokenizer and model from Hugging Face, so an internet connection is required unless the model is already cached.

## Requirements

- Python 3.10 or newer is recommended.
- Internet access for installing packages and downloading the transformer model on the first run.
- A few hundred megabytes of free disk space for the Python packages and cached transformer model.
- CPU is sufficient for the default small training run. A CUDA-enabled PyTorch installation may be used where available, but the current scripts create models and tensors on CPU.

The dependencies are listed in `requirements.txt`: pandas, PyTorch, JupyterLab, Jupytext, and Transformers.

## Quick Start: Windows PowerShell

Open PowerShell in the repository folder, or navigate there first:

```powershell
cd path\to\lexicon-lucidus
```

Create and activate an isolated virtual environment, then install the dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, allow scripts only in the current PowerShell process and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Run the pipeline from the repository root:

```powershell
python mbti_gen.py
python inference.py
```

The first command creates or replaces `mbti_texts.csv` from `mbti_texts_seeds.csv`. The second command builds the dataset, downloads/loads the transformer if needed, generates embeddings, trains the model once, then predicts the types in `inputs.csv` and prints their trait descriptions in the terminal. The `Type` column in `inputs.csv` is shown as the expected/reference label; it does not influence the predicted result. To run training without predictions, use `python train.py` instead.

The full convenience script is also available:

```powershell
.\run.ps1
```

**Note:** `run.ps1` currently trains twice: it runs `train.py`, and then `inference.py` imports `train.py`, which starts training again. Use the two commands above for a single training pass followed by predictions. The scripts execute top-level work when imported, so treat `train.py` and `inference.py` as command-line scripts rather than importable library modules.

## Quick Start: macOS or Linux

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python mbti_gen.py
python inference.py
```

This generates the training CSV, then trains once and predicts. To train without running inference, use `python train.py` instead. `run.ps1` is a PowerShell script and is intended for Windows PowerShell; use the explicit Python commands on macOS and Linux.

## Train the Model

To run only the training step, activate the virtual environment, change to the repository root, and make sure `mbti_texts.csv` exists. Generate it from the seed data if needed:

```powershell
python mbti_gen.py
python train.py
```

Use the same commands in an activated virtual environment on macOS or Linux. You can skip `mbti_gen.py` when you already have a generated `mbti_texts.csv` and do not want to replace it. The training script samples 640 rows, creates the transformer embeddings, initializes the encoder, decoder, discriminator, and classifier, then trains for the configured number of epochs. It prints the latest training loss at the end of each epoch. Adjust the sample size, batch size, or epoch count in `constants.py` before running if needed.

Training starts from newly initialized model weights on every run and currently does not save a checkpoint. Running `python train.py` by itself does not make predictions. To get predictions, run `python inference.py`; that script imports `train.py` and therefore performs a fresh training run before processing `inputs.csv`. Avoid running `train.py` immediately before `inference.py` unless you intentionally want to train twice.

## Add or Change Prediction Inputs

Edit `inputs.csv` with exactly these column headers:

```csv
Type,Text
INTP,"I enjoy exploring how ideas connect."
ENFP,"I like bringing people together around new possibilities."
```

- `Text` is the sentence sent to the embedding model and classifier.
- `Type` is an optional reference value for your own comparison, but the current inference script expects the column to exist and prints it as the expected type. Use a valid label from the 16 MBTI types if supplying a reference.
- Add one sentence per row. Keep the CSV valid by quoting text containing commas, quotes, or line breaks.

Run `python inference.py` after preparing the generated training CSV as described above. Each row produces the input sentence, expected type, predicted type, and the predicted type's trait descriptions. Results are printed to the terminal; the script does not write a prediction CSV.

## Notebook Route

The repository also provides `lexicon_lucidus.ipynb` as a notebook-based entry point. Start JupyterLab from the repository root:

```powershell
python -m jupyter lab lexicon_lucidus.ipynb
```

On macOS or Linux, use the same command in the activated virtual environment. Run the notebook cells in order. The helper `run_notebook.ps1` launches this notebook with JupyterLab on Windows PowerShell:

```powershell
.\run_notebook.ps1
```

## Repository Files

| File | Purpose |
| --- | --- |
| `mbti_texts_seeds.csv` | Small labeled seed dataset used to create training text. |
| `mbti_gen.py` | Expands the seed sentences into `mbti_texts.csv`. |
| `mbti_texts.csv` | Generated training text; recreated by `mbti_gen.py`. |
| `inputs.csv` | Input sentences and optional expected labels for inference. |
| `mbti_traits.csv` | Trait descriptions printed for each predicted type. |
| `embeddings.py` | Loads the transformer and generates masked mean-pooled sentence embeddings. |
| `dataset.py` | Samples generated text and exposes labels and embeddings to training. |
| `caae.py` | Defines the encoder, decoder, discriminator, and classifier networks. |
| `constants.py` | Defines model dimensions, labels, batch size, and epoch count. |
| `train.py` | Prepares data and trains the model. |
| `inference.py` | Predicts types for rows in `inputs.csv`. Importing it also imports and runs training. |
| `run.ps1` | Convenience pipeline script for Windows PowerShell. |
| `lexicon_lucidus.ipynb` | Notebook entry point. |

## Configuration and Runtime

The default settings in `constants.py` use 640 sampled training examples, batch size 32, latent dimension 64, and 5 epochs. The generated CSV contains 40 variations for each of the 16 labels. `mbti_gen.py` uses random sampling without a fixed seed, so regenerated examples and training samples can differ between runs. Change these values in `constants.py` if you are experimenting with the training workload or architecture; keep the embedding dimension aligned with the selected transformer model.

On CPU, initial execution may take longer because of dependency/model downloads and transformer embedding generation. Training progress is reported once per epoch. The trained model currently exists only in memory for that process; there is no persisted checkpoint to reuse in a later invocation.

## Troubleshooting

- **`ModuleNotFoundError` for a dependency:** activate the virtual environment and run `python -m pip install -r requirements.txt` again.
- **Hugging Face download or connection error:** check internet access and retry. The model is `sentence-transformers/all-MiniLM-L6-v2` and is cached locally after a successful download.
- **CSV file not found:** run commands from the repository root so relative paths such as `inputs.csv` and `mbti_texts.csv` resolve correctly.
- **Missing or renamed CSV columns:** preserve `Type` and `Text` in both training and inference CSVs. Inference requires both columns in `inputs.csv`.
- **PowerShell does not recognize `py`:** install Python from python.org and enable the Python launcher, or replace `py -3` with `python` when creating the virtual environment.
- **PowerShell blocks `.ps1` execution:** use the process-scoped execution policy command in the Windows quick start, or execute the Python commands directly.

## Limitations

This is an educational/demo codebase, not a validated psychological assessment. The generated training rows are simple variations of a small seed set, and the model is trained from scratch on a small sample. Its MBTI predictions and associated traits should not be treated as scientifically reliable, diagnostic, or suitable for decisions about people. It may also produce different outputs between runs because data sampling and generation are not seeded.
