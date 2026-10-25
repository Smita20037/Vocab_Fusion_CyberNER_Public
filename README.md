# TF-IDF Vocabulary Addition for Cybersecurity NER (BERT-large, 5-fold CV)

## Overview

Cybersecurity text is full of domain-specific terms such as malware names, file paths, protocols and vulnerability identifiers. General-purpose tokenizers break many of these words into long sequences of wordpieces, which can weaken how well a language model represents them. This project tests whether adding high-scoring, heavily fragmented domain words to the `bert-large-cased` vocabulary improves named entity recognition (NER) on cybersecurity text.

## Goal

The goal is to measure whether a simple, lightweight vocabulary extension, guided by TF-IDF scores and wordpiece fragmentation, gives a reliable gain in cybersecurity NER without changing the model architecture or requiring domain-specific pretraining.

## Corpora

The project runs the same end-to-end pipeline on two corpora, implemented as two Colab notebooks:

| Notebook | Corpus | Tag set |
|---|---|---|
| `STUCCO_BERT_LARGE_TFIDF_VocabAdd_KFold_noO_5WP.ipynb` | STUCCO (auto-labelled) | STUCCO labels |
| `CyberNER_BERT_LARGE_TFIDF_VocabAdd_KFold_noO_5WP.ipynb` | CyberNER | STIX 2.1 tags |

## Method

Each notebook performs five steps:

1. **Load and preprocess.** Reads the token-level CSV, groups tokens into sentences, and splits any sentence longer than 140 words into 140-word chunks.
2. **Wordpiece dictionaries.** Counts how many wordpieces the original (unextended) tokenizer produces for each word, then saves cumulative buckets for 2+, 3+, 4+ and 5+ wordpieces.
3. **TF-IDF scoring.** Scores every word with TF-IDF (one document per sentence unit) and builds a word to label map, where each word takes the alphabetically first label it carries in the corpus.
4. **Domain word selection.** Ranks words by score and keeps the top 250 that satisfy the ablation filters, then adds them to the tokenizer with `tokenizer.add_tokens`.
5. **5-fold cross-validation.** Trains a fresh `bert-large-cased` token classifier per fold with resized embeddings, and evaluates on the held-out fold.

Only the first wordpiece of each word is labelled; the remaining wordpieces are masked with `-100`.

### Ablation arm

Both notebooks are configured for the best-performing arm, `noO_5wp`:

| Flag | Value | Meaning |
|---|---|---|
| `N_WORDS` | 250 | Number of domain words added to the vocabulary |
| `INCLUDE_O` | `False` | Exclude words whose label is `O` |
| `USE_5WP` | `True` | Keep only words that split into 5 or more wordpieces |
| Version labels | excluded | Words labelled `B-version` / `I-version` are never selected |

## Running ablation conditions

All ablation settings live in the `# ---- ablation arm ----` block of **Section 2: Configuration**. Nothing else in the notebook needs editing for the standard arms.

### Settings you can change

| Setting | Options | Effect |
|---|---|---|
| `INCLUDE_O` | `True` / `False` | `True` lets `O`-labelled words into the selected set. `False` restricts selection to entity-labelled words. |
| `USE_5WP` | `True` / `False` | `True` keeps only words that split into 5+ wordpieces. `False` ranks every word regardless of fragmentation. |
| `N_WORDS` | any integer, e.g. `50`, `100`, `250`, `500` | Number of top-ranked words added. `0` gives the no-addition baseline (no tokens added, embeddings unchanged). |
| `EXCLUDE_LABELS` (STUCCO) | tuple of labels | Labels whose words are never selected. Default `('B-version', 'I-version')`; set to `()` to allow version strings. |
| `DROP_VERSION` (CyberNER) | `True` / `False` | Same purpose as `EXCLUDE_LABELS`, as a single flag. |
| `LEARNING_RATE` | float | Defaults differ: 1e-5 (STUCCO), 5e-5 (CyberNER). Keep it fixed across arms of one corpus. |

### The four standard arms

| Arm name | `INCLUDE_O` | `USE_5WP` | Selection pool |
|---|---|---|---|
| `noO_5wp` (default, best case) | `False` | `True` | Entity words with 5+ wordpieces |
| `withO_5wp` | `True` | `True` | All words with 5+ wordpieces |
| `noO_no5wp` | `False` | `False` | All entity words |
| `withO_no5wp` | `True` | `False` | All words |

The arm name, and therefore the results folder, is built from these two flags automatically. `N_WORDS` is part of the folder name too, so runs with different word counts do not overwrite each other.

### Other wordpiece thresholds (2+, 3+, 4+)

Section 4 saves buckets for 2+, 3+, 4+ and 5+, but `USE_5WP` only switches the 5+ filter. To test another threshold, change this line in Section 6:

```python
words_5wp = set(wp_thresholds[5].keys())   # change 5 to 2, 3 or 4
```

The arm label will still read `5wp`, so also edit the `ARM` string in Section 2 (for example to `noO_3wp`) or the results will be written to the wrong folder.

### How to run each condition cleanly

`tokenizer.add_tokens` modifies the tokenizer in place. If you change a flag and re-run only Section 6, the new words are added on top of the previous arm's words, and the run is no longer the condition you think it is.

For every new condition:

1. Change the flags in Section 2.
2. **Runtime > Restart session**, then **Run all**. At minimum, re-run from Section 4 so the tokenizer is reloaded from `bert-large-cased`.
3. Check the printed `[BEFORE] vocab` value. It must equal the base vocabulary size (28,996 for `bert-large-cased`).
4. Check `candidate pool` and `selected`. If the pool is smaller than `N_WORDS`, a warning is printed and fewer words are added; report the actual count.

Folds are fixed by `KFOLD_SEED = 200`, so every arm is evaluated on identical splits and per-fold scores can be compared pairwise.

## Differences between the two notebooks

| | STUCCO | CyberNER |
|---|---|---|
| Input columns | `sentence_id`, `words`, `labels` | `Sentence_ID`, `Word`, `STIX_Tag` |
| Learning rate | 1e-5 | 5e-5 |
| Word score | Smoothed IDF (`norm=None`) | Mean TF-IDF over the documents containing the word (default `l2` norm) |
| Word to label map | Built from the processed sentence units | Built from the raw CSV grouped by `Word` |
| CSV loading | `on_bad_lines='skip'` | `low_memory=False`, column names stripped |
| F1 key in `fold_scores.json` | `f1` | `entity_f1` |
| Extra logged fields | `added_tokens`, `candidate_pool` | none |

## Shared hyperparameters

| Parameter | Value |
|---|---|
| Model | `bert-large-cased` |
| Max sentence length | 140 words / 140 tokens |
| Batch size (train / eval) | 32 / 32 |
| Epochs | 10 |
| Optimizer | Adam |
| Gradient clip norm | 10 |
| Folds | 5 (`KFold`, shuffled, `random_state=200`) |
| Seed | `42 + fold` |

## Requirements

Designed for Google Colab with an A100 GPU. A large GPU is needed for `bert-large-cased` at batch size 32.

```
transformers
seqeval==1.2.2
torch
scikit-learn
pandas
numpy
```

The first cell installs `transformers` and builds `seqeval` with a pinned version, which avoids a build failure on recent Colab images.

## Data

The datasets are not distributed with the code. Both notebooks mount Google Drive and read a token-level CSV:

| Corpus | Expected columns | Configured by |
|---|---|---|
| STUCCO | `sentence_id`, `words`, `labels` | `DATA_PATH` in Section 2 |
| CyberNER | `Sentence_ID`, `Word`, `STIX_Tag` | `DATA_PATH` in Section 2 |

Set `BASE` (output folder) and `DATA_PATH` (input CSV) in the configuration cell to match your Drive layout. Both CSVs are read with `ISO-8859-1` encoding.

## Running

1. Open the notebook in Colab and select an A100 runtime.
2. Check the paths and ablation flags in **Section 2: Configuration**.
3. Run all cells. Each fold trains for 10 epochs, so a full run takes several hours.

## Outputs

Intermediate dictionaries (saved under `BASE`):

| File | Contents |
|---|---|
| `*_wordpiece_dicts.pickle` | `{2: {...}, 3: {...}, 4: {...}, 5: {...}}`, word to label for each wordpiece threshold |
| `*_final5.pickle` | Words with 5+ wordpieces |
| `stucco_idf_scores.pickle` | STUCCO: word to smoothed IDF |
| `stucco_word_to_label.pickle` | STUCCO: word to label |
| `cyberner_data.pickle` | CyberNER: word to mean TF-IDF |

Per-arm results (saved under `RESULTS_DIR`):

| File | Contents |
|---|---|
| `added_domain_words.json` | The words added to the vocabulary |
| `fold_scores.json` | Precision, recall, entity F1 and token F1 per fold, updated after every fold |
| `fold{k}_report.txt` | `seqeval` per-entity classification report for fold k |
| `summary.csv` | Per-fold scores plus mean and sample standard deviation |

Results directories for the default arm:

- STUCCO: `BASE/stucco_250_noO_5wp_kfold/`
- CyberNER: `BASE/kfold_250_noO_5wp/`

## Metrics

- **Entity-level precision, recall, F1** from `seqeval` (strict span match on BIO tags).
- **Token-level micro F1** from scikit-learn, computed only over tokens whose gold label is not `O`.

## Citation

If you use this code, please cite the accompanying paper (citation to be added on publication).

## Access to the code

The code is hosted in a private repository. Access will be granted upon request by email to the corresponding author [g_deepa@blr.amrita.edu](mailto:g_deepa@blr.amrita.edu). Once processed,  an invitation to the private repository will be sent by email.
