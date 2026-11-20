# BEYGIR

**B**oard-game **E**valuation: **Y**ielding to **G**ame **I**dentity or **R**ules?

Efe Kaan Güler developed BEYGIR for the 2026–27 Natural Language Processing course at Università degli Studi di Milano, Track 7 (LLMs and Board Games). The project studies how eight language models respond to a familiar game name and changes to its rules. [Course paper](BEYGIR.pdf) · [Reference paper, GAMEBoT](https://aclanthology.org/2025.acl-long.378/)

## Experiment

Each base item appears under both names, **Chess** and **Okuz**, and three rulesets. Standard chess is the baseline. Mandatory capture requires a capture whenever a legal one exists. The bishop–knight swap exchanges the two pieces' movement and attack patterns. Each comparison holds the board, task, and supplied move sequence fixed. Within each ruleset, changing the game name changes only the prompt heading.

| Task | Base items | Main scoring measures |
| --- | ---: | --- |
| Mate in one | 60 | Mate detection and correct move on positions with a mate |
| Short history | 60 | Exact occupancy of two squares and first illegal move in a separate line |
| Generate five plies | 50 | All five generated moves legal and length of the legal prefix |
| Audit five plies | 50 | Balanced legality accuracy and exact first illegal index |
| Audit twenty plies | 50 | Same audit metrics |
| Audit fifty plies | 50 | Same audit metrics |

One ply is one side's turn. The 320 base items yield **1,920 test cases per model** and **15,360 scored responses**. There are 21 separate pilot items. The eight models used in the paper are GPT-OSS 120B, GPT-OSS 20B, Qwen3 32B, GLM 4.7, GLM 4.7 Flash, DeepSeek V3.2, Devstral 2 123B, and Amazon Nova Lite v1. The final run used AWS Bedrock. [Test manifest](benchmark/extended_items.jsonl) · [Prompts](prompts/)

## Setup

From the repository root, use Python 3.12 on macOS or Linux. Analysis and tests work without API credentials.

~~~sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
~~~

To rebuild the frozen manifest from the checked-in inputs, run the generator and compare the files.

~~~sh
.venv/bin/python -m beygir.extended_dataset --output data/rebuilt-extended.jsonl
cmp benchmark/extended_items.jsonl data/rebuilt-extended.jsonl
~~~

## Reproduce the analysis from SQLite

The completed run is named `stockholm-extended-main`. These commands use the saved `runs/beygir.sqlite3` database and make **zero API calls**. The database, raw replies, and regenerated artifacts are stored locally under the ignored `runs/` directory. A fresh checkout includes the test inputs, prompts, analysis code, and paper. To reproduce the original numerical outputs, put a local copy of the original database at that path or pass its location as `--db`. Running the experiment again produces a fresh set of model responses.

~~~sh
# Check completeness.
.venv/bin/python -m beygir status stockholm-extended-main --db runs/beygir.sqlite3

# Rebuild the general exports, including per-item scores and matched comparisons.
.venv/bin/python -m beygir report stockholm-extended-main --db runs/beygir.sqlite3

# Build the four paper plots, their source tables, and the LaTeX table rows.
.venv/bin/python -m beygir.analysis paper --db runs/beygir.sqlite3 \
  --run-id stockholm-extended-main --output runs/stockholm-extended-main/analysis

# Reproduce the independent five-ply quality check.
.venv/bin/python -m beygir.analysis play5-check --db runs/beygir.sqlite3 \
  --run-id stockholm-extended-main --output runs/stockholm-extended-main/analysis
~~~

The two `beygir.analysis` commands default to the displayed database, run ID, and output directory. They open the database read-only. The paper command checks for the complete final protocol and all 15,360 responses. The general `report` command can also score partial runs.

### From recorded request to score

The SQLite database stores the run specification and prompt fingerprint in `runs`, the boards and rule-specific gold answers in `items`, and requests, responses, and retry status in `attempts`. The scorer joins successful replies by run, item, model, game name, and ruleset. The rules engine computes legality and replays the generated moves. Format errors remain visible in the exports. Balanced accuracy counts uninterpretable legality labels as incorrect within the true class. Precision, recall, and F1 use interpretable decisions and are accompanied by class counts and coverage. [Storage](beygir/storage.py) · [Rule verifier](beygir/rules.py) · [Scoring](beygir/scoring.py)

| General export in `runs/stockholm-extended-main/` | Contents |
| --- | --- |
| `run_spec.json` and `raw_attempts.jsonl` | Protocol, requests, replies, and retry history |
| `item_scores.csv` | Parsed answer, gold answer, and outcome for each case |
| `metrics.csv` | Metrics per model, task, name, and rule |
| `matched_pairs.csv` | Paired outcomes for each condition contrast |
| `name_effects.csv` | Change in the Chess–Okuz gap between standard and variant rules |

### From scores to the paper

| Output under `analysis/` | How it was built |
| --- | --- |
| `name_effect.pdf` and `name_comparisons.csv` | Match the same model, task, item, and rule across the two names. Plot the Chess-minus-Okuz difference. |
| `rule_comparison.pdf` and `rule_comparisons.csv` | Compare the same fully legal lines across all three rules. Separately, on lines whose correct label flips, count whether both legality decisions are correct and whether the exact illegal ply was found. |
| `audit_length.pdf` and `audit_length.csv` | Plot balanced accuracy and exact first-error accuracy for 5, 20, and 50 plies. The lengths use separate position sets, so the comparison is descriptive. |
| `play5_comparison.pdf` and `generation_comparisons.csv` | Plot the share of fully legal five-ply lines by model, name, and rule. The CSV includes format and first-move counts. |
| `summary.json`, `name_rows.tex`, `audit_rows.tex`, and `play_rows.tex` | Pooled counts and generated table rows used in the paper. |
| `play5_quality_audit.csv` and `play5_opening_moves.csv` | Verify prompt parity and replay the moves with a second validator. Compare strict scoring with a separate, more permissive notation check. |

For example, the recorded run had **393 Chess-only** and **366 Okuz-only** successes in 7,680 paired name comparisons. On sequences legal in all three rulesets, correct acceptance was **236/960** under standard chess, **171/960** under mandatory capture, and **172/960** under the swap. These figures are computed from the SQLite records by the analysis code. [Analysis walkthrough and methods](docs/experiments.md#extended-legality-study)

## Run the benchmark again

An active AWS Bedrock account with access to all eight models in `eu-north-1` is required for the complete on-demand run, which incurs API charges. Copy the template if you do not already have a local `.env`, then set `AWS_BEARER_TOKEN_BEDROCK` and `BEDROCK_REGION` there.

~~~sh
cp -n .env.example .env
sh scripts/run-stockholm.sh
~~~

The launcher sends up to four requests per model in parallel. Stop and rerun the same command to resume from saved responses. To watch progress in another terminal, run:

~~~sh
.venv/bin/python -m beygir status stockholm-extended-main --watch 10
~~~

The experiment guide also documents the earlier Groq protocol, a separate Bedrock pilot, and optional batch support. [Experiment guide](docs/experiments.md)

## Repository layout

| Path | Contents |
| --- | --- |
| [`benchmark/`](benchmark/) | Frozen main and pilot cases, the precursor dataset, and the Lichess input slice |
| [`prompts/`](prompts/) | English task instructions and rules for the three game variants |
| [`beygir/`](beygir/) | Dataset generation, provider clients, resumable runner, storage, verification, and general reports |
| [`beygir/analysis/`](beygir/analysis/) | Paper-specific aggregation, vector plots, LaTeX table exports, and the independent five-ply check |
| [`scripts/`](scripts/) | On-demand launcher for the final Bedrock protocol |
| [`tests/`](tests/) | Rule, dataset, scoring, storage, and runner tests |
| [`docs/`](docs/) | Detailed commands and provider notes |
| [`BEYGIR.pdf`](BEYGIR.pdf) | Final report |

## AI-use disclosure

I used ChatGPT (GPT-6.2 Sol) to help implement the experiments, find relevant papers and related work, format the report, and convert my draft to LaTeX. I read and reviewed the entire report and take full responsibility for its content.

ChatGPT (GPT-6.2 Sol) also assisted with drafting this README.
