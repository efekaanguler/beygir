# Running BEYGIR

Use Python 3.12 from the repository root.

~~~sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp -n .env.example .env
.venv/bin/python -m unittest discover -s tests
~~~

Use the local .env for credentials. The numbered Groq keys rotate across requests. Keys from the same organization share the same quota group and do not multiply capacity. Set different group labels only when the keys belong to independently metered organizations. Groq's free limits are tracked per model and quota group. The runner skips a quota group until its short Retry-After interval elapses. After a timeout or temporary server error, the affected route cools down with increasing delays and other healthy routes can continue. Each failed request is logged with the route and status. If every route is unavailable beyond the short retry window, the run pauses and can be resumed with the same run ID.

## Start and resume

~~~sh
# Eight held-out samples, 144 scheduled calls across three Groq models
.venv/bin/python -m beygir run pilot --split pilot

# 400 samples, 7,200 scheduled calls across the same three models
.venv/bin/python -m beygir run main --split main

# Run the same command again after a pause to resume
.venv/bin/python -m beygir run main --split main
~~~

The main run spans multiple free-tier quota windows. For a small live check, use a separate run ID with --max-new-per-model 3. Short checks draw from the pilot sample and are not study results. The two GPT-OSS endpoints use low reasoning effort and Qwen uses reasoning disabled. The three Groq endpoints use strict output schemas, with the same schema across all conditions for a given item. The schemas constrain the response shape and do not provide candidate moves or labels. A model response that is invalid or truncated is retained and scored for format and task performance.

To add Llama 3.3 70B Instruct, set OPENROUTER_API_KEY and set OPENROUTER_PROVIDER to a fixed slug such as deepinfra/turbo. Use the same run ID with --models gpt20 gpt120 qwen llama. The existing three models' saved replies are reused and the Llama jobs are added to the run. To add MiniMax M2.7 through the same key, set OPENROUTER_MINIMAX_PROVIDER to a fixed slug such as gmicloud/fp8 and append minimax to the model list. Keep all already-selected aliases in that list when extending a run. Both OpenRouter adapters request the same JSON in the prompt and pin their provider. Validate an OpenRouter endpoint with one or two pilot jobs before adding it to a main run. The Groq keys currently in use reject general-purpose Llama and MiniMax. Groq's allam-2-7b can also be selected explicitly with --models allam.

## Rebuild the dataset

The frozen manifest contains 400 main items and eight disjoint pilot items. The main sample contains 50 items per group in each of the two tasks. Mate positions come from the committed CC0 slice of the [Lichess puzzle database](https://database.lichess.org/#puzzles), and history items come from seeded synthetic boards. Synthetic boards are valid under the three executable rule sets and need not correspond to a recorded game. Both the inputs and their provenance are tracked.

~~~sh
.venv/bin/python -m beygir.dataset --output data/rebuilt.jsonl
cmp benchmark/items.jsonl data/rebuilt.jsonl
~~~

The build uses the checked-in puzzle slice by default. Fetching a fresh upstream slice is optional and can change the benchmark.

## Rebuild tables and plots

~~~sh
.venv/bin/python -m beygir report main
~~~

The reports read the local SQLite database. No API calls are made during report generation. Each run directory under runs/ contains the protocol fingerprint, raw attempts in JSONL, item scores, condition metrics, all 15 condition pairings per model, the matched difference in the Chess–Okuz gap between each variant and standard chess, and figure files when the relevant matched responses are available. Back up the database and its runs/ directory locally if the machine or checkout will change.

Item accuracy counts malformed answers as incorrect when the given field cannot be scored. Precision, recall and F1 use interpretable binary classifications and are accompanied by their coverage and class counts. Balanced accuracy counts failed classifications against the true class. Matching uses the same items and includes only pairs where both API responses were received. A conditional mate-move comparison across rules includes only rows where both rules have a verified mate. The matched mate figure uses the stricter subset where all three rules have a verified mate. Missing API replies are reported separately from model errors.
