# Running BEYGIR

The [README](../README.md) gives the final study's setup, results workflow,
and exact offline commands. The legacy Groq and Bedrock pilots below use
different run IDs and sample sizes.

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

# Main experiment: wait across quota windows and resume any saved run
.venv/bin/python -m beygir run main --split main --keep-waiting

# Inspect progress while the runner is active, without API calls
.venv/bin/python -m beygir status main
~~~

The main run spans multiple free-tier quota windows. Keep-waiting mode waits until an account is eligible for further calls, pauses a job for five minutes after three failed attempts, and writes partial reports every 15 minutes. Two independent Groq organizations can run in parallel within their own request and token allowances. A machine running the experiment must remain awake and connected. On macOS, prefix the run command with caffeinate -i to prevent idle sleep. Use the same run ID to resume after an interruption. For a small live check, use a separate run ID with --max-new-per-model 3. Short checks draw from the pilot sample and are not study results. The two GPT-OSS endpoints use low reasoning effort and Qwen uses reasoning disabled. The three Groq endpoints use strict output schemas, with the same schema across all conditions for a given item. The schemas constrain the response shape and do not provide candidate moves or labels. A model response that is invalid or truncated is retained and scored for format and task performance.

Groq’s allam-2-7b can also be selected explicitly with --models allam.

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

## AWS Bedrock

Stockholm is the default region. Set BEDROCK_REGION=eu-north-1 and place the matching bearer in AWS_BEARER_TOKEN_BEDROCK. The CLI rejects a recognized short-term bearer from a different AWS region. Use a new run ID when changing the region, dataset size, or sampling scheme.

The expanded benchmark selects 20 items per sampling group per task for 160 unique items. Each of seven models receives the same 960 requests across the six conditions. The endpoints are GPT-OSS 20B, GPT-OSS 120B, Qwen3 32B, GLM 4.7 Flash, Amazon Nova 2 Lite, Mistral Devstral 2 123B and GLM 4.7. Nova uses an EU inference profile.

### Sequential on demand

The delivery run is planned for 120 balanced items with 15 items per sampling group and task. Its eight models share all six conditions and receive 720 requests each. This makes 5,760 planned responses. DeepSeek uses its native JSON schema with thinking disabled. The sequential roster uses Nova Lite v1, which produced valid answers for both pilot tasks. GPT-OSS 20B can return a truncated or malformed answer, which is retained as a model outcome. Availability of Claude, Grok and GPT-6.2 Sol depends on the AWS account, so validate both tasks for each model before including it.

~~~sh
.venv/bin/python -m beygir run stockholm-serial-main --split main \
  --per-group 15 --serial --keep-waiting \
  --models aws-oss20 aws-oss120 aws-qwen32 aws-glmflash aws-nova-lite aws-devstral aws-glm47 aws-deepseek
~~~

Serial mode permits at most one in-flight request across all endpoints. Models can take turns while another endpoint cools down. Reuse the same run ID and arguments to resume. Bedrock sends native Converse requests and keeps the raw provider reply alongside normalized scoring fields. Scoring accepts a single JSON object with an optional Markdown code fence.

### Extended legality study

The frozen benchmark in benchmark/extended_items.jsonl contains 320 main items and 21 separate pilot items. It retains 60 mate-in-one and 60 short-history items, and adds 50 items in each of four tasks: generate five legal half-moves, then audit a supplied line at 5, 20 or 50 half-moves. One half-move is one side's turn. In the generation task a valid line may differ from the private reference line. The evaluator applies each proposed move and scores the exact five-move success rate and the legal prefix length.

For each audit length, 20 lines are legal under all three rulesets, 10 contain an illegal move under all three, 10 are illegal only under mandatory capture and 10 are illegal only under the bishop–knight swap. Prompts provide the same starting board and line across all six conditions. Rule-specific ground truth identifies the first illegal move. The detailed rule text defines piece movement, attacks, pins, check and the scope of special moves. The independently seeded length groups support a descriptive comparison of audit difficulty.

The eight models each receive 1,920 main requests. The catalogue and pilot plan are the same as in the sequential experiment. The extended run uses its own ID and request fingerprint.

~~~sh
# Rebuild the frozen extended dataset without connecting to a model.
.venv/bin/python -m beygir.extended_dataset --output data/rebuilt-extended.jsonl
cmp benchmark/extended_items.jsonl data/rebuilt-extended.jsonl

# Start or resume with up to four on-demand requests per model.
sh scripts/run-stockholm.sh

# Follow overall, model, and task progress from a separate terminal.
.venv/bin/python -m beygir status stockholm-extended-main --watch 10
~~~

The launcher prevents idle sleep on macOS with caffeinate. Responses and failed attempts are committed to SQLite as they arrive. The runner waits through throttling and retries temporary errors, and exports a snapshot every 15 minutes. A permanent endpoint error pauses that model while the others continue. If the short-term Bedrock token expires, replace it in .env and restart the launcher with the existing run ID. Saved responses are reused. Scores and plots can also be rebuilt offline with `.venv/bin/python -m beygir report stockholm-extended-main`.

The export adds all-moves-legal accuracy and legal prefix statistics for generation, and legality and first-illegal-index metrics for each audit length. The audit_by_length.png figure appears once there are positive and negative responses for at least one audit cell.

### Reproduce the paper analyses

With the original SQLite database in place, run the paper and quality
commands from the repository root. Both read the saved run and work without
provider credentials. See the [README analysis guide](../README.md#reproduce-the-analysis-from-sqlite)
for the figure-to-metric mapping and the output filenames.

~~~sh
.venv/bin/python -m beygir.analysis paper
.venv/bin/python -m beygir.analysis play5-check
~~~

### Batch

Use a unique run ID for a batch. Preparation creates one JSONL input per model under runs/. Batch submission and automated collection use standard AWS SDK credentials with S3 access and Bedrock job permissions. Configure AWS_PROFILE if needed, and set BEDROCK_BATCH_BUCKET to an existing Stockholm bucket and BEDROCK_BATCH_ROLE_ARN to a Bedrock batch service role. The submitting identity needs permission to pass that role. Limit the role’s S3 permissions to the relevant input and output paths.

~~~sh
# 144 records for a small one-model batch test
.venv/bin/python -m beygir batch-prepare stockholm-batch-smoke --per-group 3 --models aws-qwen32
.venv/bin/python -m beygir batch-check stockholm-batch-smoke
.venv/bin/python -m beygir batch-submit stockholm-batch-smoke
.venv/bin/python -m beygir batch-collect stockholm-batch-smoke

# Expanded experiment
.venv/bin/python -m beygir batch-prepare stockholm-delivery --per-group 20 \
  --models aws-oss20 aws-oss120 aws-qwen32 aws-glmflash aws-nova aws-devstral aws-glm47
.venv/bin/python -m beygir batch-check stockholm-delivery
.venv/bin/python -m beygir batch-submit stockholm-delivery
.venv/bin/python -m beygir batch-collect stockholm-delivery
.venv/bin/python -m beygir status stockholm-delivery
~~~

If AWS CLI credentials are unavailable, you can use the AWS console in Stockholm. Upload the prepared model JSONL to S3, select the matching model and Converse invocation type in Bedrock Batch inference, and choose the input, output and service role. Download its .jsonl.out file after completion and import it locally:

~~~sh
.venv/bin/python -m beygir batch-import stockholm-batch-smoke \
  --model aws-qwen32 --output /path/to/input.jsonl.out
~~~

Collection and local import can be repeated. The importer uses record IDs to pair results and keeps per-record errors. AWS batch completion depends on capacity.
