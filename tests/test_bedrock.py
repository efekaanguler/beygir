import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from botocore.response import StreamingBody

from beygir.batch import (BedrockBatchClient, import_batch_file, import_rows,
                           load_batch_plan, prepare_batches)
from beygir.clients import BEDROCK
from beygir.dataset import DEFAULT_ITEMS, SEED
from beygir.endpoints import catalog
from beygir.scoring import parse_object
from beygir.storage import RunStore


class BedrockTests(unittest.IsolatedAsyncioTestCase):
    async def test_bearer_scope_transport_and_native_reply(self):
        scope = "bedrock.amazonaws.com/?X-Amz-Credential=placeholder/20261120/eu-north-1/bedrock/aws4_request"
        bearer = "bedrock-api-key-" + base64.b64encode(scope.encode()).decode()
        body = {
            "output": {"message": {"content": [{"reasoningContent": {}},
                                               {"text": '{"mate_exists": false, "move": null}'}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 500, "outputTokens": 20, "totalTokens": 520},
        }

        def server(request):
            self.assertEqual("Bearer " + bearer, request.headers["Authorization"])
            self.assertEqual("bedrock-runtime.eu-north-1.amazonaws.com", request.url.host)
            self.assertTrue(request.url.path.endswith("/converse"))
            self.assertEqual("low", json.loads(request.content)["additionalModelRequestFields"]["reasoning_effort"])
            return httpx.Response(200, json=body)

        with patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": bearer, "BEDROCK_REGION": "eu-central-1"}):
            with self.assertRaisesRegex(ValueError, "belongs to eu-north-1"):
                BEDROCK.routes(catalog()["aws-oss20"])
        with patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": bearer, "BEDROCK_REGION": "eu-north-1"}):
            endpoint = catalog()["aws-oss20"]
            request = BEDROCK.payload(endpoint, "Test", {})
            async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
                response = await BEDROCK.complete(client, endpoint, BEDROCK.routes(endpoint)[0], request)
        normalized = BEDROCK.normalize(response.json())
        self.assertEqual(body, normalized["bedrock"])
        self.assertEqual(520, normalized["usage"]["total_tokens"])
        self.assertEqual({"mate_exists": False, "move": None}, parse_object(normalized))

    def test_s3_download_waits_for_terminal_batch_state(self):
        client = BedrockBatchClient.__new__(BedrockBatchClient)
        client.control = Mock()
        client.s3 = Mock()
        remote = {"arn": "arn:example", "bucket": "fixture", "output_prefix": "run/out/"}
        client.control.get_model_invocation_job.return_value = {"status": "InProgress"}
        self.assertEqual([], list(client.outputs(remote)))
        client.s3.get_object.assert_not_called()

        first = {"recordId": "ABC", "modelOutput": {}}
        content = (json.dumps(first) + "\n").encode()
        client.control.get_model_invocation_job.return_value = {"status": "Completed"}
        client.s3.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": "run/out/input.jsonl.out"}, {"Key": "run/out/manifest.json.out"}]}
        ]
        client.s3.get_object.return_value = {"Body": StreamingBody(io.BytesIO(content), len(content))}
        self.assertEqual([first], list(client.outputs(remote)))
        client.s3.get_object.assert_called_once_with(Bucket="fixture", Key="run/out/input.jsonl.out")
        self.assertEqual("Completed", remote["status"])

    def test_prepared_batch_submits_and_imports_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"BEDROCK_REGION": "eu-central-1"}):
            directory = Path(tmp)
            with RunStore(directory / "run.sqlite3") as store:
                endpoint = catalog()["aws-qwen32"]
                prepare_batches(store, directory, "trial", DEFAULT_ITEMS, "main", [endpoint], SEED, 3)
                job = load_batch_plan(directory)["jobs"]["aws-qwen32"]
                self.assertEqual(144, len(job["records"]))
                source = directory / job["file"]
                input_rows = [json.loads(line) for line in source.read_text().splitlines()]
                self.assertEqual(set(job["records"]), {row["recordId"] for row in input_rows})

                fake = BedrockBatchClient.__new__(BedrockBatchClient)
                fake.region = "eu-central-1"
                fake.s3 = Mock()
                fake.control = Mock()
                fake.control.create_model_invocation_job.return_value = {"jobArn": "arn:example"}
                job["remote"] = fake.submit("trial", "aws-qwen32", job, source, "test-bucket", "test-role")
                submission = fake.control.create_model_invocation_job.call_args.kwargs
                self.assertEqual("Converse", submission["modelInvocationType"])
                self.assertEqual("qwen.qwen3-32b-v1:0", submission["modelId"])
                self.assertTrue(submission["inputDataConfig"]["s3InputDataConfig"]["s3Uri"].endswith("/input.jsonl"))

                first, second = input_rows[0], input_rows[1]
                rows = [{**second, "error": {"errorCode": 429}},
                        {**first, "modelOutput": {
                            "output": {"message": {"content": [{"text": '{"mate_exists":false,"move":null}'}]}},
                            "usage": {"totalTokens": 100},
                        }}]
                self.assertEqual(2, import_rows(store, "trial", job, rows))
                self.assertEqual(0, import_rows(store, "trial", job, rows))
                self.assertEqual({"response", "batch_error"}, {r["status"] for r in store.attempts("trial")})
                self.assertEqual(1, store.progress("trial")[0]["saved"])

                output = directory / "console.jsonl.out"
                third = input_rows[2]
                result = {**third, "modelOutput": rows[1]["modelOutput"]}
                output.write_text(json.dumps({**result, "modelInput": {"messages": []}}) + "\n")
                with self.assertRaisesRegex(ValueError, "does not match"):
                    import_batch_file(store, directory, "trial", "aws-qwen32", output)
                output.write_text(json.dumps(result) + "\n")
                self.assertEqual(1, import_batch_file(store, directory, "trial", "aws-qwen32", output))
                self.assertEqual(0, import_batch_file(store, directory, "trial", "aws-qwen32", output))
                self.assertEqual(2, store.progress("trial")[0]["saved"])
