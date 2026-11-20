"""Prepare, submit and import Bedrock Converse batches with stable record IDs."""

import hashlib
import itertools
import json
import os
import re
import time
from pathlib import Path

import boto3
import httpx
from botocore.config import Config

from .clients import BEDROCK, bedrock_key_region
from .endpoints import Endpoint
from .prompts import NAMES, render, response_format
from .rules import Rule
from .runner import prepare_run, response_tokens
from .storage import RunStore, encode, output_directory

MIN_RECORDS = 100
TERMINAL = {"Completed", "PartiallyCompleted", "Failed", "Stopped", "Expired", "ValidationFailed"}


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(encode(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_batches(store: RunStore, directory: Path, run_id: str, dataset: Path,
                    split: str, endpoints: list[Endpoint], seed: int, per_group: int | None) -> None:
    if not endpoints or any(not endpoint.region for endpoint in endpoints):
        raise ValueError("Batch preparation requires only aws-* model aliases")
    items, templates = prepare_run(store, run_id, dataset, split, endpoints, seed, per_group)
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / "plan.json"
    saved = json.loads(plan_path.read_text()) if plan_path.exists() else {"jobs": {}}
    completed = store.completed(run_id)
    for endpoint in endpoints:
        if endpoint.alias in saved["jobs"]:
            continue
        records = {}
        model_inputs = []
        for item, name, rule in itertools.product(items, NAMES, Rule):
            keys = (endpoint.model_id, item["id"], name, rule.value)
            if keys in completed:
                continue
            record_id = hashlib.sha256(encode(keys).encode()).hexdigest()[:11].upper()
            if record_id in records:
                raise ValueError("Batch record ID collision")
            records[record_id] = {"item_id": item["id"], "name": name, "rule": rule.value}
            prompt = render(item, name, rule, templates)
            model_inputs.append({"recordId": record_id,
                                 "modelInput": BEDROCK.payload(endpoint, prompt, response_format(item))})
        if len(records) < MIN_RECORDS:
            raise ValueError(f"{endpoint.alias}: batch needs at least {MIN_RECORDS} pending records")
        filename = f"{endpoint.alias}.jsonl"
        raw = "".join(encode(row) + "\n" for row in model_inputs).encode()
        path = directory / filename
        path.write_bytes(raw)
        saved["jobs"][endpoint.alias] = {
            "region": endpoint.region, "model": endpoint.model_id, "file": filename,
            "sha256": hashlib.sha256(raw).hexdigest(), "records": records,
        }
        print(f"{endpoint.alias}: prepared {len(records)} records in {path}")
    atomic_json(plan_path, saved)


def read_prepared_input(directory: Path, job: dict) -> bytes:
    path = directory / job["file"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != job["sha256"]:
        raise ValueError(f"Batch input was modified: {path}")
    return raw


def model_inputs(directory: Path, job: dict) -> dict[str, dict]:
    return {row["recordId"]: row["modelInput"]
            for row in map(json.loads, read_prepared_input(directory, job).splitlines())}


class BedrockBatchClient:
    """Use normal AWS IAM credentials for S3 and the Bedrock control plane."""

    def __init__(self, region: str):
        self.region = region
        session = boto3.Session(region_name=region)
        config = Config(connect_timeout=10, read_timeout=30,
                        retries={"mode": "standard", "total_max_attempts": 3})
        self.s3 = session.client("s3", config=config.merge(Config(signature_version="s3v4")))
        self.control = session.client("bedrock", config=config.merge(Config(signature_version="v4")))

    def submit(self, run_id: str, alias: str, job: dict, source: Path, bucket: str, role: str) -> dict:
        raw = read_prepared_input(source.parent, job)
        prefix = f"beygir/{run_id}/{alias}/{job['sha256'][:12]}"
        input_key = f"{prefix}/input.jsonl"
        output_prefix = f"{prefix}/out/"
        self.s3.put_object(Bucket=bucket, Key=input_key, Body=raw, ContentType="application/jsonl")
        token = hashlib.sha256(encode([self.region, bucket, prefix, role, job["model"]]).encode()).hexdigest()
        job_name = f"beygir-{re.sub(r'[^A-Za-z0-9-]', '-', run_id)[:32]}-{alias}-{token[:10]}"
        response = self.control.create_model_invocation_job(
            jobName=job_name, roleArn=role, clientRequestToken=token, modelId=job["model"],
            modelInvocationType="Converse",
            inputDataConfig={"s3InputDataConfig": {"s3InputFormat": "JSONL", "s3Uri": f"s3://{bucket}/{input_key}"}},
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{bucket}/{output_prefix}"}},
        )
        return {"arn": response["jobArn"], "bucket": bucket, "output_prefix": output_prefix}

    def outputs(self, remote: dict):
        job = self.control.get_model_invocation_job(jobIdentifier=remote["arn"])
        status = job["status"]
        remote["status"] = status
        print(f"{remote['arn'].rsplit('/', 1)[-1]}: {status}")
        if status not in TERMINAL:
            return
        bucket = remote["bucket"]
        pages = self.s3.get_paginator("list_objects_v2")
        for page in pages.paginate(Bucket=bucket, Prefix=remote["output_prefix"]):
            for entry in page.get("Contents", ()):
                key = entry["Key"]
                if key.endswith(".jsonl.out"):
                    body = self.s3.get_object(Bucket=bucket, Key=key)["Body"]
                    with body:
                        for line in body.iter_lines():
                            if line.strip():
                                yield json.loads(line)


def batch_directory(database: Path, run_id: str) -> Path:
    return output_directory(database, run_id) / "batch"


def load_batch_plan(directory: Path) -> dict:
    path = directory / "plan.json"
    if not path.exists():
        raise ValueError("Prepare the batch before submitting or collecting it")
    return json.loads(path.read_text())


def submit_batches(directory: Path, run_id: str) -> None:
    bucket = os.getenv("BEDROCK_BATCH_BUCKET", "").strip()
    role = os.getenv("BEDROCK_BATCH_ROLE_ARN", "").strip()
    if not (bucket and role):
        raise ValueError("Set BEDROCK_BATCH_BUCKET and BEDROCK_BATCH_ROLE_ARN before submitting")
    plan = load_batch_plan(directory)
    clients = {}
    for alias, job in plan["jobs"].items():
        if "remote" in job:
            print(f"{alias}: already submitted ({job['remote']['arn']})")
            continue
        if job["region"] not in clients:
            clients[job["region"]] = BedrockBatchClient(job["region"])
        job["remote"] = clients[job["region"]].submit(run_id, alias, job, directory / job["file"], bucket, role)
        atomic_json(directory / "plan.json", plan)
        print(f"{alias}: submitted {job['remote']['arn']}")


def import_rows(store: RunStore, run_id: str, job: dict, rows,
                expected_inputs: dict[str, dict] | None = None) -> int:
    completed = store.completed(run_id)
    route = "batch:" + job["remote"]["arn"]
    seen = {(row["item_id"], row["name"], row["rule"]) for row in store.attempts(run_id)
            if row["route"] == route}
    imported = 0
    for row in rows:
        record_id = row["recordId"]
        key = job["records"].get(record_id)
        if key is None:
            raise ValueError("Bedrock returned a record ID outside the saved batch manifest")
        if expected_inputs is not None and row.get("modelInput") != expected_inputs[record_id]:
            raise ValueError(f"Returned request does not match the saved input for {record_id}")
        cell = (key["item_id"], key["name"], key["rule"])
        if cell in seen or (job["model"], *cell) in completed:
            continue
        success = "modelOutput" in row
        body = BEDROCK.normalize(row["modelOutput"]) if success else row.get("error")
        store.record({"run_id": run_id, **key, "model": job["model"], "route": route,
                      "quota_group": f"bedrock-batch-{job['region']}", "started": time.time(),
                      "duration": 0.0, "status": "response" if success else "batch_error",
                      "http_status": 200 if success else None, "charged_tokens": response_tokens(body, 0),
                      "request": row.get("modelInput"), "response": body})
        seen.add(cell)
        imported += 1
    return imported


def collect_batches(store: RunStore, directory: Path, run_id: str) -> None:
    plan = load_batch_plan(directory)
    clients = {}
    for alias, job in plan["jobs"].items():
        if "remote" not in job:
            continue
        if job["region"] not in clients:
            clients[job["region"]] = BedrockBatchClient(job["region"])
        rows = clients[job["region"]].outputs(job["remote"])
        imported = import_rows(store, run_id, job, rows, expected_inputs=model_inputs(directory, job))
        print(f"{alias}: imported {imported} new records")
        atomic_json(directory / "plan.json", plan)


def check_batch(directory: Path) -> None:
    """Check local inputs and AWS connectivity without submitting a chargeable job."""
    jobs = list(load_batch_plan(directory)["jobs"].values())
    for job in jobs:
        raw = read_prepared_input(directory, job)
        record_ids = [json.loads(line)["recordId"] for line in raw.splitlines()]
        if len(record_ids) < MIN_RECORDS or len(record_ids) != len(job["records"]) or set(record_ids) != set(job["records"]):
            raise ValueError(f"Invalid record IDs or count in {job['file']}")
        print(f"{job['region']} | {job['model']} | {len(record_ids)} local records verified")
    regions = {job["region"] for job in jobs}
    bearer = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "").strip()
    if bearer:
        scope = bedrock_key_region(bearer)
        with httpx.Client(timeout=15, trust_env=False) as http:
            for region in sorted(regions):
                if scope and scope != region:
                    print(f"{region}: authorization probe skipped; bearer belongs to {scope}.")
                    continue
                response = http.get(f"https://bedrock.{region}.amazonaws.com/model-invocation-jobs",
                                    params={"maxResults": 1}, headers={"Authorization": f"Bearer {bearer}"})
                print(f"{region}: Bedrock ListModelInvocationJobs returned HTTP {response.status_code}")

    bucket = os.getenv("BEDROCK_BATCH_BUCKET", "").strip()
    role = os.getenv("BEDROCK_BATCH_ROLE_ARN", "").strip()
    missing = [name for name, value in (("BEDROCK_BATCH_BUCKET", bucket), ("BEDROCK_BATCH_ROLE_ARN", role)) if not value]
    for region in sorted(regions):
        session = boto3.Session(region_name=region)
        if session.get_credentials() is None:
            missing.append("AWS SDK credentials or AWS_PROFILE")
            break
        if bucket:
            result = session.client("s3", config=Config(connect_timeout=8, read_timeout=15)).head_bucket(Bucket=bucket)
            bucket_region = result.get("BucketRegion") or result["ResponseMetadata"]["HTTPHeaders"].get("x-amz-bucket-region")
            if bucket_region and bucket_region != region:
                raise ValueError(f"S3 bucket is in {bucket_region}; batch uses {region}")
            print(f"{region}: S3 HeadBucket succeeded")
    if missing:
        raise ValueError("Batch submission needs " + ", ".join(missing))
    print("Batch preflight complete. Job creation verifies PassRole and model permissions.")


def import_batch_file(store: RunStore, directory: Path, run_id: str, alias: str, path: Path) -> int:
    """Import a Bedrock JSONL output downloaded from the AWS console."""
    job = load_batch_plan(directory)["jobs"].get(alias)
    if job is None:
        raise ValueError(f"Model {alias} is absent from the prepared batch")
    inputs = model_inputs(directory, job)
    output = path.read_bytes()
    saved = {**job, "remote": {"arn": f"file:{hashlib.sha256(output).hexdigest()}"}}
    rows = (json.loads(line) for line in output.splitlines() if line.strip())
    return import_rows(store, run_id, saved, rows, expected_inputs=inputs)
