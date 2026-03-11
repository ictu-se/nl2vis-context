import argparse
import json
import math
import random
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sacrebleu import corpus_bleu
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
DATA_JSON = ROOT / "nvBench.json"
DATABASE_DIR = ROOT / "database"
ARTIFACTS_DIR = ROOT / "artifacts_context_study"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def sql_normalize(sql: str) -> str:
    sql = sql.replace('"', "").replace("`", "").strip()
    sql = re.sub(r"\s+", " ", sql)
    sql = re.sub(r"\s*,\s*", ", ", sql)
    sql = re.sub(r"\s*\(\s*", "(", sql)
    sql = re.sub(r"\s*\)\s*", ")", sql)
    return sql.lower()


def infer_semantic_type(col_name: str, data_type: str) -> str:
    col = col_name.lower()
    dtype = (data_type or "").lower()
    if "date" in col or "time" in col or "year" in col or "date" in dtype or "time" in dtype:
        return "temporal"
    if col.endswith("id") or col == "id" or "key" in col:
        return "identifier"
    if any(token in dtype for token in ["int", "real", "numeric", "float", "double", "decimal"]):
        return "numeric"
    if any(token in dtype for token in ["char", "text", "clob", "varchar"]):
        return "categorical_or_text"
    return "unknown"


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z_]+", text.lower())


def safe_identifier(name: str) -> str:
    return f'"{name}"'


def record_sort_key(record_id: str) -> Tuple[int, str]:
    match = re.match(r"(\d+)", record_id)
    if match:
        return int(match.group(1)), record_id
    return math.inf, record_id


def fetch_rows(conn: sqlite3.Connection, sql: str) -> List[Tuple]:
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchmany(500)
    return [tuple("" if v is None else str(v) for v in row) for row in rows]


def parse_vql(vql: str) -> Dict[str, str]:
    text = " ".join(vql.split())
    match = re.match(r"Visualize\s+(.+?)\s+SELECT\s+(.*)", text, flags=re.IGNORECASE)
    if not match:
        return {"chart": "", "sql": text, "binning": ""}
    chart = match.group(1).strip().lower()
    tail = "SELECT " + match.group(2).strip()
    if " BIN " in tail:
        sql_part, binning = tail.split(" BIN ", 1)
        return {"chart": chart, "sql": sql_part.strip(), "binning": binning.strip().lower()}
    return {"chart": chart, "sql": tail.strip(), "binning": ""}


def slot_f1(pred: Dict[str, str], gold: Dict[str, str]) -> float:
    pred_slots = {k: v for k, v in pred.items() if v}
    gold_slots = {k: v for k, v in gold.items() if v}
    if not pred_slots and not gold_slots:
        return 1.0
    tp = sum(1 for k, v in pred_slots.items() if gold_slots.get(k, None) == v)
    fp = len(pred_slots) - tp
    fn = len(gold_slots) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


@dataclass
class ColumnMeta:
    name: str
    data_type: str
    pk: bool
    notnull: bool
    semantic_type: str
    distinct_count: Optional[int]
    sample_values: List[str]
    numeric_stats: Dict[str, Optional[float]]


@dataclass
class TableMeta:
    name: str
    columns: List[ColumnMeta]
    row_count: int
    foreign_keys: List[Tuple[str, str, str]]


def build_db_metadata(db_id: str) -> Dict:
    db_path = DATABASE_DIR / db_id / f"{db_id}.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    tables = [
        row[0]
        for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        if not row[0].startswith("sqlite_")
    ]
    table_meta: List[TableMeta] = []
    for table in tables:
        pragma_rows = cursor.execute(f"PRAGMA table_info({safe_identifier(table)})").fetchall()
        fk_rows = cursor.execute(f"PRAGMA foreign_key_list({safe_identifier(table)})").fetchall()
        try:
            row_count = cursor.execute(f"SELECT COUNT(*) FROM {safe_identifier(table)}").fetchone()[0]
        except Exception:
            row_count = 0
        columns: List[ColumnMeta] = []
        for col in pragma_rows:
            col_name = col[1]
            dtype = col[2] or ""
            semantic = infer_semantic_type(col_name, dtype)
            distinct_count = None
            sample_values: List[str] = []
            numeric_stats = {"min": None, "max": None, "avg": None}
            try:
                distinct_count = cursor.execute(
                    f"SELECT COUNT(DISTINCT {safe_identifier(col_name)}) FROM {safe_identifier(table)}"
                ).fetchone()[0]
            except Exception:
                distinct_count = None
            try:
                sample_query = (
                    f"SELECT DISTINCT {safe_identifier(col_name)} FROM {safe_identifier(table)} "
                    f"WHERE {safe_identifier(col_name)} IS NOT NULL LIMIT 3"
                )
                sample_values = [str(r[0]) for r in cursor.execute(sample_query).fetchall() if r[0] is not None]
            except Exception:
                sample_values = []
            if semantic == "numeric":
                try:
                    stats_row = cursor.execute(
                        f"SELECT MIN({safe_identifier(col_name)}), MAX({safe_identifier(col_name)}), "
                        f"AVG({safe_identifier(col_name)}) FROM {safe_identifier(table)} "
                        f"WHERE {safe_identifier(col_name)} IS NOT NULL"
                    ).fetchone()
                    numeric_stats = {
                        "min": None if stats_row[0] is None else float(stats_row[0]),
                        "max": None if stats_row[1] is None else float(stats_row[1]),
                        "avg": None if stats_row[2] is None else float(stats_row[2]),
                    }
                except Exception:
                    pass
            columns.append(
                ColumnMeta(
                    name=col_name,
                    data_type=dtype,
                    pk=bool(col[5]),
                    notnull=bool(col[3]),
                    semantic_type=semantic,
                    distinct_count=distinct_count,
                    sample_values=sample_values,
                    numeric_stats=numeric_stats,
                )
            )
        foreign_keys = [(fk[3], fk[2], fk[4]) for fk in fk_rows]
        table_meta.append(TableMeta(name=table, columns=columns, row_count=row_count, foreign_keys=foreign_keys))
    conn.close()
    return {
        "db_id": db_id,
        "tables": [
            {
                "name": t.name,
                "row_count": t.row_count,
                "columns": [
                    {
                        "name": c.name,
                        "data_type": c.data_type,
                        "pk": c.pk,
                        "notnull": c.notnull,
                        "semantic_type": c.semantic_type,
                        "distinct_count": c.distinct_count,
                        "sample_values": c.sample_values,
                        "numeric_stats": c.numeric_stats,
                    }
                    for c in t.columns
                ],
                "foreign_keys": [
                    {"column": src, "ref_table": ref_table, "ref_column": ref_col}
                    for src, ref_table, ref_col in t.foreign_keys
                ],
            }
            for t in table_meta
        ],
    }


def load_or_build_metadata(records: List[Dict]) -> Dict[str, Dict]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    cache_path = ARTIFACTS_DIR / "db_metadata.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    db_ids = sorted({r["db_id"] for r in records})
    metadata = {}
    print(f"[metadata] building metadata for {len(db_ids)} databases")
    for db_id in tqdm(db_ids, desc="DB metadata"):
        metadata[db_id] = build_db_metadata(db_id)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def select_relevant_tables(db_meta: Dict, query: str, limit: int = 4) -> List[Dict]:
    tokens = set(simple_tokenize(query))
    scored = []
    for table in db_meta["tables"]:
        score = 0
        table_name_tokens = set(simple_tokenize(table["name"]))
        score += len(tokens & table_name_tokens) * 3
        for col in table["columns"]:
            col_tokens = set(simple_tokenize(col["name"]))
            score += len(tokens & col_tokens) * 2
            for value in col["sample_values"]:
                if value and value.lower() in query.lower():
                    score += 3
        score += min(table["row_count"], 1000) / 1000.0
        scored.append((score, table))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [table for _, table in scored[:limit]]
    return chosen


def render_context(db_meta: Dict, query: str, variant: str) -> str:
    if variant == "baseline":
        return ""
    if variant == "schema":
        tables = db_meta["tables"]
    elif variant in {"schema_type", "selected_enriched"}:
        tables = select_relevant_tables(db_meta, query) if variant == "selected_enriched" else db_meta["tables"]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    lines: List[str] = [f"Database: {db_meta['db_id']}"]
    for table in tables:
        if variant == "schema":
            cols = ", ".join(col["name"] for col in table["columns"])
            lines.append(f"Table {table['name']}({cols})")
            continue
        col_chunks = []
        for col in table["columns"][:12]:
            chunk = f"{col['name']}:{col['semantic_type']}"
            if col["pk"]:
                chunk += ":pk"
            if variant == "selected_enriched":
                extras = []
                if col["sample_values"]:
                    extras.append("samples=" + "/".join(col["sample_values"][:3]))
                if col["semantic_type"] == "numeric" and col["numeric_stats"]["min"] is not None:
                    extras.append(
                        "stats="
                        + f"min:{col['numeric_stats']['min']:.2f},max:{col['numeric_stats']['max']:.2f},avg:{col['numeric_stats']['avg']:.2f}"
                    )
                elif col["distinct_count"] is not None:
                    extras.append(f"distinct={col['distinct_count']}")
                if extras:
                    chunk += "[" + "; ".join(extras) + "]"
            col_chunks.append(chunk)
        lines.append(f"Table {table['name']} rows={table['row_count']} :: " + ", ".join(col_chunks))
        if table["foreign_keys"]:
            fk_text = ", ".join(
                f"{fk['column']}->{fk['ref_table']}.{fk['ref_column']}" for fk in table["foreign_keys"][:6]
            )
            lines.append(f"FK {table['name']}: {fk_text}")
    return "\n".join(lines)


def compact_selected_enriched_context(db_meta: Dict, query: str, tokenizer, token_budget: int = 430) -> str:
    tables = select_relevant_tables(db_meta, query, limit=3)
    base_prefix = (
        "Task: generate VQL for data visualization.\n"
        f"Query: {query}\n"
        "Context:\n"
    )
    suffix = "\nOutput:"

    def within_budget(candidate_context: str) -> bool:
        text = base_prefix + candidate_context + suffix
        return len(tokenizer(text, truncation=False).input_ids) <= token_budget

    lines: List[str] = [f"Database: {db_meta['db_id']}"]
    if within_budget("\n".join(lines)):
        pass

    # First pass: compact schema with top columns and FK hints.
    for table in tables:
        col_parts = []
        for col in table["columns"][:6]:
            part = f"{col['name']}:{col['semantic_type']}"
            if col["pk"]:
                part += ":pk"
            col_parts.append(part)
        table_line = f"T {table['name']} rows={table['row_count']} :: " + ", ".join(col_parts)
        candidate_lines = lines + [table_line]
        if within_budget("\n".join(candidate_lines)):
            lines = candidate_lines
        fk_pairs = [
            f"{fk['column']}->{fk['ref_table']}.{fk['ref_column']}"
            for fk in table["foreign_keys"][:3]
        ]
        if fk_pairs:
            fk_line = f"FK {table['name']}: " + ", ".join(fk_pairs)
            candidate_lines = lines + [fk_line]
            if within_budget("\n".join(candidate_lines)):
                lines = candidate_lines

    # Second pass: add targeted value hints only for columns matching the query.
    query_tokens = set(simple_tokenize(query))
    for table in tables:
        matched_columns = []
        for col in table["columns"]:
            col_tokens = set(simple_tokenize(col["name"]))
            value_hit = any(value and value.lower() in query.lower() for value in col["sample_values"][:3])
            if query_tokens & col_tokens or value_hit:
                sample_text = "/".join(col["sample_values"][:2]) if col["sample_values"] else ""
                stat_text = ""
                if col["semantic_type"] == "numeric" and col["numeric_stats"]["min"] is not None:
                    stat_text = (
                        f" min={col['numeric_stats']['min']:.0f}"
                        f" max={col['numeric_stats']['max']:.0f}"
                    )
                elif col["distinct_count"] is not None:
                    stat_text = f" distinct={col['distinct_count']}"
                hint = f"{col['name']}"
                if sample_text:
                    hint += f" sample={sample_text}"
                if stat_text:
                    hint += stat_text
                matched_columns.append(hint)
        if matched_columns:
            hint_line = f"H {table['name']}: " + " | ".join(matched_columns[:3])
            candidate_lines = lines + [hint_line]
            if within_budget("\n".join(candidate_lines)):
                lines = candidate_lines

    context = "\n".join(lines)
    # Absolute guarantee for the selected_enriched prompt under the model limit.
    while len(tokenizer(base_prefix + context + suffix, truncation=False).input_ids) > token_budget and len(lines) > 1:
        lines.pop()
        context = "\n".join(lines)
    return context


def build_source_text(record: Dict, metadata: Dict[str, Dict], tokenizer, variant: str) -> str:
    query = record["query"]
    if variant == "baseline":
        return (
            "Task: generate VQL for data visualization.\n"
            f"Query: {query}\n"
            "Output:"
        )
    if variant == "selected_enriched":
        context = compact_selected_enriched_context(metadata[record["db_id"]], query, tokenizer, token_budget=510)
    else:
        context = render_context(metadata[record["db_id"]], query, variant)
    return (
        "Task: generate VQL for data visualization.\n"
        f"Query: {query}\n"
        f"Context:\n{context}\n"
        "Output:"
    )


def load_records() -> List[Dict]:
    with DATA_JSON.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records = []
    for rid, item in payload.items():
        nl_queries = item.get("nl_queries", [])
        if not nl_queries:
            continue
        record = {
            "record_id": rid,
            "db_id": item["db_id"],
            "query": nl_queries[0].strip(),
            "all_queries": nl_queries,
            "chart": item["chart"],
            "hardness": item["hardness"],
            "vql": item["vis_query"]["VQL"].strip(),
            "sql": item["vis_query"]["data_part"]["sql_part"].strip(),
            "binning": item["vis_query"]["data_part"].get("binning", "").strip(),
            "vis_obj": item["vis_obj"],
        }
        records.append(record)
    records.sort(key=lambda x: record_sort_key(x["record_id"]))
    return records


def expand_records_with_all_queries(records: List[Dict]) -> List[Dict]:
    expanded = []
    for record in records:
        queries = record.get("all_queries") or [record["query"]]
        for idx, query in enumerate(queries):
            item = dict(record)
            item["query"] = query.strip()
            item["query_variant_index"] = idx
            expanded.append(item)
    return expanded


def create_splits(records: List[Dict], seed: int, split_source_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    split_path = (split_source_dir or ARTIFACTS_DIR) / "splits.json"
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    ids = [r["record_id"] for r in records]
    labels = [r["chart"] for r in records]
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        ids, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, random_state=seed, stratify=temp_labels
    )
    splits = {
        "train": sorted(train_ids, key=record_sort_key),
        "val": sorted(val_ids, key=record_sort_key),
        "test": sorted(test_ids, key=record_sort_key),
    }
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    return splits


class NVBenchContextDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict],
        metadata: Dict[str, Dict],
        tokenizer,
        variant: str,
        max_source_len: int,
        max_target_len: int,
    ):
        self.records = list(records)
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.variant = variant
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        source = build_source_text(record, self.metadata, self.tokenizer, self.variant)
        target = record["vql"]
        src = self.tokenizer(
            source,
            max_length=self.max_source_len,
            truncation=True,
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            target,
            max_length=self.max_target_len,
            truncation=True,
            return_tensors="pt",
        )
        labels = tgt["input_ids"].squeeze(0)
        return {
            "input_ids": src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "labels": labels,
        }


def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=pad_id,
        )
        labels[labels == pad_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate


def evaluate_predictions(records: List[Dict], predictions: List[str], metadata: Dict[str, Dict]) -> Dict:
    golds = [r["vql"] for r in records]
    norm_pred = [normalize_text(p) for p in predictions]
    norm_gold = [normalize_text(g) for g in golds]
    exact = float(np.mean([p == g for p, g in zip(norm_pred, norm_gold)]))
    chart_acc = float(
        np.mean([parse_vql(p)["chart"] == parse_vql(g)["chart"] for p, g in zip(predictions, golds)])
    )
    slot_scores = [slot_f1(parse_vql(p), parse_vql(g)) for p, g in zip(predictions, golds)]
    bleu = corpus_bleu(predictions, [golds]).score

    cache_conn: Dict[str, sqlite3.Connection] = {}
    exec_matches = []
    exec_total = 0
    for record, pred in zip(records, predictions):
        pred_sql = parse_vql(pred)["sql"]
        gold_sql = parse_vql(record["vql"])["sql"]
        if not pred_sql or not gold_sql:
            continue
        db_id = record["db_id"]
        if db_id not in cache_conn:
            cache_conn[db_id] = sqlite3.connect(str(DATABASE_DIR / db_id / f"{db_id}.sqlite"))
        conn = cache_conn[db_id]
        try:
            pred_rows = fetch_rows(conn, pred_sql)
            gold_rows = fetch_rows(conn, gold_sql)
            exec_matches.append(pred_rows == gold_rows)
            exec_total += 1
        except Exception:
            exec_matches.append(False)
            exec_total += 1
    for conn in cache_conn.values():
        conn.close()

    return {
        "exact_match": exact,
        "chart_accuracy": chart_acc,
        "slot_f1": float(np.mean(slot_scores)),
        "bleu": bleu,
        "execution_accuracy": float(np.mean(exec_matches)) if exec_matches else 0.0,
        "execution_coverage": exec_total,
    }


def generate_predictions(model, tokenizer, dataset: NVBenchContextDataset, records: List[Dict], device, batch_size: int) -> List[str]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=make_collate_fn(tokenizer))
    predictions: List[str] = []
    model.eval()
    for batch in tqdm(loader, desc="Generating", leave=False):
        with torch.no_grad():
            generated = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=196,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(decoded)
    return predictions


def train_variant(
    variant: str,
    train_records: List[Dict],
    val_records: List[Dict],
    test_records: List[Dict],
    metadata: Dict[str, Dict],
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_source_len: int,
    max_target_len: int,
    seed: int,
    grad_accum_steps: int,
    use_amp: bool,
) -> Dict:
    variant_dir = ARTIFACTS_DIR / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] variant={variant}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if tokenizer.pad_token is None and model.config.pad_token_id is not None:
        tokenizer.pad_token = tokenizer.decode([model.config.pad_token_id]).strip()
    if tokenizer.eos_token is None and model.config.eos_token_id is not None:
        tokenizer.eos_token = tokenizer.decode([model.config.eos_token_id]).strip()
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = NVBenchContextDataset(
        train_records, metadata, tokenizer, variant, max_source_len=max_source_len, max_target_len=max_target_len
    )
    val_dataset = NVBenchContextDataset(
        val_records, metadata, tokenizer, variant, max_source_len=max_source_len, max_target_len=max_target_len
    )
    test_dataset = NVBenchContextDataset(
        test_records, metadata, tokenizer, variant, max_source_len=max_source_len, max_target_len=max_target_len
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=make_collate_fn(tokenizer))
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = math.ceil(len(train_loader) / max(1, grad_accum_steps)) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=max(1, total_steps)
    )
    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    best_val = -1.0
    best_path = variant_dir / "best.pt"
    for epoch in range(epochs):
        model.train()
        losses = []
        progress = tqdm(train_loader, desc=f"Train {variant} epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress, start=1):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = outputs.loss / max(1, grad_accum_steps)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            losses.append(loss.item() * max(1, grad_accum_steps))
            if step % max(1, grad_accum_steps) == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 20 == 0:
                progress.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")

        val_predictions = generate_predictions(model, tokenizer, val_dataset, val_records, device, batch_size)
        val_metrics = evaluate_predictions(val_records, val_predictions, metadata)
        print(f"[val] variant={variant} epoch={epoch + 1} metrics={json.dumps(val_metrics, indent=2)}")
        if val_metrics["slot_f1"] > best_val:
            best_val = val_metrics["slot_f1"]
            torch.save(model.state_dict(), best_path)
            with (variant_dir / "val_metrics_best.json").open("w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_predictions = generate_predictions(model, tokenizer, test_dataset, test_records, device, batch_size)
    test_metrics = evaluate_predictions(test_records, test_predictions, metadata)
    print(f"[test] variant={variant} metrics={json.dumps(test_metrics, indent=2)}")

    with (variant_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
        for record, pred in zip(test_records, test_predictions):
            f.write(
                json.dumps(
                    {
                        "record_id": record["record_id"],
                        "db_id": record["db_id"],
                        "query": record["query"],
                        "gold_vql": record["vql"],
                        "pred_vql": pred,
                        "gold_chart": record["chart"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (variant_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    return test_metrics


def build_report(
    config_results: Dict[str, Dict],
    records: List[Dict],
    metadata: Dict[str, Dict],
    model_name: str,
    epochs: int,
    batch_size: int,
    train_all_queries: bool,
    max_source_len: int,
) -> None:
    report_path = ARTIFACTS_DIR / "report.md"
    chart_counts = Counter(r["chart"] for r in records)
    hardness_counts = Counter(r["hardness"] for r in records)
    lines = [
        "# Context Engineering Study on nvBench",
        "",
        "## Dataset Summary",
        f"- Total records: {len(records)}",
        f"- Unique databases used: {len(metadata)}",
        f"- Chart distribution: {dict(chart_counts)}",
        f"- Hardness distribution: {dict(hardness_counts)}",
        "",
        "## Experimental Setup",
        "- Primary source: root `nvBench.json`",
        "- Context source: `database/<db_id>/<db_id>.sqlite`",
        f"- Model: `{model_name}` Transformer seq2seq",
        "- Target: generate full VQL",
        f"- Epochs: `{epochs}`",
        f"- Batch size: `{batch_size}`",
        f"- Train all queries: `{train_all_queries}`",
        f"- Max source length: `{max_source_len}`",
        "- Metrics: Exact Match, Chart Accuracy, Slot F1, BLEU, Execution Accuracy",
        "",
        "## Results",
        "",
        "| Context | Exact Match | Chart Acc | Slot F1 | BLEU | Exec Acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, metrics in config_results.items():
        lines.append(
            f"| {variant} | {metrics['exact_match']:.4f} | {metrics['chart_accuracy']:.4f} | "
            f"{metrics['slot_f1']:.4f} | {metrics['bleu']:.2f} | {metrics['execution_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Context Variants",
            "- `baseline`: query only",
            "- `schema`: query + full table/column schema",
            "- `schema_type`: schema plus semantic column types and foreign keys",
            "- `selected_enriched`: selected relevant tables plus semantic types, sample values, stats, and join hints",
        ]
    )
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    global ARTIFACTS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="t5-small")
    parser.add_argument("--variants", nargs="+", default=["baseline", "schema", "schema_type", "selected_enriched"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-source-len", type=int, default=320)
    parser.add_argument("--max-target-len", type=int, default=196)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--artifacts-dir", default="artifacts_context_study")
    parser.add_argument("--split-source-dir", default="")
    parser.add_argument("--train-all-queries", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--use-amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    ARTIFACTS_DIR = ROOT / args.artifacts_dir
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    split_source_dir = ROOT / args.split_source_dir if args.split_source_dir else None
    print("[data] loading nvBench.json")
    records = load_records()
    splits = create_splits(records, args.seed, split_source_dir=split_source_dir)
    record_map = {r["record_id"]: r for r in records}
    train_records = [record_map[rid] for rid in splits["train"]]
    val_records = [record_map[rid] for rid in splits["val"]]
    test_records = [record_map[rid] for rid in splits["test"]]
    if args.train_all_queries:
        train_records = expand_records_with_all_queries(train_records)
    if args.train_limit:
        train_records = train_records[: args.train_limit]
    if args.val_limit:
        val_records = val_records[: args.val_limit]
    if args.test_limit:
        test_records = test_records[: args.test_limit]
    print(
        f"[data] records train={len(train_records)} val={len(val_records)} test={len(test_records)} "
        f"unique_db={len(set(r['db_id'] for r in records))}"
    )

    metadata = load_or_build_metadata(records)
    results = {}
    for variant in args.variants:
        results[variant] = train_variant(
            variant=variant,
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            metadata=metadata,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            seed=args.seed,
            grad_accum_steps=args.grad_accum_steps,
            use_amp=args.use_amp,
        )
        with (ARTIFACTS_DIR / "all_results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    build_report(
        results,
        records,
        metadata,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_all_queries=args.train_all_queries,
        max_source_len=args.max_source_len,
    )
    print(f"[done] report saved to {ARTIFACTS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
