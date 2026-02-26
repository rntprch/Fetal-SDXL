import argparse
import ast
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from transformers import pipeline


BRAIN_PLANE_MAP = {
    "Trans-thalamic": "Trans-thalamic",
    "Trans-ventricular": "Trans-ventricular",
    "Trans-cerebellum": "Trans-cerebellum",
    "Trans thalamic": "Trans-thalamic",
    "Trans ventricular": "Trans-ventricular",
    "Trans cerebellum": "Trans-cerebellum",
}

CANONICAL_BRAIN_PLANES = {
    "Trans-thalamic",
    "Trans-ventricular",
    "Trans-cerebellum",
}


@contextmanager
def temp_generation(pipe_obj, **overrides):
    gc = pipe_obj.model.generation_config
    missing = object()
    old = {}
    for key, value in overrides.items():
        old[key] = getattr(gc, key, missing)
        setattr(gc, key, value)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is missing:
                try:
                    delattr(gc, key)
                except Exception:
                    pass
            else:
                setattr(gc, key, value)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json_block(text: str) -> Optional[str]:
    if text is None:
        return None

    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return t

    start = t.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return t[start : i + 1]
    return None


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    block = extract_json_block(text)
    if not block:
        return None
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        block2 = re.sub(r",\s*}", "}", block)
        block2 = re.sub(r",\s*]", "]", block2)
        try:
            return json.loads(block2)
        except json.JSONDecodeError:
            return None


def build_brain_prompt(base_prompt: str, plane: str, image_id: Optional[str] = None) -> str:
    plane_norm = BRAIN_PLANE_MAP.get(plane, plane)
    header = []
    if image_id is not None:
        header.append(f"image_id: {image_id}")
    header.append(f"plane (ground truth): {plane_norm}")
    return "\n".join(header) + "\n\n" + base_prompt


def validate_and_normalize(
    obj: Dict[str, Any],
    schema: Dict[str, Any],
    required_fields: List[str],
) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors = []
    norm = obj

    status_enum = set(schema["_schema"]["status_enum"])
    pos_enum = set(schema["_schema"]["position_enum"])
    qual_enum = set(schema["_schema"]["quality_enum"])
    art_enum = set(schema["_schema"]["artifacts_enum"])

    def normalize_position_value(pos: Any) -> Tuple[Any, bool]:
        if pos is None:
            return None, True
        if isinstance(pos, str):
            s = pos.strip()
            if s == "":
                return "", True
            if s in pos_enum:
                return s, True
            if "," in s:
                toks = [t.strip() for t in s.split(",") if t.strip()]
                if toks and all(t in pos_enum for t in toks):
                    return toks, True
            return pos, False
        if isinstance(pos, list):
            toks = [str(t).strip() for t in pos if str(t).strip()]
            if len(toks) == 0:
                return [], True
            if all(t in pos_enum for t in toks):
                return toks, True
            return pos, False
        return pos, False

    for key in required_fields:
        if key not in norm:
            errors.append(f"missing_required_key:{key}")

    arts = norm.get("artifacts")
    if isinstance(arts, list):
        arts2 = [a for a in arts if a in art_enum]
        if "none" in arts2 and len(arts2) > 1:
            arts2 = [a for a in arts2 if a != "none"]
        norm["artifacts"] = arts2 if arts2 else ["none"]
    else:
        norm["artifacts"] = ["none"]

    iq = norm.get("image_quality")
    if iq is None:
        errors.append("missing_image_quality")
    elif iq not in qual_enum:
        errors.append(f"bad_image_quality:{iq}")

    for key, value in norm.items():
        if not (isinstance(value, dict) and set(value.keys()) >= {"status", "evidence", "position"}):
            continue

        st = value.get("status")
        ev = value.get("evidence")
        pos_raw = value.get("position")
        pos_norm, pos_ok = normalize_position_value(pos_raw)
        value["position"] = pos_norm

        if st not in status_enum:
            errors.append(f"bad_status:{key}:{st}")
            continue

        if st in ("present", "partial"):
            if not isinstance(ev, str) or not ev.strip():
                errors.append(f"bad_evidence:{key}")
            if pos_norm in (None, "", []):
                errors.append(f"missing_position:{key}")
            elif not pos_ok:
                errors.append(f"bad_position:{key}:{pos_raw}")
        else:
            if ev is None:
                value["evidence"] = ""
            elif not isinstance(ev, str):
                errors.append(f"bad_evidence_type:{key}")
            if pos_norm not in (None, "", []) and not pos_ok:
                errors.append(f"bad_position:{key}:{pos_raw}")

    return len(errors) == 0, errors, norm


def get_assistant_text_from_generated(generated: Any) -> str:
    def from_messages(messages: Any) -> Optional[str]:
        if isinstance(messages, dict):
            messages = [messages]
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return str(msg.get("content", "")).strip()
        return None

    got = from_messages(generated)
    if got is not None:
        return got

    if isinstance(generated, str):
        s = generated.strip()
        if not s:
            return ""
        try:
            parsed = ast.literal_eval(s)
            got = from_messages(parsed)
            if got is not None:
                return got
        except Exception:
            pass

        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        for ln in reversed(lines):
            try:
                parsed = ast.literal_eval(ln)
            except Exception:
                continue
            got = from_messages(parsed)
            if got is not None:
                return got
        return s

    return str(generated).strip()


def run_pipe_messages(pipe_obj, image, text: str, max_new_tokens: int = 512) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }]

    out = pipe_obj(messages, max_new_tokens=max_new_tokens)
    try:
        gen = out[0]["generated_text"]
    except Exception:
        gen = out
    return get_assistant_text_from_generated(gen)


def brain_required_fields(landmarks_db: Dict[str, Any], plane: str) -> List[str]:
    key = {
        "Trans-thalamic": "brain_trans_thalamic",
        "Trans-ventricular": "brain_trans_ventricular",
        "Trans-cerebellum": "brain_trans_cerebellum",
    }.get(BRAIN_PLANE_MAP.get(plane, plane))
    if not key or key not in landmarks_db:
        raise ValueError(f"Unknown plane for required_fields: {plane}")
    return landmarks_db[key]["required_fields"]


def run_brain_extract_verify(
    pipe_obj,
    image: Image.Image,
    brain_plane: str,
    prompts_db: Dict[str, str],
    landmarks_db: Dict[str, Any],
    image_id: Optional[str] = None,
    max_new_tokens_extract: int = 512,
    max_new_tokens_verify: int = 768,
) -> Dict[str, Any]:
    prompt_a = build_brain_prompt(prompts_db["brain_extract_json_v1"], brain_plane, image_id=image_id)
    with temp_generation(pipe_obj, do_sample=True, temperature=0.1, top_p=0.9, use_cache=True):
        raw_a = run_pipe_messages(pipe_obj, image, prompt_a, max_new_tokens=max_new_tokens_extract)
    obj_a = safe_json_loads(raw_a)

    if obj_a is None:
        return {"ok": False, "stage": "extract_parse_failed", "raw_extract": raw_a}

    req = brain_required_fields(landmarks_db, brain_plane)
    ok_a, errs_a, obj_a_norm = validate_and_normalize(obj_a, landmarks_db, req)

    candidate_json_text = json.dumps(obj_a_norm, ensure_ascii=False)
    prompt_b = prompts_db["brain_verify_and_fix_json_v1"] + "\n\nCANDIDATE_JSON:\n" + candidate_json_text
    with temp_generation(pipe_obj, do_sample=False, use_cache=False):
        raw_b = run_pipe_messages(pipe_obj, image, prompt_b, max_new_tokens=max_new_tokens_verify)
    obj_b = safe_json_loads(raw_b)

    raw_repair = None
    if obj_b is None:
        repair_prompt = (
            "Return ONLY the complete JSON object as VALID JSON. "
            "No extra text. Close all quotes/braces. Do not truncate.\n\n"
            "BROKEN_TEXT:\n" + raw_b
        )
        with temp_generation(pipe_obj, do_sample=False, use_cache=False):
            raw_repair = run_pipe_messages(pipe_obj, image, repair_prompt, max_new_tokens=max_new_tokens_verify)
        obj_b = safe_json_loads(raw_repair)

    if obj_b is None:
        return {
            "ok": False,
            "stage": "verify_parse_failed",
            "raw_extract": raw_a,
            "extract_obj": obj_a_norm,
            "raw_verify": raw_b,
            "raw_verify_repair": raw_repair,
            "extract_validation_ok": ok_a,
            "extract_validation_errors": errs_a,
        }

    ok_b, errs_b, obj_b_norm = validate_and_normalize(obj_b, landmarks_db, req)
    return {
        "ok": ok_b,
        "stage": "done",
        "final_obj": obj_b_norm,
        "final_validation_errors": errs_b,
        "raw_extract": raw_a,
        "raw_verify": raw_b,
        "raw_verify_repair": raw_repair,
        "extract_validation_ok": ok_a,
        "extract_validation_errors": errs_a,
    }


def json_to_sdxl_caption(obj: Dict[str, Any]) -> str:
    parts = []
    parts.append(f"region={obj.get('region', 'Fetal brain')}")
    parts.append(f"plane={obj.get('plane', '')}")
    parts.append(f"quality={obj.get('image_quality', '')}")
    arts = obj.get("artifacts", [])
    if isinstance(arts, list):
        parts.append("artifacts=" + ",".join(arts))
    else:
        parts.append(f"artifacts={arts}")

    for key, value in obj.items():
        if isinstance(value, dict) and "status" in value:
            parts.append(f"{key}={value.get('status', '')}")

    lim = obj.get("limitations_note", "")
    if isinstance(lim, str) and lim.strip() and lim.strip().lower() != "none":
        parts.append("limitations=" + lim.strip().replace(";", ","))
    return "; ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate brain image descriptions for all known planes.")
    parser.add_argument(
        "--csv",
        default="./zenodo_fetal_planes/FETAL_PLANES_DB_data.csv",
        help="Path to FETAL_PLANES_DB_data.csv",
    )
    parser.add_argument(
        "--image-root",
        default="./zenodo_fetal_planes/Images",
        help="Root directory with image files",
    )
    parser.add_argument(
        "--prompts",
        default="./promts.json",
        help="Path to prompts JSON",
    )
    parser.add_argument(
        "--landmarks",
        default="./landmarks.json",
        help="Path to landmarks schema JSON",
    )
    parser.add_argument(
        "--output",
        default="./brain_descriptions_full.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-jsonl",
        default="./brain_descriptions_full.jsonl",
        help="Optional JSONL output path",
    )
    parser.add_argument(
        "--model-id",
        default="lingshu-medical-mllm/Lingshu-32B",
        help="HF model id for image-text-to-text pipeline",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="HF token; defaults to HF_TOKEN env var",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model",
    )
    parser.add_argument(
        "--max-new-tokens-extract",
        type=int,
        default=512,
        help="max_new_tokens for extraction stage",
    )
    parser.add_argument(
        "--max-new-tokens-verify",
        type=int,
        default=768,
        help="max_new_tokens for verify stage",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for quick tests (0 = full dataset)",
    )
    return parser.parse_args()


def normalize_dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def build_filtered_dataframe(csv_path: str, image_root: str) -> pd.DataFrame:
    image_root_path = Path(image_root)
    df = pd.read_csv(csv_path, sep=";")
    df["Image_name"] = df["Image_name"].astype(str).apply(lambda x: str(image_root_path / f"{x}.png"))

    df = df[df["Plane"] != "Other"]
    df = df[df["Plane"] != "Maternal cervix"]
    df = df[df["Brain_plane"] != "Not A Brain"]
    df = df[df["Brain_plane"] != "Other"]
    df = df[df["Plane"] == "Fetal brain"].copy()

    df["Brain_plane_norm"] = df["Brain_plane"].astype(str).map(BRAIN_PLANE_MAP)
    df = df[df["Brain_plane_norm"].isin(CANONICAL_BRAIN_PLANES)].copy()
    return df


def main() -> None:
    args = parse_args()
    if not args.hf_token:
        raise ValueError("HF token is required. Set HF_TOKEN env var or pass --hf-token.")

    prompts_db = load_json(args.prompts)
    landmarks_db = load_json(args.landmarks)
    df = build_filtered_dataframe(args.csv, args.image_root)

    if args.limit > 0:
        df = df.head(args.limit).copy()

    print(f"Filtered rows to process: {len(df)}")
    print(df["Brain_plane_norm"].value_counts())

    pipe_obj = pipeline(
        task="image-text-to-text",
        model=args.model_id,
        token=args.hf_token,
        device_map=args.device_map,
        dtype=normalize_dtype(args.dtype),
        use_fast=False,
        model_kwargs={"low_cpu_mem_usage": True},
    )

    rows_out = []
    for idx, row in df.iterrows():
        image_path = row["Image_name"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            rows_out.append(
                {
                    "image_path": image_path,
                    "full_decs": "",
                    "cap": "",
                    "error": str(e),
                }
            )
            print(f"[{idx}] image_open_failed: {image_path} ({e})")
            continue

        result = run_brain_extract_verify(
            pipe_obj=pipe_obj,
            image=image,
            brain_plane=row["Brain_plane_norm"],
            prompts_db=prompts_db,
            landmarks_db=landmarks_db,
            image_id=str(image_path),
            max_new_tokens_extract=args.max_new_tokens_extract,
            max_new_tokens_verify=args.max_new_tokens_verify,
        )

        final_obj = result.get("final_obj", {})
        desc_json = json.dumps(final_obj, ensure_ascii=False) if final_obj else ""
        caption = json_to_sdxl_caption(final_obj) if final_obj else ""

        out_row = {
            "image_path": image_path,
            "full_decs": desc_json,
            "cap": caption,
            "error": "",
        }
        if not result.get("ok", False):
            stage = result.get("stage", "unknown_error")
            out_row["error"] = f"generation_failed:{stage}"
        rows_out.append(out_row)

        print(f"[{idx}] ok={result.get('ok', False)} stage={result.get('stage', '')} path={image_path}")

    out_df = pd.DataFrame(rows_out)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    if args.output_jsonl:
        jsonl_path = Path(args.output_jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in rows_out:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved CSV: {args.output}")
    if args.output_jsonl:
        print(f"Saved JSONL: {args.output_jsonl}")
    print(f"Done. total={len(out_df)}")


if __name__ == "__main__":
    main()
