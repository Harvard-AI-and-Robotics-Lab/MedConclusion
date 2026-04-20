import json
import os
import re
import argparse
import math
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import threading
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

SCORE_KEYS = [
    "semantic similarity",
    "writing style similarity",
    "contradiction rate",
    "numeric consistency",
    "formality similarity",
]

RULE_BASED_KEYS = [
    "word_count_original",
    "word_count_generated",
    "word_count_ratio",
    "sentence_count_original",
    "sentence_count_generated",
    "sentence_count_ratio",
    "embedding_cosine_similarity",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "bleu",
    "perplexity_original",
    "perplexity_generated",
]

def count_words(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))

def count_sentences(text: str) -> int:
    sentences = re.split(r'[.!?]+', text.strip())
    return len([s for s in sentences if s.strip()])

def compute_rouge_bleu(original: str, generated: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F-measures and BLEU score."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(original, generated)

    # BLEU with smoothing to handle short texts
    ref_tokens = nltk.word_tokenize(original.lower())
    hyp_tokens = nltk.word_tokenize(generated.lower())
    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)

    return {
        "rouge_1": rouge_scores['rouge1'].fmeasure,
        "rouge_2": rouge_scores['rouge2'].fmeasure,
        "rouge_l": rouge_scores['rougeL'].fmeasure,
        "bleu": bleu,
    }

def compute_perplexity(text: str, model, tokenizer, device="cpu") -> float:
    """Compute perplexity score using GPT-2."""
    if not text or not text.strip():
        return None
    try:
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        if input_ids.size(1) == 0:
            return None
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        return float(perplexity.item())
    except Exception as e:
        print(f"Perplexity computation error: {e}")
        return None

def get_evaluation_prompt(original_conclusion, generated_conclusion):
    prompt = f"""You are an expert evaluator of scientific writing. Your task is to compare the Generated Conclusion against the Original (Reference) Conclusion and score multiple dimensions from 0 to 100 (decimals allowed). Use ONLY the two conclusions provided. Do NOT provide any explanations.

Scoring dimensions:
- semantic similarity: How similar the meaning and core claims are.
- writing style similarity: How similar the tone, phrasing, structure, and rhetorical style are.
- contradiction rate: Degree of contradiction between the Generated Conclusion and the Original Conclusion.
  - 100 = no contradiction
  - 0 = severe contradiction
- numeric consistency: Consistency of all numerical information, quantities, directions, and magnitudes.
  - 100 = fully consistent or no numeric content in either text
  - 0 = major numeric inconsistency
- formality similarity: Similarity in academic/formal writing level and register.

INPUT:
- Original Conclusion (reference): {original_conclusion}
- Generated Conclusion: {generated_conclusion}

OUTPUT FORMAT (STRICT):
{{"semantic similarity": <0-100>, "writing style similarity": <0-100>, "contradiction rate": <0-100>, "numeric consistency": <0-100>, "formality similarity": <0-100>}}
"""
    return prompt

def parse_json_response(response_text):
    """
    Attempt to parse the JSON response from the LLM.
    Handles potential markdown code blocks.
    """
    try:
        text = response_text.strip()
    except:
        return None
    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    try:
        text = text.strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def evaluate_single_line(line, args, client, embed_model=None, ppl_model=None, ppl_tokenizer=None, ppl_device=None):
    """
    Process a single line: parse, generate evaluation, return updated record (dict).
    Returns None if the line couldn't be parsed at all.
    """
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    
    original_conclusion = record.get("original_conclusion")
    generated_conclusion = record.get("generated_conclusion")
    
    if not original_conclusion or not generated_conclusion:
       # If missing data, record error but keep entry
       evaluation_response = ""
       record["evaluation_error"] = "No original_conclusion or generated_conclusion"
       record["raw_evaluation_response"] = evaluation_response
       return record

    prompt = get_evaluation_prompt(original_conclusion, generated_conclusion)
    
    max_retries = 3
    retry_count = 0
    evaluation_response = ""
    scores = None
    
    while retry_count < max_retries:
        try:
            try:
                response = client.chat.completions.create(
                    model=args.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=args.temperature,
                    seed=args.seed,
                    response_format={ "type": "json_object" },
                    extra_body={
                        "reasoning": {
                            "effort": "minimal"
                        }
                    },
                )
            except:
                response = client.chat.completions.create(
                    model=args.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=args.temperature,
                    seed=args.seed,
                    response_format={ "type": "json_object" }
                )
            evaluation_response = response.choices[0].message.content

            print('called api')
            
            scores = parse_json_response(evaluation_response)
            if scores:
                break
            else:
                # Failed to parse
                print(f"Failed to parse JSON response: {evaluation_response}")
                retry_count += 1
                # if retry_count < max_retries: time.sleep(1)
                
        except Exception as e:
            # API Error
            print(f"Retry {retry_count+1}/{max_retries}: API Error - {e}")
            retry_count += 1
            # if retry_count < max_retries: time.sleep(1)

    record["retry_count"] = retry_count
    
    if scores:
        record["evaluation"] = scores
        record["judge_model"] = args.model_name
    else:
        record["evaluation_error"] = "Failed after retries"
        record["raw_evaluation_response"] = evaluation_response

    # Rule-based metrics
    orig_word_count = count_words(original_conclusion)
    gen_word_count = count_words(generated_conclusion)
    orig_sen_count = count_sentences(original_conclusion)
    gen_sen_count = count_sentences(generated_conclusion)

    record["rule_based"] = {
        "word_count_original": orig_word_count,
        "word_count_generated": gen_word_count,
        "word_count_ratio": gen_word_count / orig_word_count if orig_word_count > 0 else 0.0,
        "sentence_count_original": orig_sen_count,
        "sentence_count_generated": gen_sen_count,
        "sentence_count_ratio": gen_sen_count / orig_sen_count if orig_sen_count > 0 else 0.0,
    }

    # Embedding cosine similarity
    if embed_model is not None:
        try:
            embeddings = embed_model.encode([original_conclusion, generated_conclusion])
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            record["rule_based"]["embedding_cosine_similarity"] = float(sim)
        except Exception as e:
            print(f"Embedding error: {e}")
            record["rule_based"]["embedding_cosine_similarity"] = None

    # ROUGE & BLEU
    try:
        rb_scores = compute_rouge_bleu(original_conclusion, generated_conclusion)
        record["rule_based"].update(rb_scores)
    except Exception as e:
        print(f"ROUGE/BLEU error: {e}")
        for k in ("rouge_1", "rouge_2", "rouge_l", "bleu"):
            record["rule_based"][k] = None

    # Perplexity
    if ppl_model is not None and ppl_tokenizer is not None:
        try:
            record["rule_based"]["perplexity_original"] = compute_perplexity(original_conclusion, ppl_model, ppl_tokenizer, ppl_device)
            record["rule_based"]["perplexity_generated"] = compute_perplexity(generated_conclusion, ppl_model, ppl_tokenizer, ppl_device)
        except Exception as e:
            print(f"Perplexity error: {e}")
            record["rule_based"]["perplexity_original"] = None
            record["rule_based"]["perplexity_generated"] = None

    return record

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated conclusions using an LLM judge.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file containing generated conclusions.")
    parser.add_argument("--model_name", type=str, default="gpt-5.4-mini", help="Name of the model to use for judging (e.g., gpt-4o, gemini-1.5-pro).")
    parser.add_argument("--output_dir", type=str, default="output/judging", help="Directory to save the evaluation results.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for evaluation.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file if it exists.")
    parser.add_argument("--backfill", action="store_true", help="Only append new rule-based metrics (ROUGE/BLEU) to existing output, no LLM calls.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of concurrent threads.")
    
    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv(override=True)

    # Prepare output file path early so we can check for backfill
    os.makedirs(args.output_dir, exist_ok=True)
    input_filename = os.path.basename(args.input_file)
    output_filename = f"eval_{input_filename}"
    output_path = os.path.join(args.output_dir, output_filename)

    # ------------------------------------------------------------------
    # Backfill mode: only append ROUGE/BLEU metrics to existing output,
    # no LLM calls, no old rule-based metric recomputation.
    # ------------------------------------------------------------------
    if args.backfill:
        if not os.path.exists(output_path):
            print(f"Error: output file '{output_path}' does not exist. Nothing to backfill.")
            return
        # Check if already backfilled
        needs_backfill = False
        try:
            with open(output_path, "r", encoding="utf-8") as f_check:
                for check_line in f_check:
                    check_line = check_line.strip()
                    if not check_line:
                        continue
                    try:
                        check_rec = json.loads(check_line)
                        if check_rec.get("summary_stats"):
                            continue
                        rb = check_rec.get("rule_based", {})
                        if "rouge_1" not in rb or "perplexity_generated" not in rb:
                            needs_backfill = True
                        break  # only need to check the first data line
                    except json.JSONDecodeError:
                        continue
        except Exception:
            needs_backfill = False

        if needs_backfill:
            print("Backfill mode: adding new rule-based metrics to existing output (no LLM calls)...")
            
            ppl_device = "cuda" if torch.cuda.is_available() else "cpu"
            print("Loading GPT-2 model for perplexity...")
            try:
                ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").to(ppl_device)
                ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                print("GPT-2 loaded.")
            except Exception as e:
                print(f"Skipping perplexity due to error loading GPT-2: {e}")
                ppl_model = None
                ppl_tokenizer = None
                
            _backfill_rule_based_metrics(output_path, ppl_model, ppl_tokenizer, ppl_device)
            print("Backfill complete.")
        else:
            print("Output file already contains updated rule-based metrics. Nothing to do.")
        return

    # Setup Clients
    is_openai = "gpt" in args.model_name.lower()
    client = None
    
    if is_openai:
        # OpenAI GPT models — use OPENAI_API_KEY
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not found in .env file.")
            return
        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            return
    else:
        # All other models (including Gemini) — use OPENROUTER_API_KEY via OpenAI client
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("Error: OPENROUTER_API_KEY not found in .env file.")
            return
        try:
            print("using openrouter")
            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        except Exception as e:
            print(f"Error initializing OpenRouter client: {e}")
            return

    # Load embedding model
    print("Loading embedding model sentence-transformers/all-mpnet-base-v2...")
    embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print("Embedding model loaded.")

    # Load GPT-2 for perplexity
    ppl_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading GPT-2 model for perplexity...")
    try:
        ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").to(ppl_device)
        ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        print("GPT-2 loaded.")
    except Exception as e:
        print(f"Skipping perplexity due to error loading GPT-2: {e}")
        ppl_model = None
        ppl_tokenizer = None

    success_count = 0
    total_count = 0

    import statistics
    import time

    all_scores = {key: [] for key in SCORE_KEYS}

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        if args.num_samples is not None:
            lines = lines[args.start_index:args.start_index + args.num_samples]
        if args.start_index is not None and args.num_samples is None:
            lines = lines[args.start_index:]

        # Prepare for resume if requested
        processed_pmids = set()
        file_mode = "w"

        if args.resume and os.path.exists(output_path):
            print(f"Resuming from {output_path}...")
            file_mode = "a"
            try:
                with open(output_path, "r", encoding="utf-8") as existing_f:
                    for line_existing in existing_f:
                        line_existing = line_existing.strip()
                        if not line_existing:
                            continue
                        try:
                            rec = json.loads(line_existing)
                            # Skip summary stats lines
                            if rec.get("summary_stats"):
                                continue
                            if "pmid" in rec:
                                processed_pmids.add(rec["pmid"])
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                print(f"Warning: Could not read existing output file for resume: {e}")

            print(f"Found {len(processed_pmids)} already processed records.")

            # Clean out old summary stats from the file before appending
            try:
                with open(output_path, "r", encoding="utf-8") as f_in:
                    existing_lines = f_in.readlines()
                cleaned_lines = []
                for el in existing_lines:
                    try:
                        data = json.loads(el)
                        if not data.get("summary_stats"):
                            cleaned_lines.append(el)
                    except json.JSONDecodeError:
                        cleaned_lines.append(el)
                with open(output_path, "w", encoding="utf-8") as f_out:
                    f_out.writelines(cleaned_lines)
            except Exception:
                pass
        else:
            print(f"Saving results to {output_path}")

        with open(output_path, file_mode, encoding="utf-8") as out_f:
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # Submit only tasks not already processed
                future_to_line = {}
                for line in lines:
                    try:
                        rec_check = json.loads(line)
                        pmid_check = rec_check.get("pmid")
                        if pmid_check in processed_pmids:
                            continue
                    except:
                        pass
                    future = executor.submit(evaluate_single_line, line, args, client, embed_model, ppl_model, ppl_tokenizer, ppl_device)
                    future_to_line[future] = line

                print(f"Submitting {len(future_to_line)} new tasks...")
                
                # As they complete, write to file in order of completion (or strict order if we wanted, but usage doesn't strictly require input order preservation, though nicer. 'as_completed' is simpler for throughput).
                # Actually, iterating futures in submission order and waiting for them maintains order. 
                # iterating as_completed does NOT maintain order. 
                # Let's use as_completed for speed and feedback.
                
                for future in tqdm(concurrent.futures.as_completed(future_to_line), total=len(future_to_line), desc="Judging"):
                    try:
                        record = future.result()
                    except Exception as e:
                        print(f"Unhandled exception in thread: {e}")
                        continue
                        
                    if not record:
                        continue
                    
                    total_count += 1
                    if "evaluation" in record:
                        success_count += 1
                        scores = record["evaluation"]
                        for key in SCORE_KEYS:
                            val = scores.get(key)
                            if isinstance(val, (int, float)):
                                all_scores[key].append(val)
                
                    json.dump(record, out_f, ensure_ascii=False)
                    out_f.write("\n")
                    out_f.flush()
            
            # Calculate and save summary stats by reading all lines of the file
            with open(output_path, "r") as f:
                lines = f.readlines()
            
            all_scores = {key: [] for key in SCORE_KEYS}
            all_rule_based = {key: [] for key in RULE_BASED_KEYS}
            
            for line in lines:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                scores = record.get("evaluation")
                if scores:
                    for key in SCORE_KEYS:
                        val = scores.get(key)
                        if isinstance(val, (int, float)):
                            all_scores[key].append(val if math.isfinite(val) else 0.0)
                
                rb = record.get("rule_based")
                if rb:
                    for key in RULE_BASED_KEYS:
                        val = rb.get(key)
                        if isinstance(val, (int, float)):
                            all_rule_based[key].append(val if math.isfinite(val) else 0.0)
            
            # Build summary if we have any scores
            all_combined = {**all_scores, **all_rule_based}
            if any(len(v) > 0 for v in all_combined.values()):
                summary = {"summary_stats": True}
                for key in SCORE_KEYS:
                    vals = all_scores[key]
                    if vals:
                        summary[key.replace(" ", "_")] = {
                            "min": min(vals),
                            "max": max(vals),
                            "mean": statistics.mean(vals),
                            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                            "median": statistics.median(vals)
                        }
                for key in RULE_BASED_KEYS:
                    vals = all_rule_based[key]
                    if vals:
                        summary[key] = {
                            "min": min(vals),
                            "max": max(vals),
                            "mean": statistics.mean(vals),
                            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                            "median": statistics.median(vals)
                        }
                summary["count"] = max(len(v) for v in all_combined.values() if len(v) > 0)
                json.dump(summary, out_f, ensure_ascii=False)
                out_f.write("\n")
                out_f.flush()
                print("\nSummary statistics saved to the end of the file.")
                print(json.dumps(summary, indent=2))

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
        return

    print(f"Evaluation complete. {success_count}/{total_count} records evaluated successfully.")


def _backfill_rule_based_metrics(output_path: str, ppl_model=None, ppl_tokenizer=None, ppl_device=None):
    """
    Read an existing output JSONL, compute missing rule-based metrics (ROUGE/BLEU/Perplexity)
    for every data line that is missing them, then rewrite the file with updated records and
    refreshed summary stats.
    """
    import statistics

    with open(output_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    updated_lines = []
    for line in tqdm(raw_lines, desc="Backfilling Rule-Based Metrics"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            updated_lines.append(line)
            continue

        # Skip summary stats — we'll regenerate them
        if record.get("summary_stats"):
            continue

        orig = record.get("original_conclusion", "")
        gen = record.get("generated_conclusion", "")

        if orig and gen:
            rb = record.setdefault("rule_based", {})
            if "rouge_1" not in rb:
                try:
                    rb.update(compute_rouge_bleu(orig, gen))
                except Exception as e:
                    print(f"ROUGE/BLEU error for pmid={record.get('pmid')}: {e}")
                    for k in ("rouge_1", "rouge_2", "rouge_l", "bleu"):
                        rb[k] = None

            if "perplexity_generated" not in rb and ppl_model is not None and ppl_tokenizer is not None:
                try:
                    rb["perplexity_original"] = compute_perplexity(orig, ppl_model, ppl_tokenizer, ppl_device)
                    rb["perplexity_generated"] = compute_perplexity(gen, ppl_model, ppl_tokenizer, ppl_device)
                except Exception as e:
                    print(f"Perplexity error for pmid={record.get('pmid')}: {e}")
                    rb["perplexity_original"] = None
                    rb["perplexity_generated"] = None

        updated_lines.append(json.dumps(record, ensure_ascii=False))

    # Recompute summary stats over ALL metrics
    all_scores = {key: [] for key in SCORE_KEYS}
    all_rule_based = {key: [] for key in RULE_BASED_KEYS}

    for line in updated_lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        scores = record.get("evaluation")
        if scores:
            for key in SCORE_KEYS:
                val = scores.get(key)
                if isinstance(val, (int, float)):
                    all_scores[key].append(val if math.isfinite(val) else 0.0)
        rb = record.get("rule_based")
        if rb:
            for key in RULE_BASED_KEYS:
                val = rb.get(key)
                if isinstance(val, (int, float)):
                    all_rule_based[key].append(val if math.isfinite(val) else 0.0)

    all_combined = {**all_scores, **all_rule_based}
    if any(len(v) > 0 for v in all_combined.values()):
        summary = {"summary_stats": True}
        for key in SCORE_KEYS:
            vals = all_scores[key]
            if vals:
                summary[key.replace(" ", "_")] = {
                    "min": min(vals),
                    "max": max(vals),
                    "mean": statistics.mean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "median": statistics.median(vals)
                }
        for key in RULE_BASED_KEYS:
            vals = all_rule_based[key]
            if vals:
                summary[key] = {
                    "min": min(vals),
                    "max": max(vals),
                    "mean": statistics.mean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "median": statistics.median(vals)
                }
        summary["count"] = max(len(v) for v in all_combined.values() if len(v) > 0)
        updated_lines.append(json.dumps(summary, ensure_ascii=False))
        print("\nSummary statistics:")
        print(json.dumps(summary, indent=2))

    # Write back
    with open(output_path, "w", encoding="utf-8") as f:
        for line in updated_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
