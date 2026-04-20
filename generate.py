import json
import os
import re
import argparse
import unicodedata
from openai import OpenAI
import concurrent.futures
import threading
from tqdm import tqdm
from dotenv import load_dotenv

CONCLUSION_LABELS = [
    "CONCLUSION",
    "CONCLUSIONS",
    "CONCLUSION(S)",
    "CONCLUSIONS AND RELEVANCE",
    "CONCLUSION AND RELEVANCE",
    "CONCLUSIONS AND IMPLICATIONS",
    "CONCLUSION AND IMPLICATIONS",
    "CONCLUSIONS AND IMPORTANCE",
    "CONCLUSION AND IMPORTANCE",
    "CONCLUSION AND SIGNIFICANCE",
    "CONCLUSIONS AND SIGNIFICANCE",
    "CONCLUSION AND INTERPRETATION",
    "CONCLUSIONS AND INTERPRETATION",
    "CONCLUSIONS AND CLINICAL RELEVANCE",
    "CONCLUSION AND CLINICAL RELEVANCE",
    "CONCLUSIONS AND CLINICAL IMPORTANCE",
    "CONCLUSION AND CLINICAL IMPORTANCE",
    "AUTHORS' CONCLUSIONS",
    "AUTHORS' CONCLUSION",
    "MAIN CONCLUSIONS",
    "MAIN CONCLUSION",
]

def count_sentences(text: str) -> int:
    sentences = re.split(r'[.!?]+', text.strip())
    return len([s for s in sentences if s.strip()])

def count_words(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))

def is_conclusion(section):
    """Check if a section is a conclusion based on label."""
    label = section.get("label", "").upper().strip()
    return label in CONCLUSION_LABELS

def remove_duplicates(abstract_list):
    """
    Check if the abstract list is a perfect duplication (first half == second half).
    If so, return only the first half.
    """
    n = len(abstract_list)
    if n > 0 and n % 2 == 0:
        half = n // 2
        first_half = abstract_list[:half]
        second_half = abstract_list[half:]
        
        # Compare content. Labels might slightly differ case-wise, so we check content mainly or exact dict eq
        # The user example suggests exact duplication. Let's try exact first.
        if first_half == second_half:
            return first_half
            
    return abstract_list

def format_abstract(abstract_list):
    """Format abstract sections into a string, excluding CONCLUSIONS."""
    formatted_text = ""
    
    # Pre-process to remove duplicates if any
    clean_list = remove_duplicates(abstract_list)
    
    for section in clean_list:
        if is_conclusion(section):
            continue
        label = section.get("label", "SECTION")
        text = section.get("text", "")
        # Normalize unicode characters to NFKC form (e.g. \xa0 -> space)
        text = unicodedata.normalize("NFKC", text)
        formatted_text += f"{label}: {text}\n\n"
    return formatted_text.strip()

def get_conclusion_stats(abstract_list):
    """Extract sentence and word counts from the CONCLUSIONS section."""
    # Also use clean list to find conclusion, though usually conclusion is at the end.
    # If duplicated, we just find the valid conclusion in the clean list.
    clean_list = remove_duplicates(abstract_list)
    
    for section in clean_list:
        if is_conclusion(section):
            text = section.get("text", "")
            return count_sentences(text), count_words(text), text
    return 0, 0, None

def generate_conclusion_prompt(abstract_text, sen_num, word_num):
    """Construct the prompt with dynamic constraints."""
    prompt = f"""You are a senior scientist in generating conclusions for scientific papers.

You will be provided with a structured abstract of a scientific paper.
The abstract contains sections such as Background, Objective, Methods, Results, etc., but the corresponding Conclusion section is missing.

Your task is to infer and write the most plausible CONCLUSION section that would appear in this abstract.

Here are the requirements:
- Output ONLY the text for the conclusion itself. Do NOT include any section headers or explanations.
- Only use the information provided by the abstract to derive your conclusion. 
- Do NOT introduce new experiments, datasets, numerical values, or claims that are not supported by the abstract.
- Use formal academic writing style.
- The conclusion should be a concise paragraph with {sen_num} sentences totalling {word_num} words.

Structured Abstract:
<Abstract>
{abstract_text}
</Abstract>
"""
    return prompt

def generate_conclusion_prompt_writing_style(abstract_text, sen_num, word_num):
    """Construct the prompt with dynamic constraints."""
    prompt = f"""You are a senior scientist in generating conclusions for scientific papers.

You will be provided with a structured abstract of a scientific paper.
The abstract contains sections such as Background, Objective, Methods, Results, etc., but the corresponding Conclusion section is missing.

Your task is to infer and write the most plausible CONCLUSION section that would appear in this abstract.

Here are the requirements:
- Output ONLY the text for the conclusion itself. Do NOT include any section headers or explanations.
- Only use the information provided by the abstract to derive your conclusion. 
- Do NOT introduce new experiments, datasets, numerical values, or claims that are not supported by the abstract.
- Please generate the conclusion in the same writing style as the given abstract sections.
- The conclusion should be a concise paragraph with {sen_num} sentences totalling {word_num} words.

Structured Abstract:
<Abstract>
{abstract_text}
</Abstract>
"""
    return prompt


def generate_conclusion_prompt_sentence_num(abstract_text, sen_num, word_num):
    """Construct the prompt with dynamic constraints."""
    prompt = f"""You are a senior scientist in generating conclusions for scientific papers.

You will be provided with a structured abstract of a scientific paper.
The abstract contains sections such as Background, Objective, Methods, Results, etc., but the corresponding Conclusion section is missing.

Your task is to infer and write the most plausible CONCLUSION section that would appear in this abstract.

Here are the requirements:
- Output ONLY the text for the conclusion itself. Do NOT include any section headers or explanations.
- Only use the information provided by the abstract to derive your conclusion. 
- Do NOT introduce new experiments, datasets, numerical values, or claims that are not supported by the abstract.
- Use formal academic writing style.
- The conclusion should be a concise paragraph with {sen_num} sentences.

Structured Abstract:
<Abstract>
{abstract_text}
</Abstract>
"""
    return prompt

def generate_conclusion_prompt_no_restriction(abstract_text, sen_num, word_num):
    """Construct the prompt with dynamic constraints."""
    prompt = f"""You are a senior scientist in generating conclusions for scientific papers.

You will be provided with a structured abstract of a scientific paper.
The abstract contains sections such as Background, Objective, Methods, Results, etc., but the corresponding Conclusion section is missing.

Your task is to infer and write the most plausible CONCLUSION section that would appear in this abstract.

Here are the requirements:
- Output ONLY the text for the conclusion itself. Do NOT include any section headers or explanations.
- Only use the information provided by the abstract to derive your conclusion. 
- Do NOT introduce new experiments, datasets, numerical values, or claims that are not supported by the abstract.
- Use formal academic writing style.

Structured Abstract:
<Abstract>
{abstract_text}
</Abstract>
"""

    return prompt



def generate_summarization_prompt(abstract_text, sen_num, word_num):
    """Construct the prompt with dynamic constraints."""
    prompt = f"""You are a senior scientist in generating summaries for scientific papers.

You will be provided with a structured abstract of a scientific paper.
The abstract contains sections such as Background, Objective, Methods, Results, etc.

Your task is to summarize the core information in the given abstract sections.

Here are the requirements:
- Output ONLY the text for the summary itself. Do NOT include any section headers or explanations.
- Only use the information provided by the abstract to derive your summary. 
- Do NOT introduce new experiments, datasets, numerical values, or claims that are not supported by the abstract.
- Use formal academic writing style.
- The summary should be a concise paragraph with {sen_num} sentences totalling {word_num} words.

Structured Abstract:
<Abstract>
{abstract_text}
</Abstract>
"""

    return prompt


def generate_summarization_prompt_writing_style(abstract_text, sen_num, word_num):
    """Construct the prompt with dynamic constraints."""
    prompt = f"""You are a senior scientist in generating summaries for scientific papers.

You will be provided with a structured abstract of a scientific paper.
The abstract contains sections such as Background, Objective, Methods, Results, etc.

Your task is to summarize the core information in the given abstract sections.

Here are the requirements:
- Output ONLY the text for the summary itself. Do NOT include any section headers or explanations.
- Only use the information provided by the abstract to derive your summary. 
- Do NOT introduce new experiments, datasets, numerical values, or claims that are not supported by the abstract.
- Please generate the summary in the same writing style as the given abstract sections.
- The summary should be a concise paragraph with {sen_num} sentences totalling {word_num} words.

Structured Abstract:
<Abstract>
{abstract_text}
</Abstract>
"""
    return prompt


def generate_summarization_prompt_no_restriction(abstract_text, sen_num, word_num):
    """Construct the prompt with dynamic constraints."""
    
    prompt = f"""You are a senior scientist in generating summaries for scientific papers.

You will be provided with a structured abstract of a scientific paper.
The abstract contains sections such as Background, Objective, Methods, Results, etc.

Your task is to summarize the core information in the given abstract sections.

Here are the requirements:
- Output ONLY the text for the summary itself. Do NOT include any section headers or explanations.
- Only use the information provided by the abstract to derive your summary. 
- Do NOT introduce new experiments, datasets, numerical values, or claims that are not supported by the abstract.
- Use formal academic writing style.

Structured Abstract:
<Abstract>
{abstract_text}
</Abstract>
"""
    return prompt



def generate_prompt(mode, abstract_text, sen_num, word_num):
    if mode == "conclusion":
        return generate_conclusion_prompt(abstract_text, sen_num, word_num)
    elif mode == "conclusion_writing":
        return generate_conclusion_prompt_writing_style(abstract_text, sen_num, word_num)
    elif mode == "summary":
        return generate_summarization_prompt(abstract_text, sen_num, word_num)
    elif mode == "summary_writing":
        return generate_summarization_prompt_writing_style(abstract_text, sen_num, word_num)
    elif mode == "conclusion_sentence_num":
        return generate_conclusion_prompt_sentence_num(abstract_text, sen_num, word_num)
    elif mode == "conclusion_no_restriction":
        return generate_conclusion_prompt_no_restriction(abstract_text, sen_num, word_num)
    elif mode == "summary_no_restriction":
        return generate_summarization_prompt_no_restriction(abstract_text, sen_num, word_num)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def process_line(line, args, client):
    """
    Process a single line (record) from the input JSONL.
    Returns the JSON-serializable dictionary to write to the output, or None if failed/skipped.
    """
    try:
        record = json.loads(line)
        pmid = record.get("pmid")
        
        abstract_list = record.get("abstract", [])
        
        # Calculate stats dynamically
        sen_num, word_num, original_conclusion = get_conclusion_stats(abstract_list)
        
        abstract_text = format_abstract(abstract_list)
        
        if not abstract_text:
            print(f"Skipping PMID {pmid}: No abstract text found.")
            return None

        prompt = generate_prompt(args.mode, abstract_text, sen_num, word_num)
        
        generated_conclusion = ""
        
        # OpenAI-compatible API Call (works for both OpenAI and OpenRouter)
        try:
            try:
                response = client.chat.completions.create(
                    model=args.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                temperature=0.7,
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
                        {"role": "user", "content": prompt}
                    ],
                temperature=0.7,
            )
            generated_conclusion = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API Error for PMID {pmid}: {e}")
            return None
        
        # Calculate stats for the generated conclusion
        gen_sen_num = count_sentences(generated_conclusion)
        gen_word_num = count_words(generated_conclusion)
        
        output_record = {
            "pmid": pmid,
            "prompt": prompt,
            "generated_conclusion": generated_conclusion,
            "original_conclusion": original_conclusion,
            "target_sentences": sen_num,
            "target_words": word_num,
            "generated_sentences": gen_sen_num,
            "generated_words": gen_word_num
        }
        return output_record

    except Exception as e:
        print(f"Error processing line: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate conclusions for scientific papers (Sample 1).")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use (e.g., gpt-4o, gemini-1.5-flash).")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate.")
    parser.add_argument("--mode", type=str, default="conclusion", choices=["conclusion", "conclusion_writing", "summary", "summary_writing", "conclusion_sentence_num", "conclusion_no_restriction", "summary_no_restriction"], help="Mode to generate (conclusion, writing_style, or summary).")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of threads for concurrent execution.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file if it exists.")
    
    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv(override=True)

    # Determine provider based on model name
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

    print(f"Processing {args.input_file} using {args.model_name}...")
    
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()[:args.num_samples]
            
        # Prepare for resume if requested
        processed_pmids = set()
        file_mode = "w"
        
        if args.resume and os.path.exists(args.output_file):
            print(f"Resuming from {args.output_file}...")
            file_mode = "a"
            try:
                with open(args.output_file, "r", encoding="utf-8") as existing_f:
                    for line in existing_f:
                        line = line.strip()
                        if not line: continue
                        try:
                            rec = json.loads(line)
                            if "pmid" in rec:
                                processed_pmids.add(rec["pmid"])
                        except json.JSONDecodeError:
                            # Might be partial line at end, ignore
                            pass
            except Exception as e:
                print(f"Warning: Could not read existing output file for resume: {e}")
                
            print(f"Found {len(processed_pmids)} already processed records.")
            
        # Open output file for writing (append mode or write mode)
        with open(args.output_file, file_mode, encoding="utf-8") as out_f:
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # Submit all tasks (only those not processed)
                future_to_line = {}
                for line in lines:
                    try:
                        # Quick parse to check PMID before submission (optimization)
                        # We need to parse anyway to call process_line, but process_line does parsing.
                        # Ideally we parse here to check PMID.
                        rec_check = json.loads(line)
                        pmid_check = rec_check.get("pmid")
                        if pmid_check in processed_pmids:
                            continue
                    except:
                        # If we can't parse input line, process_line will fail too, but let's submit it to handle error there or skip
                        pass
                        
                    future = executor.submit(process_line, line, args, client)
                    future_to_line[future] = line
                
                print(f"Submitting {len(future_to_line)} new tasks...")
                
                # As they complete, write to file
                for future in tqdm(concurrent.futures.as_completed(future_to_line), total=len(future_to_line), desc="Generating Conclusions"):
                    result_record = future.result()
                    if result_record:
                        try:
                            json.dump(result_record, out_f, ensure_ascii=False)
                            out_f.write("\n")
                            out_f.flush() # Ensure written to disk
                        except Exception as e:
                            print(f"Error writing record to file: {e}")

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")

    print(f"Done. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
