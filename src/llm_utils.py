import json
import os
from datetime import datetime

from openai import OpenAI

try:
    from .vlm_config import API_KEY, BASE_URL
except (ImportError, ValueError):
    from vlm_config import API_KEY, BASE_URL

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

def _log_llm(request_messages, response_content=None, error=None):
    try:
        # Re-using the same log file or a new one for llm checks
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_results", "llm_run_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().isoformat()}] --- LLM Request ---\n")
            f.write(json.dumps(request_messages, indent=2, ensure_ascii=False) + "\n")
            if response_content:
                f.write(f"--- LLM Response ---\n{response_content}\n")
            if error:
                f.write(f"--- LLM Error ---\n{str(error)}\n")
    except Exception as e:
        print(f"Failed to log LLM call: {e}")

def llm_text_quality(ground_truth_path, test_path):
    try:
        with open(ground_truth_path, encoding="utf-8") as f:
            gt_text = f.read()
        with open(test_path, encoding="utf-8") as f:
            test_text = f.read()
    except Exception as e:
        _log_llm([], error=f"File read error: {e}")
        raise e

    SCORING_PROMPT = """You are an expert evaluator. Above are two texts: the Ground Truth response and the Predicted response. 
Your task is to compare the Predicted text to the Ground Truth text and score it from 1 to 5 based on how well it matches the meaning, correctness, and comprehensiveness of the Ground Truth.

Criteria:
5 - Identical or fully equivalent in meaning, covers all points correctly.
4 - Minor differences, but semantically very close and accurate.
3 - Captures the main points but has some omissions or slight inaccuracies.
2 - Misses significant parts of the expected answer or has major inaccuracies.
1 - Completely wrong or irrelevant.

Please provide a brief explanation of your reasoning. 
Then, on a new line, write the total score out of 5 exactly in this format:
### Total Score:
x/5
"""

    success = False
    max_retry = 5
    retry = 0
    while not success:
        messages = [
            {"role": "system", "content": "You are a helpful and precise evaluation assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Ground Truth Text:\n" + gt_text},
                {"type": "text", "text": "\n\nPredicted Text:\n" + test_text},
                {"type": "text", "text": "\n\n" + SCORING_PROMPT},
            ]}
        ]
        
        try:
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages,
                temperature=0.3,
            )
            resp_content = response.choices[0].message.content
            _log_llm(messages, response_content=resp_content)

            if "### Total Score:" in resp_content:
                success = True
            else:
                retry += 1
                if retry >= max_retry:
                    _log_llm(messages, error="Max retries reached. Model response did not contain '### Total Score:'")
                    raise Exception("Failed to get response from the model.")
        except Exception as e:
            _log_llm(messages, error=str(e))
            raise e
            
    return float(resp_content.split("### Total Score:")[1].strip().split("/")[0])
