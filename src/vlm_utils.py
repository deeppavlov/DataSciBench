import os
from openai import OpenAI
import json
from datetime import datetime

try:
    from .vlm_config import API_KEY, BASE_URL
except (ImportError, ValueError):
    from vlm_config import API_KEY, BASE_URL

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

def _log_vlm(request_messages, response_content=None, error=None):
    try:
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_results", "vlm_run_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().isoformat()}] --- VLM Request ---\n")
            # Create a copy without full base64 image data for brevity
            log_messages = []
            for m in request_messages:
                if isinstance(m.get("content"), list):
                    content = []
                    for item in m["content"]:
                        if item.get("type") == "image_url":
                            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,...<truncated>..."}})
                        else:
                            content.append(item)
                    log_messages.append({**m, "content": content})
                else:
                    log_messages.append(m)

            f.write(json.dumps(log_messages, indent=2, ensure_ascii=False) + "\n")
            if response_content:
                f.write(f"--- VLM Response ---\n{response_content}\n")
            if error:
                f.write(f"--- VLM Error ---\n{str(error)}\n")
    except Exception as e:
        print(f"Failed to log VLM call: {e}")

import base64
def vlm_vis_quality(ground_truth_path, test_path):
    # Open the image file and encode it as a base64 string
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    try:
        gt_image = encode_image(ground_truth_path)
        test_image = encode_image(test_path)
    except Exception as e:
        _log_vlm([], error=f"File read error: {e}")
        raise e

    REWADRING_PROMPT = """Above are two figures, which are A and B. The first figure is the ground truth image and the second figure is the predicted image. The total score is 5. Please score B following the criteria below:
    - add 1 point for Data Representation Consistency: Ensure that the underlying data represented by the two charts is identical. This includes the values for all data points and the range of the data. Any variation in the dataset used would make the charts different.
    - add 1 point for Axis Labels and Scales: Verify that both charts have identical axis labels, units, and scales. Any difference in how the axes are labeled or scaled, such as using logarithmic vs. linear scales, can affect the interpretation of the data.
    - add 1 point for Graphical Elements: Check if the visual elements (such as lines, bars, markers, etc.) are represented the same way in both charts. Line thickness, marker styles, and colors should match across charts for them to be considered visually equal.
    - add 1 point for Legend and Annotations: Confirm that any legends, titles, or annotations (e.g., text labels, arrows, or highlights) are the same in both charts. These elements often provide crucial context for interpreting the chart.
    - add 1 point for Chart Dimensions and Layout: Ensure that the dimensions (height and width), aspect ratios, and layout of the charts are identical. Even if the content and representation are similar, a different aspect ratio or spacing between elements can change the chart’s overall appearance and interpretation.

    Please write down the total score for B based on the criteria above, and provide a brief explanation of your reasoning. If you believe that the two figures are not identical, please explain the differences you observed.

    ### Explanation:
    your explanation here

    ### Total Score:
    x/5
    """

    success = False
    max_retry = 5
    retry = 0
    while not success:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"},
            {"role": "user", "content": [
                {"type": "text", "text": "Image A:"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{gt_image}"}
                },
                {"type": "text", "text": "Image B:"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{test_image}"}
                },
                {"type": "text", "text": REWADRING_PROMPT},
            ]}
        ]
        
        try:
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages,
                temperature=1.0,
            )
            resp_content = response.choices[0].message.content
            _log_vlm(messages, response_content=resp_content)

            if "### Total Score:" in resp_content:
                success = True
            else:
                retry += 1
                if retry >= max_retry:
                    _log_vlm(messages, error="Max retries reached. Model response did not contain '### Total Score:'")
                    raise Exception("Failed to get response from the model.")
        except Exception as e:
            _log_vlm(messages, error=str(e))
            raise e
            
    return int(resp_content.split("### Total Score:")[1].strip().split("/")[0])