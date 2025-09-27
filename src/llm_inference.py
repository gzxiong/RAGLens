import os
import sys
sys.path.append("../src")
import os
import torch
import numpy as np
from data_loading import RAGEvalDataset
from vllm import LLM, SamplingParams
import json
from liquid import Template

import argparse


valid_splits = {
    "RAGTruth-Summary": ["test"],
    "RAGTruth-QA": ["test"],
    "RAGTruth-Data2txt": ["test"],
    "Dolly": ["test"],
    "AggreFact": ["test"],
    "TofuEval": ["test"],
}

input_template = Template("Decide if the following summary/answer is consistent with the corresponding article. Note that consistency means all information in the output is supported by the article.\n\nArticle: {{article}}\n\nSummary/Answer: ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for llm inference with vllm.')
    parser.add_argument('--llm_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    llm_name = args.llm_name
    dataset = args.dataset

    root_dir = "../"

    sampling_params = SamplingParams(temperature=0.0, seed=42, max_tokens=1024)
    llm = LLM(model=llm_name, seed=42, gpu_memory_utilization=0.95, dtype="auto", download_dir=os.path.join(root_dir, "../huggingface/hub"), tensor_parallel_size=torch.cuda.device_count(), max_model_len=None if "Llama-2" in llm_name else 32768, max_num_seqs=4 if "70b" in llm_name.lower() else 256)

    data_dict = {}
    for split in valid_splits[dataset]:
        data_dict[split] = RAGEvalDataset(dataset, split, root_dir).items
        os.makedirs(os.path.join(root_dir, f"predictions/{dataset}/{llm_name.replace('/', '_')}"), exist_ok=True)
    for split, items in data_dict.items():
        conversations = [
            [
                {
                    "role": "user",
                    "content": input_template.render(article=item['context']) + item['output'] + "\n\nExplain your reasoning step by step then answer (yes or no) the question:"
                }
            ] for item in items
        ]
        outputs = llm.chat(conversations, sampling_params=sampling_params)
        results = []
        for idx, output in enumerate(outputs):
            output_text = output.outputs[0].text
            pred_label = None
            if "yes" in output_text.lower():
                if "no" in output_text.lower():
                    if output_text.lower()[::-1].index("sey") < output_text.lower()[::-1].index("on"):
                        pred_label = "consistent"
                    else:
                        pred_label = "inconsistent"
                else:
                    pred_label = "consistent"
            elif "no" in output_text.lower():
                pred_label = "inconsistent"
            results.append({
                "true_label": "inconsistent" if len(items[idx]['hall_info']) > 0 else "consistent",
                "pred_label": pred_label,
                "model_output": output_text
            })
        json.dump(results, open(os.path.join(root_dir, f"predictions/{dataset}/{llm_name.replace('/', '_')}", f"{split}_results.json"), "w"), indent=4)