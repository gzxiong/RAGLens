import os
import json
import pandas as pd
from datasets import load_dataset
from liquid import Template

valid_splits = {
    "RAGTruth-Summary": ["train", "test"],
    "RAGTruth-QA": ["train", "test"],
    "RAGTruth-Data2txt": ["train", "test"],
    "Dolly": ["test"],
    "AggreFact": ["val", "test"],
    "TofuEval": ["dev", "test"],
}

output_template = Template("{{summary}}")

class RAGEvalDataset:
    def __init__(self, dataset = None, split = "test", root_dir = "../"):
        self.dataset = dataset
        self.split = split
        self.root_dir = root_dir
        assert self.split in valid_splits.get(self.dataset, [])
        if not os.path.exists(os.path.join(root_dir, f"data/{self.dataset}/{self.split}.jsonl")):
            download_data(dataset, root_dir)
        if self.dataset in ["RAGTruth-Summary", "RAGTruth-QA", "RAGTruth-Data2txt"]:
            task_type = self.dataset.split('-')[-1]
            with open(os.path.join(root_dir, f"data/RAGTruth/{task_type}-{self.split}.jsonl"), 'r') as f:
                self.items = [json.loads(line) for line in f.readlines()]
        else:
            with open(os.path.join(root_dir, f"data/{self.dataset}/{self.split}.jsonl"), 'r') as f:
                self.items = [json.loads(line) for line in f.readlines()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def download_data(dataset, root_dir):

    if dataset == "Dolly":
        task_type = 'QA'

        sources = pd.read_json(os.path.join(root_dir, 'baselines/ReDEeP-ICLR/dataset', 'source_info_dolly.jsonl'), lines=True)
        sources = sources[sources['task_type'] == task_type][['source_id', 'source_info', 'prompt']]
        source_info = sources.set_index('source_id')['source_info'].to_dict()
        source_prompt = sources.set_index('source_id')['prompt'].to_dict()
        responses = pd.read_json(os.path.join(root_dir, 'baselines/ReDEeP-ICLR/dataset', 'response_dolly.jsonl'), lines=True)
        responses = responses[responses['source_id'].isin(source_prompt.keys())]
        responses['input'] = responses.apply(lambda x: source_prompt[x['source_id']][:12000], axis=1)
        responses['output'] = responses['response'].map(lambda x: output_template.render(summary=x))
        responses['hall_info'] = responses.apply(lambda x: [{'start': 0, 'end': len(x['output']), 'text': x['output'], 'meta': {}}] if len(x['labels'] ) > 0 else [], axis=1)
        responses['context'] = responses.apply(lambda x: source_info[x['source_id']], axis=1)
        responses = responses[['input', 'output', 'hall_info', 'split', 'model', 'context']]
        os.makedirs(os.path.join(root_dir, 'data/Dolly'), exist_ok=True)
        with open(os.path.join(root_dir, f'data/Dolly/test.jsonl'), 'w') as f:
            for item in responses[responses.split == 'test'].to_dict(orient='records'):
                f.write(json.dumps(item) + '\n')
    

    elif dataset in ["RAGTruth-Summary", "RAGTruth-QA", "RAGTruth-Data2txt"]:
        if not os.path.exists(os.path.join(root_dir, 'data/RAGTruth')):
            os.system(f"git clone https://github.com/ParticleMedia/RAGTruth.git {os.path.join(root_dir, 'data/RAGTruth')}")
        task_type = dataset.split('-')[-1]
        sources = pd.read_json(os.path.join(root_dir, 'data/RAGTruth', 'dataset/source_info.jsonl'), lines=True)
        sources = sources[sources['task_type'] == task_type][['source_id', 'source_info', 'prompt']]
        source_info = sources.set_index('source_id')['source_info'].to_dict()
        source_prompt = sources.set_index('source_id')['prompt'].to_dict()
        responses = pd.read_json(os.path.join(root_dir, 'data/RAGTruth', 'dataset/response.jsonl'), lines=True)
        responses = responses[responses['source_id'].isin(source_prompt.keys())]
        responses['input'] = responses.apply(lambda x: source_prompt[x['source_id']], axis=1)
        responses['output'] = responses['response'].map(lambda x: output_template.render(summary=x))
        responses['hall_info'] = responses['labels'].map(lambda x: [{'start': hall['start'], 'end': hall['end'], 'text': hall['text'], 'meta': {'exp': hall['meta'], 'label_type': hall['label_type'], 'implicit_true': hall['implicit_true'], 'due_to_null': hall['due_to_null']}} for hall in x])
        responses['context'] = responses.apply(lambda x: source_info[x['source_id']], axis=1)
        responses = responses[['input', 'output', 'hall_info', 'split', 'model', 'context']]
        with open(os.path.join(root_dir, f'data/RAGTruth/{task_type}-train.jsonl'), 'w') as f:
            for item in responses[responses.split == 'train'].to_dict(orient='records'):
                f.write(json.dumps(item) + '\n')
        with open(os.path.join(root_dir, f'data/RAGTruth/{task_type}-test.jsonl'), 'w') as f:
            for item in responses[responses.split == 'test'].to_dict(orient='records'):
                f.write(json.dumps(item) + '\n')
    
    elif dataset == "AggreFact":
        if not os.path.exists(os.path.join(root_dir, 'data/AggreFact')):
            os.system(f"git clone https://github.com/Liyan06/AggreFact.git {os.path.join(root_dir, 'data/AggreFact')}")
        df = pd.read_csv(os.path.join(root_dir, 'data/AggreFact', 'data', 'aggre_fact_sota.csv'))[['doc', 'summary', 'label', 'cut', 'model_name']]
        df['input'] = df.apply(lambda x: f"Summarize the following news:\n{x['doc']}\n\nOutput:\n", axis=1)
        df['output'] = df['summary']
        df['hall_info'] = df.apply(lambda x: [{'start': 0, 'end': len(x['summary']), 'text': x['summary'], 'meta': {}}] if x['label'] == 0 else [], axis=1)
        df['context'] = df['doc']
        df['model'] = df['model_name']
        items_val = df[df['cut'] == 'val'][['input', 'output', 'hall_info', 'model', 'context']].to_dict(orient='records')
        items_test = df[df['cut'] == 'test'][['input', 'output', 'hall_info', 'model', 'context']].to_dict(orient='records')
        with open(os.path.join(root_dir, 'data/AggreFact/val.jsonl'), 'w') as f:
            for item in items_val:
                f.write(json.dumps(item) + '\n')
        with open(os.path.join(root_dir, 'data/AggreFact/test.jsonl'), 'w') as f:
            for item in items_test:
                f.write(json.dumps(item) + '\n')
        
    elif dataset == "TofuEval":
        if not os.path.exists(os.path.join(root_dir, 'data/TofuEval')):
            os.system(f"git clone https://github.com/amazon-science/tofueval.git {os.path.join(root_dir, 'data/TofuEval')}")
        document_mapping = json.load(open(os.path.join(root_dir, 'data/TofuEval', "document_ids_dev_test_split.json")))
        labels_dev = pd.read_csv(os.path.join(root_dir, "data/TofuEval", "factual_consistency/meetingbank_factual_eval_dev.csv"))
        labels_test = pd.read_csv(os.path.join(root_dir, "data/TofuEval", "factual_consistency/meetingbank_factual_eval_test.csv"))
        meetingbank_dev_ids = document_mapping['dev']['meetingbank']
        meetingbank_test_ids = document_mapping['test']['meetingbank']
        meetingbank = pd.DataFrame(load_dataset("lytang/MeetingBank-transcript", cache_dir=os.path.join(root_dir, "data/MeetingBank-transcript"))['test'])
        meetingbank = meetingbank[meetingbank.meeting_id.isin(meetingbank_dev_ids) | meetingbank.meeting_id.isin(meetingbank_test_ids)][['meeting_id', 'source']].reset_index(drop=True)
        meetingbank['input'] = meetingbank['source'].apply(lambda x: f"Document:\n{x}\n\nSummarize the provided document. The summary should be less than 50 words in length.\n\nSummary:\n")
        meetingbank['context'] = meetingbank['source']

        grouped_dev = labels_dev.groupby(['doc_id', 'annotation_id', 'topic', 'model_name'])
        items_dev = []
        for (doc_id, annotation_id, topic, model_name), group in grouped_dev:
            group_sorted = group.sort_values('sent_idx')
            output = output_template.render(summary='\n'.join(group_sorted['summ_sent'].values))
            hall_info = []
            current_pos = 0
            for _, row in group_sorted.iterrows():
                sent_text = row['summ_sent']
                sent_start = current_pos
                sent_end = current_pos + len(sent_text)
                if row['sent_label'] == 'no':
                    hall_info.append({
                        'start': sent_start,
                        'end': sent_end,
                        'text': sent_text,
                        'meta': {'type': row['type'], 'exp': row['exp']}
                    })
                current_pos = sent_end + 1
            meeting_input = meetingbank[meetingbank['meeting_id'] == doc_id]['input'].iloc[0]
            context = meetingbank[meetingbank['meeting_id'] == doc_id]['context'].iloc[0]
            items_dev.append({
                'input': meeting_input,
                'output': output,
                'hall_info': hall_info,
                'model': model_name,
                'context': context
            })
        with open(os.path.join(root_dir, 'data/TofuEval/dev.jsonl'), 'w') as f:
            for item in items_dev:
                f.write(json.dumps(item) + '\n')

        grouped_test = labels_test.groupby(['doc_id', 'annotation_id', 'topic', 'model_name'])
        items_test = []
        for (doc_id, annotation_id, topic, model_name), group in grouped_test:
            group_sorted = group.sort_values('sent_idx')
            output = output_template.render(summary='\n'.join(group_sorted['summ_sent'].values))
            hall_info = []
            current_pos = 0
            for _, row in group_sorted.iterrows():
                sent_text = row['summ_sent']
                sent_start = current_pos
                sent_end = current_pos + len(sent_text)
                if row['sent_label'] == 'no':
                    hall_info.append({
                        'start': sent_start,
                        'end': sent_end,
                        'text': sent_text,
                        'meta': {'type': row['type'], 'exp': row['exp']}
                    })
                current_pos = sent_end + 1
            meeting_input = meetingbank[meetingbank['meeting_id'] == doc_id]['input'].iloc[0]
            context = meetingbank[meetingbank['meeting_id'] == doc_id]['context'].iloc[0]
            items_test.append({
                'input': meeting_input,
                'output': output,
                'hall_info': hall_info,
                'model': model_name,
                'context': context
            })
        with open(os.path.join(root_dir, 'data/TofuEval/test.jsonl'), 'w') as f:
            for item in items_test:
                f.write(json.dumps(item) + '\n')

    else:
        raise ValueError(f"Unknown dataset: {dataset}")