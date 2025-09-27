import sys
sys.path.append("../src")
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae
from sae_encoding import encode_outputs
from data_loading import RAGEvalDataset
from huggingface_hub import hf_hub_download
import argparse

root_dir = "../"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

valid_splits = {
    "Dolly": ["test"],
    "RAGTruth-Summary": ["train", "test"],
    "RAGTruth-QA": ["train", "test"],
    "RAGTruth-Data2txt": ["train", "test"],
    "AggreFact": ["test", "val"],
    "TofuEval": ["test", "dev"],
}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Script for llm inference with vllm.')
    parser.add_argument('--llm_name', type=str, required=True)
    parser.add_argument('--sae_name', type=str, required=True)
    parser.add_argument('--hookpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--agg', type=str, default="max")
    parser.add_argument('--force', action='store_true', help='Force re-encoding even if files already exist')
    args = parser.parse_args()

    llm_name = args.llm_name
    sae_name = args.sae_name
    dataset = args.dataset
    hookpoint = args.hookpoint
    agg = args.agg
    force = args.force

    if "70B" in llm_name:
        tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir=os.path.join(root_dir, "../huggingface/hub"))
        model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16, cache_dir=os.path.join(root_dir, "../huggingface/hub"), device_map="auto")
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir=os.path.join(root_dir, "../huggingface/hub"))
        model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16, cache_dir=os.path.join(root_dir, "../huggingface/hub"))
        model = model.to(device)

    model.eval()

    if "Goodfire" not in sae_name:
        if os.path.exists(os.path.join(sae_name, hookpoint)):
            sae = Sae.load_from_disk(os.path.join(sae_name, hookpoint))
        else:
            sae = Sae.load_from_hub(sae_name, hookpoint=hookpoint)
        sae.cfg.transcode = True if "transcoder" in sae_name else False
        sae = sae.to(device)
        sae.eval()
    else:
        class SparseAutoEncoder(torch.nn.Module):
            def __init__(
                self,
                d_in: int,
                d_hidden: int,
                device: torch.device,
                dtype: torch.dtype = torch.bfloat16,
            ):
                super().__init__()
                self.d_in = d_in
                self.d_hidden = d_hidden
                self.device = device
                self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
                self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
                self.dtype = dtype
                self.to(self.device, self.dtype)
                self.cfg = type("Cfg", (), {})()
                self.cfg.transcode = False

            def encode(self, x: torch.Tensor) -> torch.Tensor:
                """Encode a batch of data using a linear, followed by a ReLU."""
                return torch.nn.functional.relu(self.encoder_linear(x))

            def decode(self, x: torch.Tensor) -> torch.Tensor:
                """Decode a batch of data using a linear."""
                return self.decoder_linear(x)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                """SAE forward pass. Returns the reconstruction and the encoded features."""
                f = self.encode(x)
                return self.decode(f), f
        
        checkpoint_path = os.path.join(root_dir, "checkpoints", sae_name + (".pt" if "70B" in sae_name else ".pth"))
        checkpoint_dir = checkpoint_path.rsplit("/", 1)[0]
        os.makedirs(checkpoint_dir, exist_ok=True)

        if not os.path.exists(checkpoint_path):
            print(f"Downloading checkpoint for {sae_name}...")
            hf_hub_download(
                repo_id=sae_name,
                filename=checkpoint_path.rsplit("/", 1)[1],
                local_dir=checkpoint_dir,
            )
        sae_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
        sae = SparseAutoEncoder(
                sae_dict['encoder_linear.weight'].shape[1],
                sae_dict['encoder_linear.weight'].shape[0],
            device,
        )
        sae.load_state_dict(sae_dict)
    
    data_dict = {}
    for split in valid_splits[dataset]:
        data_dict[split] = RAGEvalDataset(dataset, split, root_dir).items


    for split, items in data_dict.items():

        encoded_dir = os.path.join(root_dir, f"data/{dataset}/encoded/{sae_name.replace('/', '_')}", f"{hookpoint}")
        os.makedirs(encoded_dir, exist_ok=True)

        print(f"Encoding {split} features for {dataset} with {agg} aggregation...")
        if not force and (os.path.exists(os.path.join(encoded_dir, f"{split}-{agg}.npy")) or os.path.exists(os.path.join(encoded_dir, f"{split}-{agg}-output_lengths.npy"))):
            pass
        else:
            split_features = encode_outputs([it['input'] for it in items], [it['output'] for it in items], hookpoint, tokenizer, model, sae, agg=agg, show_progress=True).numpy()
            np.save(os.path.join(encoded_dir, f"{split}-{agg}.npy"), split_features)