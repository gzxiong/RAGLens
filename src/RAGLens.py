import sys
sys.path.append("../src")
import os
import torch
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sae_encoding import encode_outputs, sae_encoding
from utils import compute_mutual_information_chunked

class RAGLens:

    def __init__(
        self, 
        tokenizer, 
        model, 
        sae, 
        hookpoint, 
        top_k = 1000,
        max_bins = 32, 
        random_state = 0,
        early_stopping_tolerance = 1e-5,
        validation_size = 0.1,
        max_rounds = 1000,
    ):
        
        self.tokenizer = tokenizer
        self.model = model
        self.sae = sae
        self.hookpoint = hookpoint
        self.clf = ExplainableBoostingClassifier(
            interactions = 0, 
            max_bins = max_bins, 
            random_state = random_state, 
            early_stopping_tolerance = early_stopping_tolerance, 
            validation_size = validation_size, 
            max_rounds = max_rounds
        )
        self.top_k = top_k
        self.top_k_indices = None

    def fit(
        self,
        inputs,
        outputs,
        labels,
        save_path=None,
        n_bins = 50,
        chunk_size = 2000
    ):

        assert type(inputs) == type(outputs) == type(labels) == list
        assert len(inputs) == len(outputs) == len(labels)
        
        if save_path and os.path.exists(save_path):
            features = np.load(save_path)
        else:
            print("Encoding training features with SAE...")
            features = encode_outputs(
                inputs, 
                outputs, 
                self.hookpoint, 
                self.tokenizer, 
                self.model, 
                self.sae, 
                agg='max', 
                show_progress=True
            ).numpy()
            if save_path:
                np.save(save_path, features)

        # Mutual Information-based Feature Selection
        print(f"Extracting top {self.top_k} key SAE features...")
        mi_scores = compute_mutual_information_chunked(
            torch.tensor(features), 
            torch.tensor(labels), 
            n_bins = n_bins,
            chunk_size = chunk_size
        ).cpu().numpy()

        sorted_indices = np.argsort(mi_scores)[::-1]
        self.top_k_indices = sorted_indices[:self.top_k]

        # Generalized Additive Model Fitting
        print(f"Fitting an additive model based on key SAE features...")
        self.clf.fit(features[:, self.top_k_indices], labels)

    def predict_proba(self, inputs, outputs):
        
        if type(inputs) == type(outputs) == str:
            inputs = [inputs]
            outputs = [outputs]
        
        assert type(inputs) == type(outputs) == list

        features = encode_outputs(
            inputs, 
            outputs, 
            self.hookpoint, 
            self.tokenizer, 
            self.model, 
            self.sae, 
            agg='max', 
            show_progress=False
        ).numpy()[:, self.top_k_indices]
        logits = self.clf.predict_proba(features)[:, 1]
        return logits
        
    def predict(self, inputs, outputs):

        logits = self.predict_proba(inputs, outputs)
        preds = (logits > 0.5).astype(int)

        return preds