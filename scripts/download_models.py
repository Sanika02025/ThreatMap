"""
Download and verify models for RAG-ATT&CK reproduction.
"""

import os
import sys

def download_all():
    print("=== Downloading models ===")
    from sentence_transformers import SentenceTransformer, CrossEncoder
    
    print("\n[1/3] Downloading cross-encoder/ms-marco-MiniLM-L-6-v2 (~90MB)...")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("ms-marco-MiniLM-L-6-v2 downloaded successfully!")
    
    print("\n[2/3] Downloading basel/ATTACK-BERT (~440MB)...")
    ab = SentenceTransformer("basel/ATTACK-BERT")
    print("basel/ATTACK-BERT downloaded successfully! Dim:", ab.get_sentence_embedding_dimension())
    
    print("\n[3/3] Downloading nomic-ai/nomic-embed-text-v1.5 (~547MB)...")
    nomic = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    print("nomic-embed-text-v1.5 downloaded successfully! Dim:", nomic.get_sentence_embedding_dimension())
    
    print("\n=== All models successfully downloaded and verified! ===")

if __name__ == "__main__":
    download_all()
