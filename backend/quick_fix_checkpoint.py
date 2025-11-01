#!/usr/bin/env python3
"""
Quick script to copy final_ckpt.pt to custom_protein_lm.pt
Run this after training to prepare model for deployment
"""
import os
import shutil
import sys

def find_and_copy_checkpoint():
    """Find final_ckpt.pt and copy to models directory"""
    
    # Common locations to check
    search_paths = [
        os.path.expanduser("~/Downloads/final_ckpt.pt"),
        os.path.expanduser("~/Desktop/final_ckpt.pt"),
        "final_ckpt.pt",
        "../final_ckpt.pt",
        "../../final_ckpt.pt",
    ]
    
    # Also check current directory and subdirectories
    current_dir = os.getcwd()
    for root, dirs, files in os.walk(current_dir):
        if "final_ckpt.pt" in files:
            search_paths.insert(0, os.path.join(root, "final_ckpt.pt"))
    
    source = None
    for path in search_paths:
        full_path = os.path.abspath(path)
        if os.path.exists(full_path):
            source = full_path
            break
    
    if not source:
        print("=" * 60)
        print("ERROR: final_ckpt.pt not found!")
        print("=" * 60)
        print("\nPlease locate your trained checkpoint file.")
        print("Common names: final_ckpt.pt, ckpt_step*.pt, ckpt_epoch*.pt")
        print("\nOnce found, run:")
        print("  python quick_fix_checkpoint.py <path_to_final_ckpt.pt>")
        print("\nOr manually copy:")
        print("  copy <your_path>\\final_ckpt.pt backend\\models\\custom_protein_lm.pt")
        return False
    
    # Destination
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    destination = os.path.join(models_dir, "custom_protein_lm.pt")
    
    try:
        shutil.copy(source, destination)
        print("=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Source: {source}")
        print(f"Destination: {destination}")
        print("\n✅ Copied successfully!")
        print("\nNow restart your Flask server to load the trained model.")
        return True
    except Exception as e:
        print(f"\n❌ Error copying file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # User provided path
        source = sys.argv[1]
        if os.path.exists(source):
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(models_dir, exist_ok=True)
            destination = os.path.join(models_dir, "custom_protein_lm.pt")
            try:
                shutil.copy(source, destination)
                print(f"✅ Copied {source} to {destination}")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print(f"❌ File not found: {source}")
    else:
        # Auto-search
        find_and_copy_checkpoint()

