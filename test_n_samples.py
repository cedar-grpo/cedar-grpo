#!/usr/bin/env python3
"""
Test script to verify n_samples functionality works correctly
"""

import sys
import os
sys.path.append('evaluate_sml')

from loader import load_med_qa_dataset, load_med_mcqa_dataset, load_uniadilr_hgc_dataset

def test_loader_n_samples():
    """Test that all loaders respect the n_samples parameter"""
    print("Testing n_samples functionality in dataset loaders...")
    
    # Test MedQA dataset
    try:
        print("\n1. Testing MedQA dataset:")
        full_dataset = load_med_qa_dataset(n_samples=-1)
        print(f"   Full dataset size: {len(full_dataset)}")
        
        limited_dataset = load_med_qa_dataset(n_samples=5)
        print(f"   Limited dataset size (n_samples=5): {len(limited_dataset)}")
        
        assert len(limited_dataset) == 5, f"Expected 5 samples, got {len(limited_dataset)}"
        print("   ✓ MedQA n_samples test passed")
        
    except Exception as e:
        print(f"   ✗ MedQA test failed: {e}")
    
    # Test MedMCQA dataset  
    try:
        print("\n2. Testing MedMCQA dataset:")
        full_dataset = load_med_mcqa_dataset(n_samples=-1)
        print(f"   Full dataset size: {len(full_dataset)}")
        
        limited_dataset = load_med_mcqa_dataset(n_samples=3)
        print(f"   Limited dataset size (n_samples=3): {len(limited_dataset)}")
        
        assert len(limited_dataset) == 3, f"Expected 3 samples, got {len(limited_dataset)}"
        print("   ✓ MedMCQA n_samples test passed")
        
    except Exception as e:
        print(f"   ✗ MedMCQA test failed: {e}")
    
    # Test UniADILR-HGc dataset
    try:
        print("\n3. Testing UniADILR-HGc dataset:")
        full_dataset = load_uniadilr_hgc_dataset(n_samples=-1)
        print(f"   Full dataset size: {len(full_dataset)}")
        
        limited_dataset = load_uniadilr_hgc_dataset(n_samples=2)
        print(f"   Limited dataset size (n_samples=2): {len(limited_dataset)}")
        
        assert len(limited_dataset) == 2, f"Expected 2 samples, got {len(limited_dataset)}"
        print("   ✓ UniADILR-HGc n_samples test passed")
        
    except Exception as e:
        print(f"   ✗ UniADILR-HGc test failed: {e}")
    
    print("\n✓ All n_samples functionality tests completed!")

if __name__ == "__main__":
    test_loader_n_samples()
