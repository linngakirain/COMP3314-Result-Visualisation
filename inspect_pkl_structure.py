#!/usr/bin/env python3
"""Inspect the structure of the vgg16 pkl file to extract change points."""

import pickle
import numpy as np

with open('real_data/nyc_taxi/vgg16_sbs_nyc_taxi.pkl', 'rb') as f:
    data = pickle.load(f)

print("Type:", type(data))
print("Length:", len(data) if hasattr(data, '__len__') else 'N/A')

if isinstance(data, tuple):
    print(f"\nTuple has {len(data)} elements")
    for i, item in enumerate(data):
        print(f"\nElement {i}:")
        print(f"  Type: {type(item)}")
        if isinstance(item, dict):
            print(f"  Keys: {list(item.keys())}")
            for key, value in item.items():
                print(f"    {key}: {type(value)}")
                if key == 'interval':
                    print(f"      Value: {value}")
                elif key == 'output':
                    if isinstance(value, list):
                        print(f"      List length: {len(value)}")
                        if len(value) > 0:
                            print(f"      First item keys: {list(value[0].keys()) if isinstance(value[0], dict) else 'N/A'}")
                            # Look for change points
                            for idx, output_item in enumerate(value):
                                if isinstance(output_item, dict) and 'output' in output_item:
                                    output_data = output_item['output']
                                    if 'ch_pt' in output_data:
                                        print(f"      Change point {idx}: {output_data['ch_pt']}")
                                        if 'interval' in output_item:
                                            print(f"        Interval: {output_item['interval']}")

