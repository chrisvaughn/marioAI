#!/usr/bin/env bash

ls -1tp train/ | grep -E "best_model_[0-9]+" | tail -n +6 | xargs -I {} rm -- train/{}