# Genie Pytorch Implementation

## Overview
This project is an experimental implementation of the Genie case study world model for the CoinRun environment. It uses a combination of video tokenization, latent action modeling, and dynamics prediction to learn and generate game trajectories.

## Status
**IMPORTANT**: This project is currently untested and has no results yet. It is in early development stage.

## Components
- Video Tokenizer: Encodes game frames into latent representations
- Latent Action Model: Infers actions from sequences of latent states
- Dynamics Model: Predicts future latent states based on current state and action

## Requirements
See requirements.txt for dependencies.

## Usage
1. Generate training data: `python utils/generate_frames.py`
2. Generate validation data: `python utils/generate_val.py`
3. Train the models: `python train.py`
4. Test the models: `python test.py`