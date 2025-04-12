# Meta-Curriculum-LLM

Using meta scheduler in LLM's RL curriculum training process.

## What's this?

- An automated scoring script to label difficulties in dataset
    - For math dataset, we use PASS@N to measure difficulties
    - For logic dataset, we use num of items (e.g. number of people) to measure difficulties
- A automated sampler in [verl](https://github.com/volcengine/verl), we base on [Dynamic Sampler](https://github.com/volcengine/verl/pull/631) with modifications and a meta scheduler implemented by LLM API calls.


## Install

Same as [verl](https://github.com/volcengine/verl)

## Relevant datasets

[kk-difficulty](https://huggingface.co/datasets/siyuan-zhu/kk-difficulty)

[gsm8k-doubao-lite-difficulties](https://huggingface.co/datasets/siyuan-zhu/gsm8k-doubao-lite-difficulties)

[ORZ-MATH-57k-Filter-difficulty-decontaminated](https://huggingface.co/datasets/pe-nlp/ORZ-MATH-57k-Filter-difficulty-decontaminated)

## Citation

[Dynamic Sampler](https://github.com/volcengine/verl/pull/631)

[Logic-RL](https://github.com/Unakar/Logic-RL)

