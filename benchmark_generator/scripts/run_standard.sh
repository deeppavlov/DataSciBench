#!/bin/bash
# Env vars для стандартного пайплайна. Ключ OPENAI_API_KEY берётся из .env в корне проекта.

export BENCH_GEN_LLM_MODEL="gemini-3-flash-preview"
export BENCH_GEN_LLM_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"

python -m benchmark_generator.pipelines.pipeline_standard "$@"
