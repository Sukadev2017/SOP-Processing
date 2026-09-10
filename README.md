# SOP-Processing
# SOP GenAI Migration Platform

Provider-independent GenAI application for controlled SOP migration.

## Supported LLM Providers

The application supports:

- Hugging Face
- OpenAI
- Anthropic
- Ollama

Provider selection is controlled through:

`config/app_config.yaml`

No SOP pipeline code needs to change when switching providers.


## Ollama

Ollama can be used to run the LLM locally or against a
self-hosted Ollama server.

Install Ollama separately.

Start Ollama:

```bash
ollama serve
