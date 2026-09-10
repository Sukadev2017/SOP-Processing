def test_models():

    from sopgenai.models import (
        CanonicalDocument,
        KnowledgeUnit,
        Mapping,
    )

    assert CanonicalDocument
    assert KnowledgeUnit
    assert Mapping


def test_llm_factory():

    from sopgenai.llm import (
        LLMFactory,
        HuggingFaceProvider,
        OpenAIProvider,
        AnthropicProvider,
        OllamaProvider,
    )

    assert LLMFactory

    assert HuggingFaceProvider

    assert OpenAIProvider

    assert AnthropicProvider

    assert OllamaProvider


def test_pipeline():

    from sopgenai.pipeline import (
        SOPPipeline,
    )

    assert SOPPipeline
