from model import QWenForCausalLM

MODEL_MAP = {
    'QWenLMHeadModel': QWenForCausalLM,
    'QWenForCausalLM': QWenForCausalLM,
    'Qwen2ForCausalLM': QWenForCausalLM,
    'Qwen2MoeForCausalLM': QWenForCausalLM,
    'Qwen2VLForConditionalGeneration': QWenForCausalLM,
}