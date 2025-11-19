from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger('llm')


def call_openai_chat(system: str, user_prompt: str, max_tokens: int = 512) -> str:
    try:
        import openai
    except Exception:
        logger.exception('openai package not available')
        raise

    if not settings.OPENAI_API_KEY:
        logger.error('OPENAI_API_KEY is not set')
        raise RuntimeError('OpenAI API key not configured')

    try:
        openai.api_key = settings.OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception:
        logger.exception('OpenAI API call failed')
        raise


_hf_generator = None


def _get_hf_generator():
    global _hf_generator
    if _hf_generator is None:
        try:
            from transformers import pipeline

            _hf_generator = pipeline('text-generation', model='gpt2', device=-1)
        except Exception:
            logger.exception('Failed to create HF generator')
            raise
    return _hf_generator


def call_llm(system: str, user_prompt: str, max_tokens: int = 512) -> str:
    """Unified LLM caller. Uses configured backend and provides clear errors.

    Raises RuntimeError on configuration issues and propagates API errors.
    """
    if settings.LLM_BACKEND == 'openai':
        return call_openai_chat(system, user_prompt, max_tokens)

    # fallback to HF text-generation pipeline
    try:
        gen = _get_hf_generator()
        out = gen(system + '\n' + user_prompt, max_length=min(200, max_tokens))
        # generated_text often repeats input; return the model output portion
        return out[0].get('generated_text', '')
    except Exception:
        logger.exception('HF fallback failed')
        raise
from ..core.config import settings
from ..core.logger import get_logger
import openai
from transformers import pipeline


logger = get_logger('llm')




def call_openai_chat(system: str, user_prompt: str, max_tokens: int = 512):

        openai.api_key = settings.OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role":"system","content":system}, {"role":"user","content":user_prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp['choices'][0]['message']['content'].strip()


def call_llm(system: str, user_prompt: str, max_tokens: int = 512):
        if settings.LLM_BACKEND == 'openai':
            return call_openai_chat(system, user_prompt, max_tokens)
        else:
            # simple HF fallback

            gen = pipeline('text-generation', model='gpt2', device=-1)
            out = gen(system + '\n' + user_prompt, max_length=200)
            return out[0]['generated_text']