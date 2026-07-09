from realtime_audio_demo.config import FINAL_MAX_TOKENS
from realtime_audio_demo.services.interfaces import ChatModel, SpeechSynthesizer
from realtime_audio_demo.services.model_gateway import model_gateway


SPEECH_PROMPT = (
    "请把用户输入作为语音播报文本。"
    "只按原文朗读，不要解释、不要改写、不要补充任何内容。"
)


class ModelSpeechSynthesizer(SpeechSynthesizer):
    def __init__(self, model_client: ChatModel) -> None:
        self.model_client = model_client

    async def synthesize(self, *, model: str, text: str) -> str | None:
        speech_text = text.strip()
        if not speech_text:
            return None

        result, status_code = await self.model_client.complete_text(
            model=model,
            text=speech_text,
            prompt=SPEECH_PROMPT,
            history=[],
            max_tokens=FINAL_MAX_TOKENS,
            output_audio=True,
        )
        if status_code >= 400:
            return None
        audio_data_url = result.get("audio_data_url")
        return str(audio_data_url) if audio_data_url else None


speech_synthesizer = ModelSpeechSynthesizer(model_gateway)
