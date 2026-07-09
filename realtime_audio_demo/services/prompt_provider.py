from realtime_audio_demo.config import SYSTEM_PROMPT_PATH


class FileSystemPromptProvider:
    def __init__(self, path=SYSTEM_PROMPT_PATH) -> None:
        self.path = path

    def get_prompt(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""


system_prompt_provider = FileSystemPromptProvider()
