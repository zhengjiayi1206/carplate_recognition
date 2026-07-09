from pathlib import Path

from realtime_audio_demo.services.interfaces import AudioTranscoder
from realtime_audio_demo.utils.audio import wav_bytes_from_float32_chunks


class WavAudioTranscoder(AudioTranscoder):
    def float32_chunks_to_wav(self, chunks: list[bytes], source_rate: int, target_rate: int) -> bytes:
        return wav_bytes_from_float32_chunks(chunks, source_rate, target_rate)

    def save_wav(self, wav_bytes: bytes, output_path: Path) -> None:
        output_path.write_bytes(wav_bytes)


audio_transcoder = WavAudioTranscoder()
