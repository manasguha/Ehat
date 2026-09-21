"""
Narration audio generator using Kokoro-82M TTS.
Generates per-section WAV files, concatenates with gaps, normalizes loudness.
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from narration_script import NARRATION_SECTIONS


def generate_section_audio(pipeline, text, out_path):
    """Generate a single WAV file from text using Kokoro TTS."""
    segments = []
    for _, _, audio in pipeline(text, voice="af_heart", speed=1.0):
        segments.append(audio)
    audio = np.concatenate(segments)
    sf.write(str(out_path), audio, 24000)
    return out_path


def concatenate_with_gaps(section_files, gap_ms=400, sample_rate=24000):
    """Concatenate WAV files with silence gaps between sections."""
    silence = AudioSegment.silent(duration=gap_ms, frame_rate=sample_rate)
    combined = AudioSegment.empty()
    for i, f in enumerate(section_files):
        seg = AudioSegment.from_wav(str(f))
        combined += seg
        if i < len(section_files) - 1:
            combined += silence
    return combined


def normalize_loudness(audio_segment, target_dbfs=-20):
    """Normalize audio loudness."""
    change_in_dbfs = target_dbfs - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_dbfs)


def main():
    from kokoro import KPipeline

    project_root = Path(__file__).resolve().parent.parent
    dist_dir = project_root / "dist"
    dist_dir.mkdir(exist_ok=True)

    print("Loading Kokoro TTS pipeline...")
    pipeline = KPipeline(lang_code="a")

    section_files = []
    for section in NARRATION_SECTIONS:
        out_path = dist_dir / f"narration_{section['id']}.wav"
        print(f"  Generating: {section['id']}...")
        generate_section_audio(pipeline, section["text"], out_path)
        section_files.append(out_path)

    print("Concatenating with gaps...")
    combined = concatenate_with_gaps(section_files, gap_ms=400)

    print("Normalizing loudness...")
    combined = normalize_loudness(combined, target_dbfs=-20)

    final_path = dist_dir / "narration.wav"
    combined.export(str(final_path), format="wav")
    print(f"Saved: {final_path}")

    duration_s = len(combined) / 1000
    print(f"Total duration: {duration_s:.1f} seconds")


if __name__ == "__main__":
    main()
