"""Generate subtitles.srt from narration section durations."""

import sys
from pathlib import Path
from pydub import AudioSegment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from narration_script import NARRATION_SECTIONS


def format_srt_time(ms):
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms_r = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms_r:03d}"


def split_into_subtitle_segments(text, max_chars=60):
    """Split text into subtitle-sized chunks."""
    words = text.split()
    segments = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            segments.append(current.strip())
            current = word
        else:
            current += " " + word
    if current.strip():
        segments.append(current.strip())
    return segments


def main():
    project_root = Path(__file__).resolve().parent.parent
    dist_dir = project_root / "dist"

    # Get durations of each section
    section_durations = []
    for section in NARRATION_SECTIONS:
        wav_path = dist_dir / f"narration_{section['id']}.wav"
        audio = AudioSegment.from_wav(str(wav_path))
        section_durations.append((section["id"], section["text"], len(audio)))

    # Generate SRT
    srt_lines = []
    sub_idx = 1
    current_time_ms = 0

    for section_id, text, duration_ms in section_durations:
        segments = split_into_subtitle_segments(text, max_chars=60)
        # Calculate time per segment proportionally
        time_per_seg = duration_ms // len(segments) if segments else duration_ms

        for i, seg in enumerate(segments):
            start_ms = current_time_ms + i * time_per_seg
            end_ms = current_time_ms + (i + 1) * time_per_seg
            if i == len(segments) - 1:
                end_ms = current_time_ms + duration_ms

            srt_lines.append(str(sub_idx))
            srt_lines.append(f"{format_srt_time(start_ms)} --> {format_srt_time(end_ms)}")
            srt_lines.append(seg)
            srt_lines.append("")
            sub_idx += 1

        current_time_ms += duration_ms

    srt_path = dist_dir / "subtitles.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    print(f"Saved: {srt_path}")
    print(f"Total subtitle entries: {sub_idx - 1}")


if __name__ == "__main__":
    main()
