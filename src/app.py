import gc
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
import torch

from libs.transcription import Transcription

load_dotenv()

HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]


def parse_hhmmss(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = "0"
        minutes, seconds = parts
    else:
        raise ValueError(f"Invalid time format: {value}")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def format_seconds(seconds: float) -> str:
    temp_seconds = int(seconds)
    return str(timedelta(seconds=temp_seconds))


def get_media_duration_seconds(media_file_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        media_file_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def build_chunk_prefix(template: str, index: int) -> str:
    if "01" in template:
        return template.replace("01", f"{index:02d}")
    if "{i}" in template:
        return template.format(i=index)
    return f"{template}{index:02d}_"


def split_media_with_overlap(
    media_file_path: str,
    chunk_seconds: int,
    overlap_seconds: int,
    output_dir: str,
) -> list[tuple[str, float]]:
    step = chunk_seconds - overlap_seconds
    if step <= 0:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")

    duration_seconds = get_media_duration_seconds(media_file_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunks: list[tuple[str, float]] = []
    start_seconds = 0.0
    index = 1
    stem = Path(media_file_path).stem

    while start_seconds < duration_seconds:
        current_duration = min(chunk_seconds, duration_seconds - start_seconds)
        chunk_file = output_path / f"{stem}_chunk{index:02d}.wav"
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            media_file_path,
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{current_duration:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(chunk_file),
        ]
        subprocess.run(command, check=True)
        chunks.append((str(chunk_file), start_seconds))
        start_seconds += step
        index += 1

    return chunks


def normalize_text(text: str) -> str:
    return "".join(text.split())


def merge_duplicate_utterances(
    items: list[dict], merge_window_seconds: float
) -> list[dict]:
    merged: list[dict] = []
    for item in items:
        if not merged:
            merged.append(item)
            continue

        previous = merged[-1]
        if (
            normalize_text(item["text"]) == normalize_text(previous["text"])
            and item["start_sec"] <= previous["end_sec"] + merge_window_seconds
        ):
            previous["end_sec"] = max(previous["end_sec"], item["end_sec"])
        else:
            merged.append(item)

    return merged

if __name__ == "__main__":
    prompt: str = "音声ファイル又は動画ファイルの名称を入力してください（拡張子あり）: "
    split_prompt: str = "分割して解析しますか？ (y/n): "
    media_dir_path: str = "./assets/media"
    media_file_name: str = input(prompt)
    media_file_path: str = f"{media_dir_path}/{media_file_name}"
    export_dir_path: str = "./assets/texts"
    export_file_path: str = ""
    prefix: str = ""
    model_size: str = "large-v3-turbo"
    split_mode: bool = input(split_prompt).strip().lower() == "y"

    if not split_mode:
        transcription: Transcription = Transcription(media_file_path)
        is_video: bool = transcription.is_video()

        if is_video:
            transcription.convert_video_to_audio()

        transcription.transcribe_audio(model_size)
        transcription.diarize_audio(HUGGING_FACE_TOKEN)
        transcription.merge_results()

        prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
        export_file_path = f"{export_dir_path}/{prefix}_result"

        transcription.export_results_to_csv(f"{export_file_path}.csv")
        transcription.export_results_to_json(f"{export_file_path}.json")
        transcription.export_results_to_md(f"{export_file_path}.md")

        if is_video:
            os.remove(transcription.media_file_path)

        print("処理が完了しました")
    else:
        chunk_seconds: int = int(
            input("分割秒数（例: 600）: ").strip() or "600"
        )
        overlap_seconds: int = int(
            input("重ねる秒数（例: 30）: ").strip() or "30"
        )
        merge_window_seconds: float = float(
            input("マージ許容秒数（例: 5）: ").strip() or "5"
        )
        prefix_template: str = input("話者プレフィックス（例: CHUNK01_）: ").strip() or "CHUNK01_"

        transcription: Transcription = Transcription(media_file_path)
        is_video: bool = transcription.is_video()

        if is_video:
            transcription.convert_video_to_audio()
            media_file_path = transcription.media_file_path

        chunk_output_dir = f"{media_dir_path}/_chunks/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        chunks = split_media_with_overlap(
            media_file_path, chunk_seconds, overlap_seconds, chunk_output_dir
        )

        merged_items: list[dict] = []

        for index, (chunk_path, start_offset) in enumerate(chunks, start=1):
            chunk_transcription = Transcription(chunk_path)
            chunk_transcription.transcribe_audio(model_size)
            chunk_transcription.diarize_audio(HUGGING_FACE_TOKEN)
            chunk_transcription.merge_results()

            chunk_prefix = build_chunk_prefix(prefix_template, index)
            for item in chunk_transcription.merged_results:
                start_sec = parse_hhmmss(item["start_time"]) + start_offset
                end_sec = parse_hhmmss(item["end_time"]) + start_offset
                merged_items.append(
                    {
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "speaker": f"{chunk_prefix}{item['speaker']}",
                        "text": item["text"],
                    }
                )

            del chunk_transcription
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        merged_items.sort(key=lambda x: x["start_sec"])
        merged_items = merge_duplicate_utterances(merged_items, merge_window_seconds)

        final_results: list[dict] = []
        for item in merged_items:
            final_results.append(
                {
                    "start_time": format_seconds(item["start_sec"]),
                    "end_time": format_seconds(item["end_sec"]),
                    "speaker": item["speaker"],
                    "text": item["text"],
                }
            )

        export_transcription = Transcription(media_file_path)
        export_transcription.merged_results = final_results
        prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
        export_file_path = f"{export_dir_path}/{prefix}_result"

        export_transcription.export_results_to_csv(f"{export_file_path}.csv")
        export_transcription.export_results_to_json(f"{export_file_path}.json")
        export_transcription.export_results_to_md(f"{export_file_path}.md")

        if is_video:
            os.remove(media_file_path)

        print("処理が完了しました")
