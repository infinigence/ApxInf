"""Record an offline audio deployment observation. Quality approval remains manual."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/apxinf/apxinf/serving"))
from audio_contracts import MAX_OUTPUT, MAX_UPLOAD
from audio_wav import inspect


def exchange(url, data=None, content_type=None):
    request = urllib.request.Request(url, data=data, headers={} if content_type is None else {"Content-Type":content_type})
    with urllib.request.urlopen(request,timeout=135) as response:
        payload = response.read(MAX_OUTPUT + 1)
        if len(payload) > MAX_OUTPUT:
            raise ValueError("response exceeded the complete output limit")
        return payload


def word_error_rate(actual, expected):
    observed, reference = actual.split(), expected.split()
    if not reference or max(len(observed),len(reference)) > 4096:
        raise ValueError("reference must contain 1 through 4096 words")
    previous = list(range(len(observed) + 1))
    for row, wanted in enumerate(reference,1):
        current = [row]
        for column, got in enumerate(observed,1):
            current.append(min(current[-1] + 1,previous[column] + 1,previous[column - 1] + (wanted != got)))
        previous = current
    return previous[-1] / len(reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url",default="http://127.0.0.1:8010")
    parser.add_argument("--task",choices=("asr","tts"),required=True)
    parser.add_argument("--model",required=True)
    parser.add_argument("--expected-revision",required=True)
    parser.add_argument("--input-wav",type=Path)
    parser.add_argument("--reference-text",type=Path)
    parser.add_argument("--text")
    parser.add_argument("--voice")
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    base = args.service_url.rstrip("/")
    readiness = json.loads(exchange(base + "/readyz"))
    if not readiness["ready"] or readiness["provider"] != "mlx_audio" or readiness["task"] != args.task or readiness["model"] != args.model or readiness["model_revision"] != args.expected_revision:
        parser.error("ready deployment does not match the required production identity")
    if args.task == "asr":
        if args.input_wav is None:
            parser.error("ASR requires --input-wav")
        with args.input_wav.open("rb") as source:
            audio = source.read(MAX_UPLOAD + 1)
        info = inspect(audio,True)
        boundary = "apxinf-original-qualification"
        data = (f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{args.model}\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="qualification.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode() + audio + (f"\r\n--{boundary}--\r\n").encode()
        route, content_type = "/v1/audio/transcriptions", "multipart/form-data; boundary=" + boundary
    else:
        if not args.text or not args.voice:
            parser.error("TTS requires --text and --voice")
        data = json.dumps({"model":args.model,"input":args.text,"voice":args.voice,"response_format":"wav"}).encode()
        route, content_type = "/v1/audio/speech", "application/json"
    started = time.monotonic()
    result = exchange(base + route,data,content_type)
    elapsed = time.monotonic() - started
    report = {"readiness":readiness,"task":args.task,"elapsed_seconds":elapsed,
        "timing_boundary":"request send through complete response read", "quality_approved":False,
        "device_allocation_qualified":False,"result_sha256":hashlib.sha256(result).hexdigest()}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    if args.task == "asr":
        transcript = json.loads(result)["text"]
        report.update({"text":transcript,"input_frames":info.frames,"input_sample_rate":info.sample_rate})
        if args.reference_text:
            expected = args.reference_text.read_text(encoding="utf-8")
            report["wer"] = word_error_rate(transcript,expected)
            report["wer_policy"] = "case-sensitive Unicode whitespace word split, no other normalization"
    else:
        info = inspect(result,False)
        report.update({"output_frames":info.frames,"sample_rate":info.sample_rate,
            "duration_seconds":info.frames / info.sample_rate,
            "saturated_samples":sum(sample[0] in (-32768,32767) for sample in struct.iter_unpack("<h",memoryview(result)[info.data_offset:info.data_offset+info.data_bytes]))})
        (args.output_dir / "speech.wav").write_bytes(result)
    (args.output_dir / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2) + "\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
