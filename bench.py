import contextlib
import inspect
import io
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
import types
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from pydub import AudioSegment

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
DATA_DIR = Path(os.getenv("BENCH_DATA_DIR", "./data"))
AUDIO_DIR = DATA_DIR / "audio"
MUSIC_DIR = DATA_DIR / "music"
CONTEXT_DIR = DATA_DIR / "context"
DB_PATH = DATA_DIR / "bench.db"

for folder in (DATA_DIR, AUDIO_DIR, MUSIC_DIR, CONTEXT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Stimgen 3")

# context docs: per doc caps by size class, one shared total cap
CONTEXT_CHAR_CAP = 80000
DOC_CLASS_CAPS = {"small": 10000, "medium": 20000, "large": 40000}
CONTEXT_GROUPS = ("substance", "form", "technique")

TAGS = ["Reject", "Good", "Best so far"]

# one time translation of the v2 tag set, runs at every start, harmless when done
OLD_TAG_MAP = {
    "rejected": "Reject",
    "best so far": "Best so far",
    "chills": "Good",
    "close, needs work": "Good",
    "day 1 pick": "Good",
    "day 2 pick": "Good",
    "day 3 pick": "Good",
    "day 4 pick": "Good",
    "day 5 pick": "Good",
}

MODELS = ["claude-opus-4-8", "claude-fable-5", "claude-sonnet-4-6"]
AUTOGEN_MODEL = "claude-haiku-4-5-20251001"

SEED_VOICES = [
    ("Christian", "lMILJ9d29MrRXy9BIgcz"),
    ("Alicia", "OOk3INdXVLRmSaQoAX9D"),
]


# database

def db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def add_column(connection, table, column, definition):
    try:
        connection.execute(f"alter table {table} add column {column} {definition}")
    except Exception:
        pass


def init_db():
    connection = db()
    connection.execute("""
        create table if not exists experiments (
            id integer primary key autoincrement,
            created_at text not null,
            topic text default '',
            prompt_py text default '',
            mix_py text default '',
            model text default '',
            voice_id text default '',
            stability real default 0.5,
            style real default 0,
            boost integer default 1,
            music_filename text default '',
            music_file text default '',
            speech_text text default '',
            voice_file text default '',
            mix_file text default '',
            mix_source text default '',
            tts_provider text default 'elevenlabs',
            verdict text default '',
            comment text default '',
            parent_id integer
        )
    """)
    connection.execute("""
        create table if not exists saved_files (
            id integer primary key autoincrement,
            created_at text not null,
            kind text not null,
            name text not null,
            content text not null
        )
    """)
    connection.execute("""
        create table if not exists context_files (
            id integer primary key autoincrement,
            created_at text not null,
            name text not null,
            kind text not null,
            stored_name text not null,
            extracted text default '',
            chars integer default 0,
            note text default ''
        )
    """)
    connection.execute("""
        create table if not exists voices (
            id integer primary key autoincrement,
            created_at text not null,
            name text not null,
            voice_id text not null
        )
    """)
    connection.execute("""
        create table if not exists compose_state (
            id integer primary key check (id = 1),
            state text not null,
            updated_at text not null
        )
    """)

    # columns added after the first version, safe to run every start
    add_column(connection, "experiments", "mix_source", "text default ''")
    add_column(connection, "experiments", "tts_provider", "text default 'elevenlabs'")
    add_column(connection, "experiments", "title", "text default ''")
    add_column(connection, "experiments", "tag", "text default ''")
    add_column(connection, "experiments", "reflection", "text default ''")
    add_column(connection, "experiments", "protocol_id", "integer")
    add_column(connection, "experiments", "day_number", "integer default 1")
    add_column(connection, "experiments", "prior_id", "integer")
    add_column(connection, "experiments", "context_ids", "text default ''")
    add_column(connection, "experiments", "run_log", "text default ''")
    add_column(connection, "experiments", "validation", "text default ''")
    add_column(connection, "experiments", "word_count", "integer default 0")
    add_column(connection, "experiments", "music_gain_db", "real")
    add_column(connection, "experiments", "voice_lufs", "real")
    add_column(connection, "experiments", "sync_mode", "text default ''")
    add_column(connection, "experiments", "mix_profile", "text default ''")
    add_column(connection, "experiments", "prompt_source", "text default ''")

    # stimgen 3 columns
    add_column(connection, "experiments", "questions", "text default '[]'")
    add_column(connection, "experiments", "doc_html", "text default ''")
    add_column(connection, "experiments", "balance_db", "real")
    add_column(connection, "experiments", "fade_in_s", "real")
    add_column(connection, "experiments", "fade_out_s", "real")
    add_column(connection, "experiments", "pause_ms", "integer")
    add_column(connection, "experiments", "long_pause_ms", "integer")
    add_column(connection, "context_files", "size_class", "text default ''")

    connection.execute("update experiments set protocol_id = id where protocol_id is null")
    connection.execute("update experiments set day_number = 1 where day_number is null")

    # v2 to v3 migrations
    for old_tag, new_tag in OLD_TAG_MAP.items():
        connection.execute("update experiments set tag = ? where tag = ?", (new_tag, old_tag))
    connection.execute("update context_files set kind = 'substance' where kind = 'reference'")
    connection.execute("update context_files set kind = 'form' where kind = 'example'")
    rows = connection.execute(
        "select id, chars from context_files where size_class = '' or size_class is null"
    ).fetchall()
    for row in rows:
        connection.execute(
            "update context_files set size_class = ? where id = ?",
            (size_class_for(row["chars"] or 0), row["id"]),
        )

    seeded = connection.execute("select count(*) as n from voices").fetchone()["n"]
    if seeded == 0:
        for name, voice_id in SEED_VOICES:
            connection.execute(
                "insert into voices (created_at, name, voice_id) values (?, ?, ?)",
                (now(), name, voice_id),
            )
        print(f"seeded {len(SEED_VOICES)} voices")

    connection.commit()
    connection.close()


def now():
    return datetime.now(timezone.utc).isoformat()


def size_class_for(chars):
    if chars <= DOC_CLASS_CAPS["small"]:
        return "small"
    if chars <= DOC_CLASS_CAPS["medium"]:
        return "medium"
    return "large"


# audio helpers, ported from the main repo so pasted mix.py files work standalone

def load_audio(path):
    return AudioSegment.from_file(path)


def normalize_dbfs(segment, target_dbfs):
    return segment.apply_gain(target_dbfs - segment.dBFS)


def make_stereo(segment):
    return segment.set_channels(2)


def duration_ms(segment):
    return len(segment)


def measure_lufs(segment):
    try:
        import pyloudnorm
        raw = np.frombuffer(segment.raw_data, dtype=np.int16).astype(np.float64) / 32768.0
        data = raw.reshape(-1, segment.channels)
        meter = pyloudnorm.Meter(segment.frame_rate)
        value = meter.integrated_loudness(data)
        if not np.isfinite(value) or value < -70.0:
            return None
        return float(value)
    except Exception:
        return None


def normalize_lufs(segment, target_lufs):
    measured = measure_lufs(segment)
    if measured is None:
        return segment
    return segment.apply_gain(target_lufs - measured)


def content_duration_sec(music_path):
    try:
        segment = load_audio(music_path)
        total_ms = len(segment)
        if total_ms <= 0:
            return None
        window_ms = 500
        position = total_ms
        while position > 0:
            chunk = segment[max(0, position - window_ms):position]
            if chunk.rms > 0:
                full_scale = float(1 << (8 * chunk.sample_width - 1))
                rms_dbfs = 20.0 * math.log10(chunk.rms / full_scale)
            else:
                rms_dbfs = -120.0
            if rms_dbfs > -45.0:
                silent_tail = total_ms - position
                if silent_tail < 2000:
                    return total_ms / 1000.0
                print(f"music content ends at {position}ms, stripped {silent_tail}ms of tail")
                return position / 1000.0
            position -= window_ms
        return total_ms / 1000.0
    except Exception as error:
        print(f"content duration measurement failed: {error}")
        return None


def register_repo_modules():
    # pasted repo mix.py does "from ..utils.audio import ...", this makes that resolve
    app_module = types.ModuleType("app")
    app_module.__path__ = []
    utils_module = types.ModuleType("app.utils")
    utils_module.__path__ = []
    audio_module = types.ModuleType("app.utils.audio")
    audio_module.load_audio = load_audio
    audio_module.normalize_dbfs = normalize_dbfs
    audio_module.make_stereo = make_stereo
    audio_module.duration_ms = duration_ms
    app_module.utils = utils_module
    utils_module.audio = audio_module
    sys.modules.setdefault("app", app_module)
    sys.modules["app.utils"] = utils_module
    sys.modules["app.utils.audio"] = audio_module


register_repo_modules()


# exec loaders for pasted files

def exec_pasted(code, module_name):
    namespace = {
        "__name__": module_name,
        "__package__": "app.services",
        "__builtins__": __builtins__,
    }
    exec(compile(code, module_name + ".py", "exec"), namespace)
    return namespace


PROMPT_NAMES = (
    "SYSTEM_PROMPT",
    "PROMPT",
    "MEDITATION_SYSTEM",
    "SPEECH1_SYSTEM",
    "JOURNAL_SYSTEM",
    "PLAN_SYSTEM",
)


def load_prompt_file(code, topic):
    try:
        namespace = exec_pasted(code, "pasted_prompt")
    except Exception as error:
        raise HTTPException(400, f"prompt file failed to run: {error}")

    system_prompt = None
    source_name = ""
    for name in PROMPT_NAMES:
        value = namespace.get(name)
        if isinstance(value, str) and value.strip():
            system_prompt = value
            source_name = name
            break
    if system_prompt is None:
        candidates = [(k, v) for k, v in namespace.items()
                      if isinstance(v, str) and len(v) > 200 and not k.startswith("__")]
        if candidates:
            source_name, system_prompt = max(candidates, key=lambda pair: len(pair[1]))
            source_name = source_name + " (largest string, no known prompt name found)"
    if system_prompt is None:
        raise HTTPException(400, "no prompt string found in the pasted file, expected one of "
                                 + ", ".join(PROMPT_NAMES))

    builder = namespace.get("build_user_prompt")
    if callable(builder):
        try:
            user_prompt = call_builder(builder, topic)
            source_name += " plus build_user_prompt()"
        except Exception as error:
            raise HTTPException(400, f"build_user_prompt failed: {error}")
    else:
        user_prompt = topic if topic else "Write the piece now."

    validate = namespace.get("validate")
    if not callable(validate):
        validate = None

    return system_prompt, user_prompt, source_name, validate


def call_builder(builder, topic):
    signature = inspect.signature(builder)
    arguments = {}
    first = True
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if first:
            arguments[name] = topic
            first = False
            continue
        if parameter.default is not parameter.empty:
            continue
        lowered = name.lower()
        if "word" in lowered:
            arguments[name] = 300
        elif "minute" in lowered:
            arguments[name] = 2
        elif lowered in ("challenges", "tips", "completed_actions"):
            arguments[name] = []
        else:
            arguments[name] = ""
    return builder(**arguments)


def run_validate(validate, script):
    if validate is None:
        return ""
    try:
        result = validate(script)
    except Exception as error:
        return f"validate() raised: {error}"
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    try:
        items = [str(item) for item in result]
    except Exception:
        return str(result)
    return "\n".join(items)


def load_mix_function(code):
    if not code.strip():
        return default_mix
    try:
        namespace = exec_pasted(code, "pasted_mix")
    except Exception as error:
        raise HTTPException(400, f"mix file failed to run: {error}")
    mix_function = namespace.get("mix")
    if not callable(mix_function):
        raise HTTPException(400, "no mix() function found in the pasted file")
    return mix_function


def default_mix(voice_path, music_path, out_path,
                music_premix_gain_db=-14.0, fade_in_s=None, fade_out_s=None, **_):
    # plain fallback used when the mix box is empty, plays the full music track
    voice = make_stereo(load_audio(voice_path).set_frame_rate(44100))
    if music_path and Path(music_path).exists():
        music = make_stereo(load_audio(music_path).set_frame_rate(44100))
        music = music.apply_gain(float(music_premix_gain_db))
        if len(music) < len(voice):
            loops = len(voice) // len(music) + 1
            music = (music * loops)[:len(voice) + 3000]
        mixed = music.overlay(voice)
    else:
        mixed = voice
    if fade_in_s:
        mixed = mixed.fade_in(int(float(fade_in_s) * 1000))
    if fade_out_s:
        mixed = mixed.fade_out(int(float(fade_out_s) * 1000))
    elif music_path and Path(music_path).exists():
        mixed = mixed.fade_out(2000)
    mixed.export(out_path, format="mp3", bitrate="256k")
    return len(mixed)


DEFAULT_MIX_SOURCE = """# bench default mix, used because no mix file was pasted, plays the full music track
from pydub import AudioSegment
from pathlib import Path

def mix(voice_path, music_path, out_path,
        music_premix_gain_db=-14.0, fade_in_s=None, fade_out_s=None, **_):
    voice = AudioSegment.from_file(voice_path).set_frame_rate(44100).set_channels(2)
    if music_path and Path(music_path).exists():
        music = AudioSegment.from_file(music_path).set_frame_rate(44100).set_channels(2)
        music = music.apply_gain(float(music_premix_gain_db))
        if len(music) < len(voice):
            loops = len(voice) // len(music) + 1
            music = (music * loops)[:len(voice) + 3000]
        mixed = music.overlay(voice)
    else:
        mixed = voice
    if fade_in_s:
        mixed = mixed.fade_in(int(float(fade_in_s) * 1000))
    if fade_out_s:
        mixed = mixed.fade_out(int(float(fade_out_s) * 1000))
    elif music_path and Path(music_path).exists():
        mixed = mixed.fade_out(2000)
    mixed.export(out_path, format="mp3", bitrate="256k")
"""


def mix_source_for(mix_py, music_path):
    if mix_py.strip():
        return "pasted mix.py"
    if music_path:
        return "bench default"
    return "voice only"


def explicit_params(function):
    # names the function actually declares, ignoring **kwargs
    try:
        signature = inspect.signature(function)
    except Exception:
        return set()
    return {
        name for name, parameter in signature.parameters.items()
        if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    }


def pre_gain_file(source_path, gain_db, scratch_dir, label):
    # used when the mix function has no gain argument of its own
    segment = load_audio(source_path)
    segment = segment.apply_gain(float(gain_db))
    target = Path(scratch_dir) / f"{label}_gained.wav"
    segment.export(target, format="wav")
    return str(target)


def run_mix(mix_function, voice_path, music_path, out_path, settings, log):
    # settings holds balance_db, fade_in_s, fade_out_s, any may be None
    content_sec = None
    if music_path and Path(str(music_path)).exists():
        content_sec = content_duration_sec(music_path)
        log.append(f"music content duration {content_sec if content_sec is not None else 'not measured'}")

    accepted = explicit_params(mix_function)
    log.append("mix accepts: " + (", ".join(sorted(accepted)) if accepted else "nothing declared"))

    kwargs = {}
    dropped = []

    if content_sec is not None:
        if "content_duration_sec" in accepted:
            kwargs["content_duration_sec"] = content_sec
        else:
            dropped.append("content_duration_sec")

    balance = settings.get("balance_db")
    fade_in = settings.get("fade_in_s")
    fade_out = settings.get("fade_out_s")

    if balance is not None:
        if "music_premix_gain_db" in accepted:
            kwargs["music_premix_gain_db"] = balance
            log.append(f"balance {balance} dB passed as music_premix_gain_db")
        else:
            dropped.append("music_premix_gain_db")
    fades_accepted = "fade_in_s" in accepted or "fade_out_s" in accepted
    if fade_in is not None and "fade_in_s" in accepted:
        kwargs["fade_in_s"] = fade_in
        log.append(f"fade in {fade_in} s passed to the mix")
    if fade_out is not None and "fade_out_s" in accepted:
        kwargs["fade_out_s"] = fade_out
        log.append(f"fade out {fade_out} s passed to the mix")

    scratch = tempfile.mkdtemp(prefix="bench_mix_")
    effective_voice = str(voice_path)
    effective_music = str(music_path) if music_path else ""

    try:
        # fallback path: the mix has no balance argument, so gain the music file
        if balance is not None and "music_premix_gain_db" not in accepted and effective_music:
            effective_music = pre_gain_file(effective_music, balance, scratch, "music")
            log.append(f"balance {balance} dB applied to the music file instead, "
                       "the mix may renormalize and cancel it")

        if dropped:
            log.append("this mix has no argument for: " + ", ".join(dropped))

        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                returned = mix_function(
                    voice_path=effective_voice,
                    music_path=effective_music,
                    out_path=str(out_path),
                    **kwargs,
                )
        except TypeError as error:
            log.append(f"keyword call rejected ({error}), retrying positionally")
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                returned = mix_function(effective_voice, effective_music, str(out_path))

        printed = captured.getvalue().strip()
        if printed:
            log.append("mix output:")
            for line in printed.splitlines():
                log.append("  " + line)
        if isinstance(returned, int):
            log.append(f"mix returned {returned} ms")

    except HTTPException:
        raise
    except Exception as error:
        log.append("mix failed")
        log.append(traceback.format_exc().strip())
        shutil.rmtree(scratch, ignore_errors=True)
        raise HTTPException(500, f"mix failed: {error}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if not Path(out_path).exists():
        raise HTTPException(500, "mix ran but produced no output file")

    # fades the mix could not take are applied to the finished file
    if (fade_in is not None or fade_out is not None) and not fades_accepted:
        try:
            segment = load_audio(out_path)
            if fade_in is not None:
                segment = segment.fade_in(int(float(fade_in) * 1000))
            if fade_out is not None:
                segment = segment.fade_out(int(float(fade_out) * 1000))
            segment.export(out_path, format="mp3", bitrate="256k")
            log.append("fades applied to the output file, the mix had no fade arguments")
        except Exception as error:
            log.append(f"fade on the output failed: {error}")


def describe_audio(path, label, log):
    try:
        segment = load_audio(path)
        peak = segment.max_dBFS
        peak_text = f"{peak:.1f} dBFS" if peak != float("-inf") else "silent"
        lufs = measure_lufs(segment)
        lufs_text = f", {lufs:.1f} LUFS" if lufs is not None else ""
        log.append(f"{label}: {len(segment)} ms, peak {peak_text}{lufs_text}")
        return len(segment)
    except Exception as error:
        log.append(f"{label}: could not measure ({error})")
        return None


# claude

RETRYABLE = {429, 500, 502, 503, 504, 529}


def call_claude(model, system_prompt, user_prompt, max_tokens=4096):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set on the server")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    last_error = None
    for attempt in range(1, 4):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = "".join(block.text for block in message.content if block.type == "text")
            return text.strip()
        except Exception as error:
            last_error = error
            status = getattr(error, "status_code", None)
            transient = status in RETRYABLE or (status is None and ("connect" in str(error).lower() or "timeout" in str(error).lower()))
            if transient and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise HTTPException(502, f"claude call failed: {error}")
    raise HTTPException(502, f"claude call failed: {last_error}")


AUTOGEN_SYSTEM = (
    "You write a short realistic test answer to an onboarding question for a "
    "meditation app, in the first person. Two or three sentences, plain everyday "
    "language, specific rather than generic. Output only the answer."
)


UPDATE_PROMPT_SYSTEM = """You revise a python prompt file for a meditation speech generator.

You get the current prompt file, the speech it produced, and the editor's feedback on that speech: passages marked good, passages marked bad, comments on them, and text the editor typed directly into the speech.

Rewrite the prompt file so future speeches keep doing what was marked good and stop doing what was marked bad. Fold the intent of the comments and the typed edits into the instructions. Change only what the feedback justifies, and keep everything else exactly as it was. The file must stay a working python file: keep its structure, variable names, and any build_user_prompt or validate functions intact.

Output the complete revised python file and nothing else. No code fences, no explanation before or after."""


def build_feedback_prompt(prompt_py, original_speech, final_speech, marks, edits):
    pieces = ["CURRENT PROMPT FILE\n" + prompt_py.strip()]
    if original_speech.strip():
        pieces.append("SPEECH IT PRODUCED\n" + original_speech.strip())
    good = [m for m in marks if m.get("sentiment") == "good"]
    bad = [m for m in marks if m.get("sentiment") == "bad"]
    if good:
        lines = []
        for mark in good:
            line = f'- "{(mark.get("text") or "").strip()}"'
            if (mark.get("comment") or "").strip():
                line += f' (comment: {mark["comment"].strip()})'
            lines.append(line)
        pieces.append("MARKED GOOD, keep doing this\n" + "\n".join(lines))
    if bad:
        lines = []
        for mark in bad:
            line = f'- "{(mark.get("text") or "").strip()}"'
            if (mark.get("comment") or "").strip():
                line += f' (comment: {mark["comment"].strip()})'
            lines.append(line)
        pieces.append("MARKED BAD, stop doing this\n" + "\n".join(lines))
    if edits:
        pieces.append("TYPED EDITS, text the editor wrote into the speech by hand\n"
                      + "\n".join(f'- "{e.strip()}"' for e in edits if e.strip()))
    if final_speech.strip() and final_speech.strip() != original_speech.strip():
        pieces.append("FINAL EDITED SPEECH\n" + final_speech.strip())
    return "\n\n".join(pieces)


def strip_code_fences(text):
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


# elevenlabs tts
# v3 path: no splitting at pauses. [pause] and [long pause] go to the model
# inline, the model renders the gap, and the rendered gap is then cut or
# extended to the exact length set in the mix panel. splits happen only for
# length, at paragraph boundaries first. chunks come back as pcm where the
# account tier allows and are joined in the sample domain with short fades.
# noise reduction is off on this path, it suppresses the audio around every
# join and swallows the word before a pause.
# v2 path: unchanged. ssml breaks up to 3s are rendered natively, everything
# else becomes inserted silence.

SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
PAUSE_TOKEN = "[pause]"
LONG_PAUSE_TOKEN = "[long pause]"
PAUSE_ANY_RE = re.compile(r"\[long pause\]|\[pause\]")
BREAK_TAG_RE = re.compile(r'<break\s+time="([0-9.]+)\s*(ms|s)"\s*/?\s*>', re.IGNORECASE)
BREAK_CLOSE_RE = re.compile(r"</\s*break\s*>", re.IGNORECASE)
SENTINEL_RE = re.compile(r"(<<<BREAK:\d+>>>)")
MAX_CHARS = 3200
V3_MAX_CHARS = 5000
DEFAULT_PAUSE_MS = 3000
DEFAULT_LONG_PAUSE_MS = 6000
CHUNK_GAP_MS = 350
JOIN_FADE_MS = 15
PAUSE_DETECT_MS = 1200
PAUSE_SILENCE_DBFS = -50.0
PAUSE_FRAME_MS = 5
ZC_SEARCH_MS = 2
CUT_MARGIN_MS = 150
FORMAT_CANDIDATES = ("pcm_44100", "pcm_24000", "mp3_44100_128")
OUT_RATE = 44100


def atomic_units(text, max_chars):
    # (separator, unit) pairs no longer than max_chars. paragraphs stay whole
    # where they fit, over long paragraphs break into sentences, over long
    # sentences are hard split
    units = []
    for paragraph in PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(("\n\n", paragraph))
            continue
        separator = "\n\n"
        for sentence in SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_chars:
                units.append((separator, sentence))
            else:
                for i in range(0, len(sentence), max_chars):
                    units.append((separator if i == 0 else "", sentence[i:i + max_chars]))
            separator = " "
    return units


def split_for_length(text, max_chars=MAX_CHARS):
    # greedy pack of units into the smallest number of chunks under the cap.
    # pause tags are not split points, they ride along inside the text
    max_chars = max(1, min(max_chars, V3_MAX_CHARS))
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks = []
    current = ""
    for separator, unit in atomic_units(text, max_chars):
        candidate = unit if not current else current + separator + unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks


def looks_like_format_rejection(status, body):
    if status not in (403, 422):
        return False
    lowered = (body or "").lower()
    return "format" in lowered or "subscription_required" in lowered


class FormatUnsupported(Exception):
    pass


def synth_chunk_v3(text, voice_id, voice_settings, output_format):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "accept": "*/*", "Content-Type": "application/json"}
    payload = {"text": text, "model_id": "eleven_v3", "voice_settings": voice_settings}
    last_error = None
    for attempt in range(1, 4):
        try:
            response = requests.post(url, headers=headers, json=payload,
                                     params={"output_format": output_format},
                                     stream=True, timeout=120)
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = Exception(f"elevenlabs http {response.status_code}")
                if attempt < 3:
                    time.sleep(1.5 ** attempt)
                    continue
                response.raise_for_status()
            if response.status_code in (403, 422):
                body = ""
                try:
                    body = response.text
                except Exception:
                    pass
                if looks_like_format_rejection(response.status_code, body):
                    raise FormatUnsupported(f"output_format {output_format} rejected: {body[:160]}")
            response.raise_for_status()
            buffer = io.BytesIO()
            for piece in response.iter_content(16384):
                if piece:
                    buffer.write(piece)
            return buffer.getvalue()
        except FormatUnsupported:
            raise
        except requests.RequestException as error:
            last_error = error
            if attempt < 3:
                time.sleep(1.5 ** attempt)
                continue
            raise HTTPException(502, f"tts failed: {error}")
    raise HTTPException(502, f"tts failed: {last_error}")


def synth_chunk_autoformat(text, voice_id, voice_settings, known_format, log):
    # probe the account tier once per synth run, then reuse the format
    candidates = [known_format] if known_format else list(FORMAT_CANDIDATES)
    last = None
    for candidate in candidates:
        try:
            return synth_chunk_v3(text, voice_id, voice_settings, candidate), candidate
        except FormatUnsupported as error:
            last = error
            log.append(f"{error}, trying the next output format")
    raise HTTPException(502, f"no usable output format for this account: {last}")


def decode_blob(raw, output_format):
    # pcm is used as is, mp3 is decoded exactly once
    if output_format.startswith("pcm_"):
        rate = int(output_format.split("_")[1])
        return np.frombuffer(raw, dtype="<i2").astype(np.float32), rate
    segment = AudioSegment.from_file(io.BytesIO(raw), format="mp3").set_channels(1)
    return np.array(segment.get_array_of_samples(), dtype=np.float32), segment.frame_rate


def raised_cosine(n, rising):
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    window = 0.5 * (1.0 - np.cos(np.pi * t))
    return window if rising else window[::-1].copy()


def join_chunks(chunks, rate, gap_ms):
    # chunks come back hard truncated at full level, butting them clicks,
    # so fade both sides of every join
    if len(chunks) == 1:
        return chunks[0]
    n = min(int(rate * JOIN_FADE_MS / 1000), min(len(c) for c in chunks) // 2)
    n = max(n, 1)
    gap = int(rate * max(0, gap_ms) / 1000)
    parts = []
    silence = np.zeros(gap, dtype=np.float32)
    for index, chunk in enumerate(chunks):
        chunk = chunk.copy()
        if index > 0:
            chunk[:n] *= raised_cosine(n, rising=True)
        if index < len(chunks) - 1:
            chunk[-n:] *= raised_cosine(n, rising=False)
        parts.append(chunk)
        if index < len(chunks) - 1:
            parts.append(silence)
    return np.concatenate(parts)


def find_pauses(samples, rate, min_ms=PAUSE_DETECT_MS):
    # silent runs longer than min_ms as [start_ms, end_ms] pairs. frame peak,
    # not rms: meditation speech is delivered very softly and an rms threshold
    # reads whole whispered phrases as silence, which let the normalizer cut
    # them away with the pause next to them
    hop = max(1, int(rate * PAUSE_FRAME_MS / 1000))
    frames_count = len(samples) // hop
    if frames_count == 0:
        return []
    peaks = np.abs(samples[: frames_count * hop]).reshape(frames_count, hop).max(axis=1)
    quiet = peaks < 32768.0 * (10 ** (PAUSE_SILENCE_DBFS / 20.0))
    runs = []
    i = 0
    while i < frames_count:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < frames_count and quiet[j]:
            j += 1
        if (j - i) * PAUSE_FRAME_MS >= min_ms:
            runs.append([i * PAUSE_FRAME_MS, j * PAUSE_FRAME_MS])
        i = j
    return runs


def near_zero(samples, index, radius):
    lo = max(0, index - radius)
    hi = min(len(samples), index + radius + 1)
    if hi <= lo:
        return min(max(index, 0), len(samples))
    return lo + int(np.argmin(np.abs(samples[lo:hi])))


def normalize_pauses(samples, rate, targets_ms, log):
    # drive every rendered pause to its own target length, cutting at zero
    # crossings. fail safe: if the count of long silences does not match the
    # count of pause tags, the audio is returned untouched rather than risk
    # cutting speech
    if not targets_ms:
        return samples
    runs = find_pauses(samples, rate)
    if len(runs) != len(targets_ms):
        log.append(f"pause normalisation skipped, found {len(runs)} long silences "
                   f"for {len(targets_ms)} pause tags, audio left as rendered")
        return samples
    radius = max(1, int(rate * ZC_SEARCH_MS / 1000))
    margin = int(rate * CUT_MARGIN_MS / 1000)
    parts = []
    cursor = 0
    for (start_ms, end_ms), target_ms in zip(runs, targets_ms):
        a = int(start_ms * rate / 1000)
        b = min(len(samples), int(end_ms * rate / 1000))
        have = b - a
        target = int(round(rate * target_ms / 1000))
        mid = (a + b) // 2
        if have > target:
            # cut only from the interior of the silent run, the edges stay,
            # so a misread onset or tail of quiet speech can never be removed
            excess = min(have - target, max(0, have - 2 * margin))
            if excess <= 0:
                continue
            cut_a = near_zero(samples, mid - excess // 2, radius)
            cut_a = max(cursor, max(a + margin, min(cut_a, b - margin - excess)))
            cut_b = near_zero(samples, cut_a + excess, radius)
            cut_b = max(cut_a, min(cut_b, b - margin))
            parts.append(samples[cursor:cut_a])
            cursor = cut_b
        elif have < target:
            insert_at = near_zero(samples, mid, radius)
            insert_at = max(cursor, min(insert_at, len(samples)))
            parts.append(samples[cursor:insert_at])
            parts.append(np.zeros(target - have, dtype=np.float32))
            cursor = insert_at
    parts.append(samples[cursor:])
    out = np.concatenate(parts)
    log.append(f"{len(targets_ms)} pauses set to their exact lengths")
    return out


def break_to_sentinel(match):
    value = float(match.group(1))
    ms = int(value) if match.group(2).lower() == "ms" else int(value * 1000)
    return f" <<<BREAK:{ms}>>> "


STANDALONE_PAUSE_RE = re.compile(r"[ \t]*\n\s*(\[(?:long )?pause\])[ \t]*(?=\n|$)")


def synth_v3(text, voice_id, voice_settings, out_path, pause_ms, long_pause_ms, log):
    raw = text.strip().replace("[breath]", " ")
    # a pause tag written on its own line is glued to the end of the line
    # before it. standing alone the tag loses its context and the model
    # sometimes speaks it or glitches a word coming out of the silence,
    # inline after a sentence is the validated safe position
    glued = STANDALONE_PAUSE_RE.subn(r" \1", raw)
    if glued[1]:
        raw = glued[0]
        log.append(f"{glued[1]} standalone pause tags glued to the line before them")
    raw = BREAK_CLOSE_RE.sub(" ", raw)
    break_tags = len(BREAK_TAG_RE.findall(raw))
    raw = BREAK_TAG_RE.sub(break_to_sentinel, raw)

    long_count = raw.count(LONG_PAUSE_TOKEN)
    short_count = len(PAUSE_ANY_RE.findall(raw)) - long_count
    log.append(f"markers found: {break_tags} break tags, {long_count} [long pause], {short_count} [pause]")
    log.append(f"[pause] {pause_ms} ms, [long pause] {long_pause_ms} ms, set in the mix panel")

    tokens = []
    for part in SENTINEL_RE.split(raw):
        if part.startswith("<<<BREAK:"):
            tokens.append(("break", int(part[9:-3])))
            continue
        part = part.strip()
        if part:
            tokens.append(("speech", part))

    known_format = None
    rate = None
    rendered = []
    api_calls = 0
    inserted_silence_ms = 0

    for kind, value in tokens:
        if kind == "break":
            rendered.append(("break", value))
            inserted_silence_ms += value
            continue
        targets = [long_pause_ms if match.group(0) == LONG_PAUSE_TOKEN else pause_ms
                   for match in PAUSE_ANY_RE.finditer(value)]
        api_text = value.replace(LONG_PAUSE_TOKEN, PAUSE_TOKEN)
        chunks = split_for_length(api_text)
        decoded = []
        for index, chunk in enumerate(chunks):
            log.append(f"tts call {api_calls + 1}, {len(chunk)} chars")
            print(f"tts chunk {index + 1}/{len(chunks)}, {len(chunk)} chars")
            blob, known_format = synth_chunk_autoformat(chunk, voice_id, voice_settings, known_format, log)
            samples, chunk_rate = decode_blob(blob, known_format)
            if rate is None:
                rate = chunk_rate
                log.append(f"audio format {known_format}")
            decoded.append(samples)
            api_calls += 1
        if len(chunks) > 1:
            inserted_silence_ms += CHUNK_GAP_MS * (len(chunks) - 1)
        joined = join_chunks(decoded, rate, CHUNK_GAP_MS)
        joined = normalize_pauses(joined, rate, targets, log)
        rendered.append(("audio", joined))

    if rate is None:
        raise HTTPException(400, "speech text is empty after cleanup")

    pieces = []
    for kind, value in rendered:
        if kind == "break":
            pieces.append(np.zeros(int(rate * value / 1000), dtype=np.float32))
        else:
            pieces.append(value)
    full = np.concatenate(pieces)

    seams = max(0, api_calls - 1)
    log.append(f"{api_calls} tts calls, {seams} length seams, {inserted_silence_ms} ms of inserted silence")
    if seams == 0:
        log.append("one call, no stitching")
    log.append("noise reduction off on the v3 path, it damages the audio around every join")

    segment = AudioSegment(
        data=np.clip(full, -32768, 32767).astype("<i2").tobytes(),
        sample_width=2, frame_rate=rate, channels=1,
    )
    if rate != OUT_RATE:
        segment = segment.set_frame_rate(OUT_RATE)
    segment.export(out_path, format="wav")
    return out_path


def synth_chunk_v2(text, voice_id, voice_settings):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "accept": "audio/mpeg", "Content-Type": "application/json"}
    payload = {"text": text, "model_id": "eleven_multilingual_v2", "voice_settings": voice_settings}
    last_error = None
    for attempt in range(1, 4):
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = Exception(f"elevenlabs http {response.status_code}")
                if attempt < 3:
                    time.sleep(1.5 ** attempt)
                    continue
                response.raise_for_status()
            response.raise_for_status()
            buffer = io.BytesIO()
            for piece in response.iter_content(16384):
                if piece:
                    buffer.write(piece)
            buffer.seek(0)
            return AudioSegment.from_file(buffer, format="mp3")
        except requests.RequestException as error:
            last_error = error
            if attempt < 3:
                time.sleep(1.5 ** attempt)
                continue
            raise HTTPException(502, f"tts failed: {error}")
    raise HTTPException(502, f"tts failed: {last_error}")


def split_sentences_v2(text, max_chars=MAX_CHARS):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    sentences = SENTENCE_SPLIT.split(text)
    chunks = []
    current = []
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        add = len(sentence) + (1 if current_length else 0)
        if current_length + add <= max_chars:
            current.append(sentence)
            current_length += add
        else:
            if current:
                chunks.append(" ".join(current))
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i + max_chars])
                current, current_length = [], 0
            else:
                current, current_length = [sentence], len(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def synth_v2(text, voice_id, voice_settings, out_path, pause_ms, long_pause_ms, log):
    raw = text.strip().replace("[breath]", " ")
    raw = BREAK_CLOSE_RE.sub(" ", raw)

    break_tags = len(BREAK_TAG_RE.findall(raw))
    long_pauses = raw.count(LONG_PAUSE_TOKEN)
    short_pauses = len(PAUSE_ANY_RE.findall(raw)) - long_pauses

    raw = BREAK_TAG_RE.sub(break_to_sentinel, raw)
    raw = raw.replace(LONG_PAUSE_TOKEN, f" <<<BREAK:{long_pause_ms}>>> ")
    raw = raw.replace(PAUSE_TOKEN, f" <<<BREAK:{pause_ms}>>> ")

    log.append(f"markers found: {break_tags} break tags, {long_pauses} [long pause], {short_pauses} [pause]")

    native_breaks = 0

    def native_or_keep(match):
        nonlocal native_breaks
        ms = int(match.group(1))
        if ms <= 3000:
            native_breaks += 1
            return f' <break time="{ms / 1000:.1f}s" /> '
        return match.group(0)

    raw = re.sub(r"<<<BREAK:(\d+)>>>", native_or_keep, raw)
    log.append(f"{native_breaks} pauses rendered natively by v2, no seam at those points")

    parts = SENTINEL_RE.split(raw)
    segments = []
    spoke = False
    api_calls = 0
    inserted_silence_ms = 0
    for part in parts:
        if part.startswith("<<<BREAK:"):
            ms = int(part[9:-3])
            segments.append(AudioSegment.silent(duration=ms, frame_rate=OUT_RATE))
            inserted_silence_ms += ms
            continue
        part = part.strip()
        if not part:
            continue
        chunks = split_sentences_v2(part)
        for chunk_index, chunk in enumerate(chunks):
            print(f"tts chunk {chunk_index + 1}/{len(chunks)}, {len(chunk)} chars")
            log.append(f"tts call {api_calls + 1}, {len(chunk)} chars")
            if chunk_index:
                segments.append(AudioSegment.silent(duration=CHUNK_GAP_MS, frame_rate=OUT_RATE))
                inserted_silence_ms += CHUNK_GAP_MS
            segments.append(synth_chunk_v2(chunk, voice_id, voice_settings))
            api_calls += 1
        spoke = True
    if not spoke:
        raise HTTPException(400, "speech text is empty after cleanup")

    seams = max(0, api_calls - 1)
    log.append(f"{api_calls} tts calls, {seams} seam points, {inserted_silence_ms} ms of inserted silence")
    if seams == 0:
        log.append("one call, no stitching")

    full = segments[0]
    for segment in segments[1:]:
        full += segment

    try:
        import noisereduce
        samples = np.array(full.get_array_of_samples(), dtype=np.float32)
        reduced = noisereduce.reduce_noise(y=samples, sr=full.frame_rate, stationary=True, prop_decrease=0.75)
        reduced_int = np.int16(np.clip(reduced, -32768, 32767))
        full = AudioSegment(data=reduced_int.tobytes(), sample_width=full.sample_width,
                            frame_rate=full.frame_rate, channels=full.channels)
        log.append("noise reduction applied")
    except Exception as error:
        log.append(f"noise reduction skipped: {error}")

    full.export(out_path, format="wav")
    return out_path


def synth(text, voice_id, voice_settings, out_path, tts_provider, pause_ms, long_pause_ms, log):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(503, "ELEVENLABS_API_KEY is not set on the server")
    log.append(f"tts engine {tts_provider}")
    if tts_provider == "eleven_v2":
        return synth_v2(text, voice_id, voice_settings, out_path, pause_ms, long_pause_ms, log)
    return synth_v3(text, voice_id, voice_settings, out_path, pause_ms, long_pause_ms, log)


# context files

def extract_text(path, name):
    lowered = name.lower()
    if lowered.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except Exception as error:
            raise HTTPException(500, f"pdf reader is not installed: {error}")
        try:
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n\n".join(pages).strip(), f"{len(reader.pages)} pages"
        except Exception as error:
            raise HTTPException(400, f"could not read this pdf: {error}")
    if lowered.endswith(".docx"):
        try:
            import docx
        except Exception as error:
            raise HTTPException(500, f"docx reader is not installed: {error}")
        try:
            document = docx.Document(str(path))
            paragraphs = [p.text for p in document.paragraphs]
            return "\n".join(paragraphs).strip(), f"{len(paragraphs)} paragraphs"
        except Exception as error:
            raise HTTPException(400, f"could not read this docx: {error}")
    if lowered.endswith(".doc"):
        raise HTTPException(400, "old .doc files are not supported, open it and save as .docx")
    if lowered.endswith(".txt") or lowered.endswith(".md"):
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace").strip(), "plain text"
        except Exception as error:
            raise HTTPException(400, f"could not read this file: {error}")
    raise HTTPException(400, "supported types are pdf, docx, txt and md")


GROUP_FRAMING = {
    "substance": (
        "SUBSTANCE MATERIAL\n"
        "Source material to draw ideas, images and content from. Do not quote it, "
        "do not mention it, and do not let its vocabulary override the instructions below.\n\n"
    ),
    "form": (
        "FORM MATERIAL\n"
        "Rules and examples for how to write: style, voice, structure. Follow their "
        "approach and level. Do not reuse their content or their sentences.\n\n"
    ),
    "technique": (
        "TECHNIQUE MATERIAL\n"
        "Methods to apply in the piece: how to guide, pace and structure the experience.\n\n"
    ),
}


def build_context_block(connection, context_ids):
    # returns the text to prepend to the system prompt, log lines, and used ids.
    # per doc caps are enforced by truncation, the 80k total is a hard reject
    lines = []
    ids = [int(piece) for piece in str(context_ids).split(",") if piece.strip().isdigit()]
    if not ids:
        return "", lines, []

    rows = []
    for context_id in ids:
        row = connection.execute("select * from context_files where id = ?", (context_id,)).fetchone()
        if row:
            rows.append(row)
    if not rows:
        return "", lines, []

    grouped = {"substance": [], "form": [], "technique": []}
    total = 0
    for row in rows:
        text = row["extracted"] or ""
        cap = DOC_CLASS_CAPS.get(row["size_class"] or "large", DOC_CLASS_CAPS["large"])
        if len(text) > cap:
            text = text[:cap]
            lines.append(f"context {row['name']} cut to its {row['size_class']} cap, {cap} chars")
        total += len(text)
        group = row["kind"] if row["kind"] in grouped else "substance"
        grouped[group].append(f"--- {row['name']} ---\n{text}")
        lines.append(f"context {row['name']} used as {group}, {len(text)} chars")

    if total > CONTEXT_CHAR_CAP:
        raise HTTPException(400, f"selected context is {total} chars, the cap is "
                                 f"{CONTEXT_CHAR_CAP}, untick something")

    block = ""
    for group in CONTEXT_GROUPS:
        if grouped[group]:
            block += GROUP_FRAMING[group] + "\n\n".join(grouped[group]) + "\n\n"
    if block:
        lines.append(f"context total {total} chars of a {CONTEXT_CHAR_CAP} cap")
    return block, lines, [row["id"] for row in rows]


# helpers

def short_model(model):
    if not model:
        return ""
    lowered = model.lower()
    for name in ("opus", "fable", "sonnet", "haiku"):
        if name in lowered:
            return name
    return model


def voice_name_for(connection, voice_id):
    if not voice_id:
        return ""
    row = connection.execute("select name from voices where voice_id = ?", (voice_id,)).fetchone()
    return row["name"].lower() if row else ""


def build_title(connection, topic, day_number, model, voice_id):
    first_line = (topic or "").strip().splitlines()[0] if (topic or "").strip() else ""
    first_line = first_line.strip()
    if len(first_line) > 40:
        first_line = first_line[:40].rstrip() + "..."
    pieces = []
    if first_line:
        pieces.append(first_line)
    pieces.append(f"day {day_number or 1}")
    model_name = short_model(model)
    if model_name:
        pieces.append(model_name)
    voice = voice_name_for(connection, voice_id)
    if voice:
        pieces.append(voice)
    return ", ".join(pieces)


def parse_questions(raw):
    try:
        data = json.loads(raw or "[]")
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    cleaned = []
    for item in data:
        if isinstance(item, dict):
            cleaned.append({
                "question": str(item.get("question") or "").strip(),
                "answer": str(item.get("answer") or "").strip(),
            })
    return cleaned


def questions_block(questions):
    parts = []
    for index, qa in enumerate(questions, start=1):
        if not qa["question"] and not qa["answer"]:
            continue
        answer = qa["answer"] if qa["answer"] else "no answer given"
        parts.append(f"QUESTION {index}: {qa['question']}\nANSWER {index}: {answer}")
    return "\n\n".join(parts)


def topic_from_questions(questions):
    for qa in questions:
        if qa["answer"]:
            return qa["answer"].splitlines()[0].strip()
    for qa in questions:
        if qa["question"]:
            return qa["question"].splitlines()[0].strip()
    return ""


def experiment_row(row):
    try:
        questions = json.loads(row["questions"] or "[]")
    except Exception:
        questions = []
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "title": row["title"] or "",
        "topic": row["topic"],
        "questions": questions,
        "prompt_py": row["prompt_py"],
        "mix_py": row["mix_py"],
        "model": row["model"],
        "voice_id": row["voice_id"],
        "stability": row["stability"],
        "style": row["style"],
        "boost": bool(row["boost"]),
        "music_filename": row["music_filename"],
        "music_file": row["music_file"] or "",
        "speech_text": row["speech_text"],
        "doc_html": row["doc_html"] or "",
        "voice_url": f"/api/bench/audio/{row['voice_file']}" if row["voice_file"] else None,
        "mix_url": f"/api/bench/audio/{row['mix_file']}" if row["mix_file"] else None,
        "mix_source": row["mix_source"] or "",
        "tts_provider": row["tts_provider"] or "elevenlabs",
        "comment": row["comment"],
        "tag": row["tag"] or "",
        "day_number": row["day_number"] or 1,
        "context_ids": row["context_ids"] or "",
        "run_log": row["run_log"] or "",
        "validation": row["validation"] or "",
        "word_count": row["word_count"] or 0,
        "balance_db": row["balance_db"],
        "fade_in_s": row["fade_in_s"],
        "fade_out_s": row["fade_out_s"],
        "pause_ms": row["pause_ms"],
        "long_pause_ms": row["long_pause_ms"],
        "prompt_source": row["prompt_source"] or "",
    }


def save_upload(upload, destination):
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as handle:
        while True:
            piece = upload.file.read(1024 * 1024)
            if not piece:
                break
            handle.write(piece)


def fetch_experiment(connection, experiment_id):
    return connection.execute("select * from experiments where id = ?", (experiment_id,)).fetchone()


def optional_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        raise HTTPException(400, f"expected a number, got {text}")


def optional_int(value, fallback=None):
    number = optional_float(value)
    if number is None:
        return fallback
    return int(number)


def file_still_used(connection, field, value, exclude_id):
    if not value:
        return False
    row = connection.execute(
        f"select count(*) as n from experiments where {field} = ? and id != ?",
        (value, exclude_id),
    ).fetchone()
    return row["n"] > 0


init_db()


# endpoints

class WriteReq(BaseModel):
    questions: list = []
    prompt_py: str = ""
    model: str = "claude-sonnet-4-6"
    context_ids: str = ""


@app.post("/api/bench/write")
def write_answer(req: WriteReq):
    if not req.prompt_py.strip():
        raise HTTPException(400, "paste a prompt file first")

    questions = parse_questions(json.dumps(req.questions))
    topic_block = questions_block(questions)
    topic = topic_from_questions(questions)

    log = [f"write started {now()}"]
    connection = db()
    try:
        system_prompt, user_prompt, source_name, validate = load_prompt_file(
            req.prompt_py, topic_block
        )
        log.append(f"prompt taken from {source_name}")
        log.append(f"model {req.model}")
        answered = sum(1 for qa in questions if qa["answer"])
        log.append(f"{len(questions)} questions, {answered} answered")

        context_block, context_lines, used_ids = build_context_block(connection, req.context_ids)
        log.extend(context_lines)
        if context_block:
            system_prompt = context_block + system_prompt

        log.append(f"system prompt {len(system_prompt)} chars, user prompt {len(user_prompt)} chars")

        speech = call_claude(req.model, system_prompt, user_prompt)
        if not speech:
            raise HTTPException(502, "claude returned empty text")

        word_count = len(speech.split())
        log.append(f"claude returned {word_count} words, {len(speech)} chars")

        validation = run_validate(validate, speech)
        if validate is None:
            log.append("prompt file has no validate(), no check run")
        elif validation:
            log.append("validate() found problems:")
            for line in validation.splitlines():
                log.append("  " + line)
        else:
            log.append("validate() found no problems")

        cursor = connection.execute(
            "insert into experiments (created_at, topic, questions, prompt_py, model, speech_text, "
            "context_ids, day_number, run_log, validation, word_count, prompt_source) "
            "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now(), topic, json.dumps(questions), req.prompt_py, req.model, speech,
             ",".join(str(i) for i in used_ids), 1,
             "\n".join(log), validation, word_count, source_name),
        )
        experiment_id = cursor.lastrowid
        title = build_title(connection, topic, 1, req.model, "")
        connection.execute(
            "update experiments set protocol_id = ?, title = ? where id = ?",
            (experiment_id, title, experiment_id),
        )
        connection.commit()
        print(f"experiment {experiment_id} text saved")
        return {
            "speech": speech,
            "experiment_id": experiment_id,
            "word_count": word_count,
            "validation": validation,
            "prompt_source": source_name,
            "run_log": "\n".join(log),
        }
    finally:
        connection.close()


class AutogenReq(BaseModel):
    question: str = ""


@app.post("/api/bench/autogen_answer")
def autogen_answer(req: AutogenReq):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "no question given")
    answer = call_claude(AUTOGEN_MODEL, AUTOGEN_SYSTEM, question, max_tokens=300)
    if not answer:
        raise HTTPException(502, "claude returned empty text")
    return {"answer": answer}


class UpdatePromptReq(BaseModel):
    experiment_id: int = 0
    model: str = "claude-sonnet-4-6"
    prompt_py: str = ""
    original_speech: str = ""
    final_speech: str = ""
    marks: list = []
    edits: list = []
    doc_html: str = ""


@app.post("/api/bench/update_prompt")
def update_prompt(req: UpdatePromptReq):
    if not req.prompt_py.strip():
        raise HTTPException(400, "no prompt file to revise")
    marks = [m for m in req.marks if isinstance(m, dict)]
    edits = [str(e) for e in req.edits]
    if not marks and not edits and req.final_speech.strip() == req.original_speech.strip():
        raise HTTPException(400, "no feedback to work from, mark or edit the speech first")

    user_content = build_feedback_prompt(req.prompt_py, req.original_speech,
                                         req.final_speech, marks, edits)
    revised = call_claude(req.model, UPDATE_PROMPT_SYSTEM, user_content, max_tokens=16000)
    revised = strip_code_fences(revised)
    if not revised:
        raise HTTPException(502, "claude returned empty text")
    try:
        compile(revised, "revised_prompt.py", "exec")
    except SyntaxError as error:
        raise HTTPException(502, f"the revised prompt is not valid python: {error}")

    if req.experiment_id:
        connection = db()
        if fetch_experiment(connection, req.experiment_id):
            connection.execute("update experiments set doc_html = ? where id = ?",
                               (req.doc_html, req.experiment_id))
            connection.commit()
        connection.close()

    return {"prompt_py": revised,
            "marks_used": len(marks),
            "edits_used": len(edits)}


class DocReq(BaseModel):
    doc_html: str = ""
    speech_text: str = ""


@app.post("/api/bench/experiments/{experiment_id}/doc")
def save_doc(experiment_id: int, req: DocReq):
    connection = db()
    row = fetch_experiment(connection, experiment_id)
    if not row:
        connection.close()
        raise HTTPException(404, "experiment not found")
    if req.speech_text.strip():
        connection.execute("update experiments set doc_html = ?, speech_text = ?, word_count = ? where id = ?",
                           (req.doc_html, req.speech_text, len(req.speech_text.split()), experiment_id))
    else:
        connection.execute("update experiments set doc_html = ? where id = ?",
                           (req.doc_html, experiment_id))
    connection.commit()
    connection.close()
    return {"status": "ok"}


def attach_or_create(connection, experiment_id, topic, questions_json, prompt_py, model, speech):
    # attach audio to the write card if it exists and has no audio yet, else new card
    if experiment_id:
        row = fetch_experiment(connection, experiment_id)
        if row and not row["voice_file"] and not row["mix_file"]:
            return experiment_id, True
    cursor = connection.execute(
        "insert into experiments (created_at, topic, questions, prompt_py, model, speech_text) "
        "values (?, ?, ?, ?, ?, ?)",
        (now(), topic, questions_json, prompt_py, model, speech),
    )
    new_id = cursor.lastrowid
    connection.execute("update experiments set protocol_id = ? where id = ?", (new_id, new_id))
    connection.commit()
    return new_id, False


@app.post("/api/bench/make")
def make_mp3(
    topic: str = Form(""),
    questions: str = Form("[]"),
    speech: str = Form(...),
    prompt_py: str = Form(""),
    mix_py: str = Form(""),
    model: str = Form(""),
    voice_id: str = Form(...),
    stability: float = Form(0.5),
    style: float = Form(0.0),
    boost: bool = Form(True),
    tts_provider: str = Form("elevenlabs"),
    experiment_id: int = Form(0),
    voice_only: bool = Form(False),
    balance_db: str = Form(""),
    fade_in_s: str = Form(""),
    fade_out_s: str = Form(""),
    pause_ms: str = Form(""),
    long_pause_ms: str = Form(""),
    music_ref: str = Form(""),
    music: UploadFile | None = File(default=None),
):
    tts_provider = tts_provider.strip() or "elevenlabs"
    if tts_provider not in ("elevenlabs", "eleven_v2"):
        raise HTTPException(400, "tts provider must be elevenlabs or eleven_v2")
    speech = speech.strip()
    voice_id = voice_id.strip()
    if not speech:
        raise HTTPException(400, "speech text is empty")
    if not voice_id:
        raise HTTPException(400, "voice id is empty")

    settings = {
        "balance_db": optional_float(balance_db),
        "fade_in_s": optional_float(fade_in_s),
        "fade_out_s": optional_float(fade_out_s),
    }
    pause = optional_int(pause_ms, DEFAULT_PAUSE_MS)
    long_pause = optional_int(long_pause_ms, DEFAULT_LONG_PAUSE_MS)

    parsed_questions = parse_questions(questions)
    if not topic.strip():
        topic = topic_from_questions(parsed_questions)

    if voice_only:
        mix_py = ""
        music = None
        music_ref = ""

    mix_function = load_mix_function(mix_py)

    log = [f"make started {now()}"]
    connection = db()
    target_id, attached = attach_or_create(connection, experiment_id, topic.strip(),
                                           json.dumps(parsed_questions), prompt_py, model, speech)
    if attached:
        existing = fetch_experiment(connection, target_id)
        if existing and existing["run_log"]:
            log = existing["run_log"].splitlines() + ["", f"make started {now()}"]

    try:
        music_path = ""
        music_filename = ""
        music_rel = ""
        if music is not None and music.filename:
            # keep the original filename inside a per experiment folder, a prefix
            # would break mix files that read the name to pick a profile
            music_filename = Path(music.filename).name
            music_rel = f"{target_id}/{music_filename}"
            music_path = MUSIC_DIR / music_rel
            save_upload(music, music_path)
            log.append(f"music uploaded as {music_rel}")
        elif music_ref.strip():
            candidate = (MUSIC_DIR / music_ref.strip()).resolve()
            if MUSIC_DIR.resolve() not in candidate.parents:
                raise HTTPException(400, "music reference is outside the music folder")
            if candidate.exists():
                music_path = candidate
                music_rel = music_ref.strip()
                music_filename = candidate.name
                log.append(f"music reused from the server, {music_rel}")
            else:
                log.append(f"music reference {music_ref.strip()} not found on the server, no music used")

        source = mix_source_for(mix_py, music_path)
        log.append(f"mix source {source}")

        voice_settings = {"stability": stability, "similarity_boost": 0.7, "style": style, "use_speaker_boost": boost}
        log.append(f"voice {voice_id}, stability {stability}, style {style}, "
                   f"boost {'on' if boost else 'off'}")

        voice_file = f"{target_id}_voice.wav"
        synth(speech, voice_id, voice_settings, str(AUDIO_DIR / voice_file),
              tts_provider, pause, long_pause, log)
        print(f"experiment {target_id} voice saved")
        describe_audio(AUDIO_DIR / voice_file, "voice", log)

        if voice_only:
            mix_file = ""
            log.append("voice only, no mix run")
        else:
            if music_path:
                describe_audio(music_path, "music", log)
            mix_file = f"{target_id}_mix.mp3"
            run_mix(mix_function, AUDIO_DIR / voice_file, music_path, AUDIO_DIR / mix_file, settings, log)
            print(f"experiment {target_id} mix saved, {source}")
            describe_audio(AUDIO_DIR / mix_file, "output", log)

        row_now = fetch_experiment(connection, target_id)
        title = (row_now["title"] or "") if row_now else ""
        if not title:
            title = build_title(connection, topic.strip(), 1, model, voice_id)

        connection.execute(
            "update experiments set topic = ?, questions = ?, prompt_py = ?, mix_py = ?, model = ?, "
            "voice_id = ?, stability = ?, style = ?, boost = ?, speech_text = ?, music_filename = ?, "
            "music_file = ?, voice_file = ?, mix_file = ?, mix_source = ?, tts_provider = ?, "
            "run_log = ?, word_count = ?, balance_db = ?, fade_in_s = ?, fade_out_s = ?, "
            "pause_ms = ?, long_pause_ms = ?, title = ? where id = ?",
            (topic.strip(), json.dumps(parsed_questions), prompt_py, mix_py, model, voice_id,
             stability, style, int(boost), speech, music_filename, music_rel, voice_file, mix_file,
             source, tts_provider, "\n".join(log), len(speech.split()), settings["balance_db"],
             settings["fade_in_s"], settings["fade_out_s"], pause, long_pause, title, target_id),
        )
        connection.commit()
        row = fetch_experiment(connection, target_id)
        return experiment_row(row)
    except HTTPException:
        if not attached:
            connection.execute("delete from experiments where id = ?", (target_id,))
            connection.commit()
        else:
            connection.execute("update experiments set run_log = ? where id = ?",
                               ("\n".join(log), target_id))
            connection.commit()
        raise
    except Exception as error:
        log.append(traceback.format_exc().strip())
        if not attached:
            connection.execute("delete from experiments where id = ?", (target_id,))
            connection.commit()
        else:
            connection.execute("update experiments set run_log = ? where id = ?",
                               ("\n".join(log), target_id))
            connection.commit()
        raise HTTPException(500, f"make failed: {error}")
    finally:
        connection.close()


@app.delete("/api/bench/experiments/{experiment_id}")
def delete_experiment(experiment_id: int):
    connection = db()
    row = fetch_experiment(connection, experiment_id)
    if not row:
        connection.close()
        raise HTTPException(404, "experiment not found")

    # old remixes shared voice files and reused music, so only remove a file
    # when no other row still points at it
    removed = []
    kept = []
    for field, folder in (("voice_file", AUDIO_DIR), ("mix_file", AUDIO_DIR), ("music_file", MUSIC_DIR)):
        name = row[field]
        if not name:
            continue
        if file_still_used(connection, field, name, experiment_id):
            kept.append(name)
            continue
        path = folder / name
        if path.exists():
            path.unlink()
            removed.append(name)
        if field == "music_file" and "/" in name:
            parent_folder = (MUSIC_DIR / name).parent
            try:
                if parent_folder.exists() and not any(parent_folder.iterdir()):
                    parent_folder.rmdir()
            except Exception:
                pass

    connection.execute("update experiments set prior_id = null where prior_id = ?", (experiment_id,))
    connection.execute("update experiments set parent_id = null where parent_id = ?", (experiment_id,))
    connection.execute("delete from experiments where id = ?", (experiment_id,))
    connection.commit()
    connection.close()
    print(f"deleted experiment {experiment_id}, removed {len(removed)} files, kept {len(kept)} shared")
    return {"status": "ok", "removed": removed, "kept_shared": kept}


@app.get("/api/bench/experiments")
def list_experiments():
    connection = db()
    rows = connection.execute("select * from experiments order by id desc").fetchall()
    connection.close()
    return {"experiments": [experiment_row(row) for row in rows], "tags": TAGS, "models": MODELS}


class TagReq(BaseModel):
    tag: str = ""


@app.post("/api/bench/experiments/{experiment_id}/tag")
def set_tag(experiment_id: int, req: TagReq):
    tag = req.tag.strip()
    if tag and tag not in TAGS:
        raise HTTPException(400, "unknown tag")
    connection = db()
    row = fetch_experiment(connection, experiment_id)
    if not row:
        connection.close()
        raise HTTPException(404, "experiment not found")
    connection.execute("update experiments set tag = ? where id = ?", (tag, experiment_id))
    connection.commit()
    connection.close()
    return {"status": "ok", "tag": tag}


class CommentReq(BaseModel):
    comment: str = ""


@app.post("/api/bench/experiments/{experiment_id}/comment")
def set_comment(experiment_id: int, req: CommentReq):
    connection = db()
    row = fetch_experiment(connection, experiment_id)
    if not row:
        connection.close()
        raise HTTPException(404, "experiment not found")
    connection.execute("update experiments set comment = ? where id = ?",
                       (req.comment.strip(), experiment_id))
    connection.commit()
    connection.close()
    return {"status": "ok"}


class TitleReq(BaseModel):
    title: str = ""


@app.post("/api/bench/experiments/{experiment_id}/title")
def set_title(experiment_id: int, req: TitleReq):
    connection = db()
    row = fetch_experiment(connection, experiment_id)
    if not row:
        connection.close()
        raise HTTPException(404, "experiment not found")
    connection.execute("update experiments set title = ? where id = ?",
                       (req.title.strip()[:200], experiment_id))
    connection.commit()
    connection.close()
    return {"status": "ok"}


# compose state, the whole middle column saved as one blob

class ComposeReq(BaseModel):
    state: dict = {}


@app.get("/api/bench/compose")
def get_compose():
    connection = db()
    row = connection.execute("select state from compose_state where id = 1").fetchone()
    connection.close()
    if not row:
        return {"state": {}}
    try:
        return {"state": json.loads(row["state"])}
    except Exception:
        return {"state": {}}


@app.post("/api/bench/compose")
def save_compose(req: ComposeReq):
    connection = db()
    connection.execute(
        "insert or replace into compose_state (id, state, updated_at) values (1, ?, ?)",
        (json.dumps(req.state), now()),
    )
    connection.commit()
    connection.close()
    return {"status": "ok"}


# saved file library, kept for prompt and mix files

class SaveFileReq(BaseModel):
    kind: str
    name: str
    content: str


@app.get("/api/bench/files")
def list_files(kind: str = ""):
    connection = db()
    if kind:
        rows = connection.execute("select id, created_at, kind, name from saved_files where kind = ? order by id desc", (kind,)).fetchall()
    else:
        rows = connection.execute("select id, created_at, kind, name from saved_files order by id desc").fetchall()
    connection.close()
    return {"files": [dict(row) for row in rows]}


@app.get("/api/bench/files/{file_id}")
def get_file(file_id: int):
    connection = db()
    row = connection.execute("select * from saved_files where id = ?", (file_id,)).fetchone()
    connection.close()
    if not row:
        raise HTTPException(404, "file not found")
    return dict(row)


@app.post("/api/bench/files")
def save_file(req: SaveFileReq):
    if req.kind not in ("prompt", "mix"):
        raise HTTPException(400, "kind must be prompt or mix")
    if not req.name.strip():
        raise HTTPException(400, "name required")
    if not req.content.strip():
        raise HTTPException(400, "content is empty")
    connection = db()
    cursor = connection.execute(
        "insert into saved_files (created_at, kind, name, content) values (?, ?, ?, ?)",
        (now(), req.kind, req.name.strip()[:100], req.content),
    )
    connection.commit()
    file_id = cursor.lastrowid
    connection.close()
    return {"id": file_id}


@app.delete("/api/bench/files/{file_id}")
def delete_file(file_id: int):
    connection = db()
    connection.execute("delete from saved_files where id = ?", (file_id,))
    connection.commit()
    connection.close()
    return {"status": "ok"}


# context library

@app.get("/api/bench/context")
def list_context():
    connection = db()
    rows = connection.execute(
        "select id, created_at, name, kind, size_class, chars, note from context_files order by id desc"
    ).fetchall()
    connection.close()
    return {"context": [dict(row) for row in rows], "cap": CONTEXT_CHAR_CAP,
            "class_caps": DOC_CLASS_CAPS, "groups": list(CONTEXT_GROUPS)}


@app.get("/api/bench/context/{context_id}")
def get_context(context_id: int):
    connection = db()
    row = connection.execute("select * from context_files where id = ?", (context_id,)).fetchone()
    connection.close()
    if not row:
        raise HTTPException(404, "context file not found")
    data = dict(row)
    data["preview"] = (row["extracted"] or "")[:4000]
    data.pop("extracted", None)
    return data


@app.post("/api/bench/context")
def upload_context(
    kind: str = Form("substance"),
    file: UploadFile = File(...),
):
    kind = kind.strip().lower()
    if kind not in CONTEXT_GROUPS:
        raise HTTPException(400, "kind must be substance, form or technique")
    if not file or not file.filename:
        raise HTTPException(400, "no file uploaded")

    name = Path(file.filename).name
    connection = db()
    cursor = connection.execute(
        "insert into context_files (created_at, name, kind, stored_name) values (?, ?, ?, ?)",
        (now(), name, kind, ""),
    )
    context_id = cursor.lastrowid
    connection.commit()

    stored_name = f"{context_id}_{name}"
    stored_path = CONTEXT_DIR / stored_name
    try:
        save_upload(file, stored_path)
        text, note = extract_text(stored_path, name)
        if not text.strip():
            raise HTTPException(400, "no text could be read from this file, it may be a scan")
        if len(text) > DOC_CLASS_CAPS["large"]:
            raise HTTPException(400, f"this file is {len(text)} chars, the large cap is "
                                     f"{DOC_CLASS_CAPS['large']}, trim it down first")
        connection.execute(
            "update context_files set stored_name = ?, extracted = ?, chars = ?, note = ?, "
            "size_class = ? where id = ?",
            (stored_name, text, len(text), note, size_class_for(len(text)), context_id),
        )
        connection.commit()
        print(f"context {context_id} saved: {name}, {len(text)} chars")
        row = connection.execute(
            "select id, created_at, name, kind, size_class, chars, note from context_files where id = ?",
            (context_id,),
        ).fetchone()
        return dict(row)
    except HTTPException:
        connection.execute("delete from context_files where id = ?", (context_id,))
        connection.commit()
        if stored_path.exists():
            stored_path.unlink()
        raise
    except Exception as error:
        connection.execute("delete from context_files where id = ?", (context_id,))
        connection.commit()
        if stored_path.exists():
            stored_path.unlink()
        raise HTTPException(500, f"context upload failed: {error}")
    finally:
        connection.close()


@app.delete("/api/bench/context/{context_id}")
def delete_context(context_id: int):
    connection = db()
    row = connection.execute("select * from context_files where id = ?", (context_id,)).fetchone()
    if not row:
        connection.close()
        raise HTTPException(404, "context file not found")
    if row["stored_name"]:
        path = CONTEXT_DIR / row["stored_name"]
        if path.exists():
            path.unlink()
    connection.execute("delete from context_files where id = ?", (context_id,))
    connection.commit()
    connection.close()
    return {"status": "ok"}


# voices

class VoiceReq(BaseModel):
    name: str
    voice_id: str


@app.get("/api/bench/voices")
def list_voices():
    connection = db()
    rows = connection.execute("select * from voices order by name").fetchall()
    connection.close()
    return {"voices": [dict(row) for row in rows]}


@app.post("/api/bench/voices")
def save_voice(req: VoiceReq):
    name = req.name.strip()[:60]
    voice_id = req.voice_id.strip()
    if not name:
        raise HTTPException(400, "name required")
    if not voice_id:
        raise HTTPException(400, "voice id required")
    connection = db()
    existing = connection.execute("select id from voices where voice_id = ?", (voice_id,)).fetchone()
    if existing:
        connection.execute("update voices set name = ? where id = ?", (name, existing["id"]))
        voice_row_id = existing["id"]
    else:
        cursor = connection.execute(
            "insert into voices (created_at, name, voice_id) values (?, ?, ?)",
            (now(), name, voice_id),
        )
        voice_row_id = cursor.lastrowid
    connection.commit()
    connection.close()
    return {"id": voice_row_id, "name": name, "voice_id": voice_id}


@app.delete("/api/bench/voices/{voice_row_id}")
def delete_voice(voice_row_id: int):
    connection = db()
    connection.execute("delete from voices where id = ?", (voice_row_id,))
    connection.commit()
    connection.close()
    return {"status": "ok"}


@app.get("/api/bench/experiments/{experiment_id}/zip")
def get_zip(experiment_id: int):
    connection = db()
    row = fetch_experiment(connection, experiment_id)
    if not row:
        connection.close()
        raise HTTPException(404, "experiment not found")

    context_rows = []
    for piece in str(row["context_ids"] or "").split(","):
        if piece.strip().isdigit():
            found = connection.execute(
                "select * from context_files where id = ?", (int(piece),)
            ).fetchone()
            if found:
                context_rows.append(found)
    connection.close()

    try:
        questions = json.loads(row["questions"] or "[]")
    except Exception:
        questions = []

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        if row["prompt_py"]:
            bundle.writestr("prompt.py", row["prompt_py"])
        if row["mix_py"]:
            bundle.writestr("mix.py", row["mix_py"])
        elif row["mix_file"]:
            bundle.writestr("default_mix.py", DEFAULT_MIX_SOURCE)
        if row["speech_text"]:
            bundle.writestr("answer.txt", row["speech_text"])
        if questions:
            lines = []
            for index, qa in enumerate(questions, start=1):
                lines.append(f"Q{index}: {qa.get('question', '')}")
                lines.append(f"A{index}: {qa.get('answer', '')}")
                lines.append("")
            bundle.writestr("questions.txt", "\n".join(lines))
        if row["doc_html"]:
            bundle.writestr("feedback.html", row["doc_html"])
        if row["run_log"]:
            bundle.writestr("log.txt", row["run_log"])
        if row["validation"]:
            bundle.writestr("validation.txt", row["validation"])

        info = [
            f"experiment {row['id']}",
            f"title: {row['title'] or 'none'}",
            f"created {row['created_at']}",
            f"day {row['day_number'] or 1}",
            f"topic: {row['topic'] or 'none'}",
            f"model: {row['model'] or 'none'}",
            f"prompt taken from: {row['prompt_source'] or 'unknown'}",
            f"word count: {row['word_count'] or 0}",
            f"voice id: {row['voice_id'] or 'none'}",
            f"stability: {row['stability']}",
            f"style: {row['style']}",
            f"speaker boost: {'on' if row['boost'] else 'off'}",
            f"music: {row['music_filename'] or 'none'}",
            f"mix source: {row['mix_source'] or 'none'}",
            f"balance db: {row['balance_db'] if row['balance_db'] is not None else 'mix default'}",
            f"fade in s: {row['fade_in_s'] if row['fade_in_s'] is not None else 'mix default'}",
            f"fade out s: {row['fade_out_s'] if row['fade_out_s'] is not None else 'mix default'}",
            f"pause ms: {row['pause_ms'] if row['pause_ms'] is not None else DEFAULT_PAUSE_MS}",
            f"long pause ms: {row['long_pause_ms'] if row['long_pause_ms'] is not None else DEFAULT_LONG_PAUSE_MS}",
            f"tts provider: {row['tts_provider'] or 'elevenlabs'}",
            f"tag: {row['tag'] or 'none'}",
            f"comment: {row['comment'] or 'none'}",
        ]
        bundle.writestr("info.txt", "\n".join(info) + "\n")

        if context_rows:
            listing = []
            for found in context_rows:
                listing.append(f"{found['name']}  group {found['kind']}  {found['size_class']}  "
                               f"{found['chars']} chars  {found['note']}")
                source = CONTEXT_DIR / (found["stored_name"] or "")
                if found["stored_name"] and source.exists():
                    bundle.write(source, f"context/{found['name']}")
            bundle.writestr("context/context.txt", "\n".join(listing) + "\n")

        if row["mix_file"] and (AUDIO_DIR / row["mix_file"]).exists():
            bundle.write(AUDIO_DIR / row["mix_file"], "mix.mp3")
        if row["voice_file"] and (AUDIO_DIR / row["voice_file"]).exists():
            bundle.write(AUDIO_DIR / row["voice_file"], "voice.wav")
        if row["music_file"] and (MUSIC_DIR / row["music_file"]).exists():
            bundle.write(MUSIC_DIR / row["music_file"], "music_" + (row["music_filename"] or "track"))

    buffer.seek(0)
    return Response(
        content=buffer.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=experiment_{experiment_id}.zip"},
    )


@app.get("/api/bench/audio/{filename}")
def get_audio(filename: str, request: Request):
    safe_name = Path(filename).name
    file_path = AUDIO_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(404, "file not found")
    media_type = "audio/mpeg" if safe_name.endswith(".mp3") else "audio/wav"
    data = file_path.read_bytes()
    total = len(data)

    range_header = request.headers.get("range")
    if range_header:
        try:
            range_spec = range_header.strip()
            if range_spec.lower().startswith("bytes="):
                range_spec = range_spec[6:]
            parts = range_spec.split("-", 1)
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if len(parts) > 1 and parts[1] else total - 1
            end = min(end, total - 1)
            if start > end or start >= total:
                return Response(content=b"", status_code=416,
                                headers={"Content-Range": f"bytes */{total}"})
            chunk = data[start:end + 1]
            return Response(
                content=chunk,
                status_code=206,
                media_type=media_type,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{total}",
                    "Content-Length": str(len(chunk)),
                    "Accept-Ranges": "bytes",
                    "Content-Disposition": f"inline; filename={safe_name}",
                },
            )
        except (ValueError, IndexError):
            pass

    return Response(
        content=data,
        media_type=media_type,
        headers={
            "Content-Length": str(total),
            "Accept-Ranges": "bytes",
            "Content-Disposition": f"inline; filename={safe_name}",
        },
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "stimgen-3"}


HTML_PATH = Path(__file__).resolve().parent / "bench.html"


@app.get("/")
def serve_ui():
    if HTML_PATH.exists():
        return FileResponse(str(HTML_PATH), media_type="text/html", headers={"Cache-Control": "no-store"})
    return JSONResponse({"error": "bench.html not found"}, status_code=404)
