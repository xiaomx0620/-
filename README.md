#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
日语视频自动字幕生成工具 (独立术语替换版)
- 阶段1：语音识别 (默认 SenseVoiceSmall)
- 阶段2：本地术语强制替换 (独立步骤，逐句匹配替换)
- 阶段3：多引擎翻译 + 智能择优
"""

import os
import sys
import time
import re
import hashlib
import random
import requests
import difflib
import threading
import queue
import json
import webbrowser
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
from concurrent.futures import ThreadPoolExecutor, as_completed
import av
from opencc import OpenCC

# ---------------------------- 依赖检查 ----------------------------
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# ======================== 默认配置 =========================
DEFAULT_CONFIG = {
    "youdao": {"enabled": False, "appid": "", "secret": ""},
    "baidu": {"enabled": False, "appid": "", "key": ""},
    "deeplx": {
        "enabled": False,
        "url": "https://api.deeplx.org/translate"
    },
    "siliconflow_translate": {
        "enabled": True,
        "api_key": "",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "url": "https://api.siliconflow.cn/v1/chat/completions"
    },
    "siliconflow_stt": {
        "enabled": True,
        "api_key": "",
        "model": "FunAudioLLM/SenseVoiceSmall",
        "url": "https://api.siliconflow.cn/v1/audio/transcriptions"
    },
    "speech_engine": {
        "primary": "siliconflow_stt",
        "fallback_enabled": True,
        "whisper": {
            "enabled": False,
            "device": "cuda",
            "compute_type": "float16",
            "model_size": "large-v3"
        }
    },
    "params": {
        "max_retry": 2,
        "retry_delay": 0.35,
        "translate_timeout": 15,
        "parallel_workers": 3,
        "enable_context": True,
        "silence_min_len": 350,
        "silence_thresh": -38,
        "keep_silence": 200,
        "enable_local_term_replace": True
    }
}

CONFIG_FILE = "config.json"
REQUIRED_PACKAGES = ["faster-whisper", "pydub", "av", "opencc-python-reimplemented", "requests", "tqdm"]
PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

# ---------------------------- 配置管理 ----------------------------
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            config = DEFAULT_CONFIG.copy()
            for key in config:
                if key in user_config:
                    if isinstance(config[key], dict):
                        config[key].update(user_config[key])
                    else:
                        config[key] = user_config[key]
            return config
        except Exception:
            return DEFAULT_CONFIG.copy()
    else:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        return DEFAULT_CONFIG.copy()

def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

CONFIG = load_config()
cc = OpenCC("t2s")

# ---------------------------- 【核心】本地术语解析与替换引擎 ----------------------------
def parse_term_rules(context_text):
    """
    从背景文本中解析所有术语规则，返回：
    - term_list: 按长度降序排列的 (日文, 中文, 是否整句) 列表
    - forward_map: 日文 -> 占位符
    - reverse_map: 占位符 -> 中文
    """
    if not context_text or not CONFIG["params"].get("enable_local_term_replace", True):
        return [], {}, {}

    term_list = []
    lines = context_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 兼容多种箭头格式
        arrow = None
        if '→' in line:
            arrow = '→'
        elif '->' in line:
            arrow = '->'
        elif '→' in line:
            arrow = '→'
        
        if not arrow:
            continue
        
        parts = line.split(arrow, 1)
        if len(parts) != 2:
            continue
        
        jp = parts[0].strip()
        cn = parts[1].strip()
        if not jp or not cn:
            continue
        
        # 判断是否为整句替换（带标点或长度较长）
        is_full_sentence = len(jp) >= 6 and ('。' in jp or '！' in jp or '？' in jp)
        term_list.append((jp, cn, is_full_sentence))

    # 按日文长度降序，长词优先替换
    term_list.sort(key=lambda x: len(x[0]), reverse=True)

    # 构建占位符映射
    forward_map = {}
    reverse_map = {}
    for idx, (jp, cn, _) in enumerate(term_list):
        placeholder = f"{{{{TERM_{idx:03d}}}}}"
        forward_map[jp] = placeholder
        reverse_map[placeholder] = cn

    return term_list, forward_map, reverse_map

def apply_term_replace(text, term_list, forward_map, log_callback=None, line_idx=0):
    """
    对单句应用术语替换
    返回: (处理后的文本, 替换次数, 替换详情列表, 是否整句命中)
    """
    if not term_list:
        return text, 0, [], False

    original = text
    replace_details = []
    replace_count = 0

    # 先检查整句匹配
    for jp, cn, is_full in term_list:
        if is_full and text.strip() == jp.strip():
            if log_callback:
                log_callback(f"  第{line_idx}句：整句命中「{jp}」→ 直接使用「{cn}」\n")
            return cn, 1, [(jp, cn)], True

    # 再做局部关键词替换
    for jp, cn, _ in term_list:
        if jp in text:
            placeholder = forward_map[jp]
            text = text.replace(jp, placeholder)
            replace_count += 1
            replace_details.append((jp, cn))

    if replace_count > 0 and log_callback:
        detail_str = "、".join([f"{j}→{c}" for j,c in replace_details])
        log_callback(f"  第{line_idx}句：替换 {replace_count} 个术语 [{detail_str}]\n")

    return text, replace_count, replace_details, False

def restore_placeholders(text, reverse_map):
    """把译文中的占位符还原为中文术语"""
    if not reverse_map:
        return text
    for placeholder, cn in reverse_map.items():
        text = text.replace(placeholder, cn)
    return text

# ---------------------------- 工具函数 ----------------------------
def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def is_valid_translation(text, original):
    if not text:
        return False
    text = text.strip()
    if not text or text == original.strip():
        return False
    if "error" in text.lower() or "错误" in text or "code" in text.lower():
        return False
    if "http" in text.lower() or "www." in text.lower():
        return False
    if "<" in text and ">" in text:
        return False
    if all(ch.isascii() for ch in text) and len(text) > 2:
        return False
    invalid_prefixes = [
        "翻译", "译文", "这句话", "这句", "意思是",
        "翻译成中文是", "中文是", "解析", "结果：",
        "翻译结果", "答案：", "回答：", "以下是",
        "「", "」", "『", "』", "“", "”"
    ]
    for prefix in invalid_prefixes:
        if text.startswith(prefix):
            return False
    chinese_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    total_chars = len(text)
    if total_chars <= 2:
        return True
    if chinese_chars / total_chars < 0.4:
        if any('\u3040' <= ch <= '\u309f' or '\u30a0' <= ch <= '\u30ff' for ch in text):
            return False
        if all(ch.isascii() for ch in text):
            return False
    if len(text) > len(original) * 3:
        return False
    return True

def calc_quality_score(text, original):
    score = 0.0
    text = text.strip()
    if not text:
        return 0.0
    chinese_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    total_chars = len(text)
    if total_chars > 0:
        score += (chinese_chars / total_chars) * 50
    ratio = len(text) / max(len(original), 1)
    if 0.5 <= ratio <= 2.0:
        score += 30
    elif 0.3 <= ratio <= 3.0:
        score += 15
    if not text[0] in "，。！？、；：":
        score += 10
    bad_words = ["那个", "这个", "嗯", "呃", "啊", "哦"]
    bad_count = sum(text.count(w) for w in bad_words)
    score += max(0, 10 - bad_count * 2)
    return score

# ---------------------------- 语音识别引擎 ----------------------------
def whisper_transcribe(video_path, log_callback=None, stop_event=None):
    if not WHISPER_AVAILABLE:
        if log_callback:
            log_callback("❌ [Whisper] 未安装 faster-whisper，请先部署环境。\n")
        return None, "依赖缺失"
    model_size = CONFIG["speech_engine"]["whisper"]["model_size"]
    device = CONFIG["speech_engine"]["whisper"]["device"]
    compute_type = CONFIG["speech_engine"]["whisper"]["compute_type"]
    if log_callback:
        log_callback(f"🚀 [Whisper] 加载 {model_size} 模型...\n")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        if log_callback:
            log_callback(f"❌ [Whisper] 模型加载失败: {e}\n")
        return None, f"模型加载失败: {e}"
    if log_callback:
        log_callback("🎤 [Whisper] 开始日语识别...\n")
    try:
        segments, _ = model.transcribe(
            video_path,
            beam_size=10,
            best_of=10,
            language="ja",
            word_timestamps=False,
            vad_filter=True,
            temperature=(0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0),
            compression_ratio_threshold=2.2,
            no_speech_threshold=0.55,
            condition_on_previous_text=False
        )
        if stop_event and stop_event.is_set():
            return None, "已中断"
        seg_list = [{"start": s.start, "end": s.end, "jp": s.text.strip()} for s in segments]
        if log_callback:
            log_callback(f"✅ [Whisper] 识别完成，共 {len(seg_list)} 条\n")
        return seg_list, "成功"
    except Exception as e:
        if log_callback:
            log_callback(f"❌ [Whisper] 识别失败: {e}\n")
        return None, f"识别失败: {e}"

def siliconflow_stt_transcribe(video_path, log_callback=None, stop_event=None):
    if not PYDUB_AVAILABLE:
        if log_callback:
            log_callback("❌ [云端STT] 未安装 pydub，请先部署环境。\n")
        return None, "依赖缺失"
    
    api_key = CONFIG["siliconflow_stt"]["api_key"] or CONFIG["siliconflow_translate"]["api_key"]
    if not api_key:
        if log_callback:
            log_callback("❌ [云端STT] 未配置 API Key，请在设置中填写。\n")
            log_callback("💡 获取地址：https://siliconflow.cn/\n")
        return None, "未配置 API Key"
    
    model = CONFIG["siliconflow_stt"]["model"]
    url = CONFIG["siliconflow_stt"]["url"]
    params = CONFIG["params"]
    
    if log_callback:
        log_callback(f"🎤 [云端STT] 使用模型: {model}\n")
        log_callback("🎵 提取音频并重采样为16kHz单声道...\n")
    
    try:
        with av.open(video_path) as container:
            audio_stream = container.streams.audio[0]
            resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=16000)
            pcm = bytearray()
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    for f in resampler.resample(frame):
                        pcm.extend(f.planes[0])
            full_audio = AudioSegment(
                data=bytes(pcm),
                sample_width=2,
                frame_rate=16000,
                channels=1
            )
    except Exception as e:
        if log_callback:
            log_callback(f"❌ 音频提取失败: {e}\n")
        return None, f"音频提取失败: {e}"
    
    if log_callback:
        log_callback(f"🔊 音频总时长: {len(full_audio)/1000:.1f} 秒\n")
        log_callback("✂️ 按静音分割片段...\n")
    
    try:
        chunks = split_on_silence(
            full_audio,
            min_silence_len=params["silence_min_len"],
            silence_thresh=params["silence_thresh"],
            keep_silence=params["keep_silence"]
        )
    except Exception as e:
        if log_callback:
            log_callback(f"❌ 静音分割失败: {e}\n")
        return None, f"静音分割失败: {e}"
    
    if not chunks:
        if log_callback:
            log_callback("⚠️ 未检测到有效语音片段\n")
        return None, "无语音片段"
    
    if log_callback:
        log_callback(f"📦 分割为 {len(chunks)} 个语音片段\n")
    
    seg_list = []
    current_time = 0.0
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    
    for idx, chunk in enumerate(chunks):
        if stop_event and stop_event.is_set():
            if log_callback:
                log_callback(f"⏹️ 用户停止（已处理 {idx}/{len(chunks)}）\n")
            break
        
        start_ms = current_time * 1000
        end_ms = start_ms + len(chunk)
        if log_callback:
            log_callback(f"  片段 {idx+1}/{len(chunks)}: {start_ms/1000:.1f}s-{end_ms/1000:.1f}s ... ")
        
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                chunk.export(tmp_path, format="wav")
            
            with open(tmp_path, 'rb') as f:
                file_data = f.read()
            
            files = {
                'file': (os.path.basename(tmp_path), file_data, 'audio/wav'),
                'model': (None, model)
            }
            data = {'language': 'ja'}
            
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=30)
            
            if resp.status_code == 200:
                text = resp.json().get('text', '').strip()
                if text:
                    seg_list.append({
                        "start": start_ms / 1000.0,
                        "end": end_ms / 1000.0,
                        "jp": text
                    })
                    if log_callback:
                        log_callback(f"✅ {text[:30]}{'...' if len(text)>30 else ''}\n")
                else:
                    if log_callback:
                        log_callback("⚠️ 空结果\n")
            else:
                err_msg = resp.text[:200]
                if log_callback:
                    log_callback(f"❌ HTTP {resp.status_code}: {err_msg}\n")
                if "Model does not exist" in err_msg:
                    if log_callback:
                        log_callback("💡 模型名称错误，请在设置中检查模型名。\n")
                elif "Unauthorized" in err_msg or "invalid_api_key" in err_msg:
                    if log_callback:
                        log_callback("💡 API Key 无效，请检查密钥是否正确。\n")
        except Exception as e:
            if log_callback:
                log_callback(f"❌ 异常: {e}\n")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
        
        current_time = end_ms / 1000.0
    
    if stop_event and stop_event.is_set():
        if seg_list:
            if log_callback:
                log_callback(f"⏹️ 已停止，返回 {len(seg_list)} 条\n")
            return seg_list, "已中断"
        return None, "已中断"
    
    if log_callback:
        log_callback(f"✅ 识别完成，共 {len(seg_list)} 条字幕\n")
    return seg_list, "成功" if seg_list else "无结果"

def recognize_speech(video_path, log_callback=None, stop_event=None):
    primary = CONFIG["speech_engine"]["primary"]
    fallback = CONFIG["speech_engine"]["fallback_enabled"]
    
    if primary == "whisper":
        primary_enabled = CONFIG["speech_engine"]["whisper"]["enabled"]
    else:
        primary_enabled = CONFIG["siliconflow_stt"]["enabled"]
    
    if not primary_enabled:
        if log_callback:
            log_callback(f"⚠️ 主引擎 {primary} 未启用\n")
        primary_result = None
        primary_status = "未启用"
    else:
        if log_callback:
            log_callback(f"🔊 主引擎: {primary}\n")
        if primary == "whisper":
            primary_result, primary_status = whisper_transcribe(video_path, log_callback, stop_event)
        else:
            primary_result, primary_status = siliconflow_stt_transcribe(video_path, log_callback, stop_event)
    
    if primary_result and len(primary_result) > 0:
        return primary_result
    
    if fallback:
        backup = "whisper" if primary != "whisper" else "siliconflow_stt"
        
        if backup == "whisper":
            backup_enabled = CONFIG["speech_engine"]["whisper"]["enabled"]
        else:
            backup_enabled = CONFIG["siliconflow_stt"]["enabled"]
        
        if not backup_enabled:
            if log_callback:
                log_callback(f"⚠️ 备用引擎 {backup} 未启用\n")
            return None
        
        if log_callback:
            log_callback(f"🔁 切换到备用引擎: {backup}\n")
        
        if backup == "whisper":
            backup_result, _ = whisper_transcribe(video_path, log_callback, stop_event)
        else:
            backup_result, _ = siliconflow_stt_transcribe(video_path, log_callback, stop_event)
        
        return backup_result
    else:
        if log_callback:
            log_callback("❌ 主引擎失败且备用未启用\n")
        return None

# ---------------------------- 翻译接口 ----------------------------
def translate_youdao(text):
    if not CONFIG["youdao"]["enabled"]:
        return None, "未启用"
    appid, secret = CONFIG["youdao"]["appid"], CONFIG["youdao"]["secret"]
    if not appid or not secret:
        return None, "缺少密钥"
    salt = str(int(time.time() * 1000))
    sign = hashlib.md5(f"{appid}{text}{salt}{secret}".encode()).hexdigest()
    params = {"q": text, "from": "ja", "to": "zh-CHS", "appKey": appid, "salt": salt, "sign": sign}
    try:
        resp = requests.get("https://openapi.youdao.com/api", params=params, timeout=CONFIG["params"]["translate_timeout"])
        res = resp.json()
        error_code = res.get("errorCode", "")
        if error_code == "101":
            return None, "密钥错误"
        if error_code == "102":
            return None, "余额不足"
        if error_code == "411":
            return None, "频率超限"
        if res.get("translation"):
            translated = cc.convert(res["translation"][0].strip())
            if is_valid_translation(translated, text):
                return translated, "成功"
            return None, "译文无效"
        return None, f"错误码:{error_code}"
    except Exception as e:
        return None, "请求异常"

def translate_deeplx(text):
    if not CONFIG["deeplx"]["enabled"]:
        return None, "未启用"
    url = CONFIG["deeplx"]["url"]
    if not url:
        return None, "URL为空"
    try:
        resp = requests.post(url, json={"text": text, "source_lang": "JA", "target_lang": "ZH"}, timeout=CONFIG["params"]["translate_timeout"])
        if resp.status_code == 200:
            data = resp.json().get("data", "").strip()
            if "Hello DeeplX" in data or len(data) < 2:
                return None, "接口失效"
            translated = cc.convert(data)
            if is_valid_translation(translated, text):
                return translated, "成功"
            return None, "译文无效"
        return None, f"HTTP{resp.status_code}"
    except Exception as e:
        return None, "请求异常"

def translate_baidu(text):
    if not CONFIG["baidu"]["enabled"]:
        return None, "未启用"
    appid, key = CONFIG["baidu"]["appid"], CONFIG["baidu"]["key"]
    if not appid or not key:
        return None, "缺少密钥"
    salt = str(random.randint(10000, 99999))
    sign = hashlib.md5(f"{appid}{text}{salt}{key}".encode()).hexdigest()
    params = {"q": text, "from": "jp", "to": "zh", "appid": appid, "salt": salt, "sign": sign}
    try:
        resp = requests.get("https://fanyi-api.baidu.com/api/trans/vip/translate", params=params, timeout=CONFIG["params"]["translate_timeout"])
        res = resp.json()
        if "error_code" in res:
            return None, f"错误码:{res['error_code']}"
        if "trans_result" in res:
            translated = cc.convert(res["trans_result"][0]["dst"].strip())
            if is_valid_translation(translated, text):
                return translated, "成功"
            return None, "译文无效"
        return None, "未知错误"
    except Exception as e:
        return None, "请求异常"

def translate_siliconflow(text, context="", prev_text=""):
    if not CONFIG["siliconflow_translate"]["enabled"]:
        return None, "未启用"
    api_key = CONFIG["siliconflow_translate"]["api_key"]
    url = CONFIG["siliconflow_translate"]["url"]
    model = CONFIG["siliconflow_translate"]["model"]
    if not api_key:
        return None, "缺少密钥"
    if not url or not model:
        return None, "缺少配置"
    
    system_prompt = """你是专业的动漫字幕翻译员，负责将日文台词翻译成简体中文字幕。
【强制规则】
1. 仅输出简体中文译文，绝对不要加任何前缀、解释、备注、引号，不要重复原文
2. 句子中的 {{TERM_XXX}} 格式占位符必须原样保留，绝对不要修改、翻译或删除
3. 口语化、自然流畅，符合中文日常说话习惯，符合动漫人物语气
4. 译文长度适合做字幕，简洁通顺，不要过度扩展、不要补充内容"""
    
    if prev_text:
        system_prompt += f"\n【上一句译文】{prev_text}\n保持剧情和语气连贯性。"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.2,
        "max_tokens": 256,
        "top_p": 0.9
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=CONFIG["params"]["translate_timeout"])
        res = resp.json()
        if res.get("choices") and len(res["choices"]) > 0:
            translated = res["choices"][0]["message"]["content"].strip()
            translated = cc.convert(translated)
            translated = translated.strip('"').strip("'").strip("「」").strip("『』")
            if is_valid_translation(translated, text):
                return translated, "成功"
            return None, "译文无效"
        err = res.get("error", {}).get("message", "未知错误")
        return None, err[:12]
    except Exception as e:
        return None, "请求异常"

# ---------------------------- 智能择优融合 ----------------------------
def merge_translation(jp_origin, results, statuses):
    candidates = [(name, text) for name, text in results.items()
                  if text and text.strip() != jp_origin.strip() and is_valid_translation(text, jp_origin)]
    
    if not candidates:
        return jp_origin.strip(), "全部失效", statuses
    if len(candidates) == 1:
        return candidates[0][1], candidates[0][0], statuses

    final_scores = {}
    for idx, (name, text) in enumerate(candidates):
        sim_sum = 0.0
        for j, (other_name, other_text) in enumerate(candidates):
            if idx != j:
                sim_sum += difflib.SequenceMatcher(None, text, other_text).ratio()
        sim_score = (sim_sum / (len(candidates) - 1)) * 100 if len(candidates) > 1 else 0
        quality_score = calc_quality_score(text, jp_origin)
        final_scores[name] = sim_score * 0.6 + quality_score * 0.4

    best_name = max(final_scores, key=final_scores.get)
    best_text = next(t for n, t in candidates if n == best_name)
    return best_text, f"智能择优({best_name})", statuses

# ---------------------------- 核心处理流程 ----------------------------
def process_video(video_path, log_callback, finish_callback, stop_event, context=""):
    try:
        log_callback(f"📂 文件: {video_path}\n")

        # 解析术语规则
        term_list, forward_map, reverse_map = parse_term_rules(context)
        if term_list:
            log_callback(f"📝 已解析术语规则 {len(term_list)} 条，阶段二将逐句匹配替换\n")
        else:
            log_callback("ℹ️ 未检测到术语规则，跳过阶段二术语替换\n")

        engine_name_map = {
            "youdao": "有道",
            "baidu": "百度",
            "deeplx": "DeepLX",
            "siliconflow_translate": "硅流大模型"
        }

        translators_enabled = []
        if CONFIG["youdao"]["enabled"]: translators_enabled.append("youdao")
        if CONFIG["baidu"]["enabled"]: translators_enabled.append("baidu")
        if CONFIG["deeplx"]["enabled"]: translators_enabled.append("deeplx")
        if CONFIG["siliconflow_translate"]["enabled"]: translators_enabled.append("siliconflow_translate")

        log_callback(f"🌐 已启用翻译引擎: {len(translators_enabled)} 个\n")

        # ========== 阶段1：语音识别 ==========
        log_callback("\n【阶段1】语音识别\n")
        seg_list = recognize_speech(video_path, log_callback, stop_event)
        
        if stop_event.is_set():
            log_callback("⏹️ 用户停止\n")
            return
        if not seg_list:
            log_callback("⚠️ 未识别到任何语音内容\n")
            return
        log_callback(f"✅ 识别完成，共 {len(seg_list)} 条字幕\n")

        # ========== 阶段2：本地术语替换 ==========
        log_callback("\n【阶段2】本地术语匹配替换\n")
        total_replace = 0
        full_hit_count = 0
        processed_segs = []

        for idx, seg in enumerate(seg_list):
            if stop_event.is_set():
                break
            
            original_jp = seg["jp"]
            # 应用术语替换
            processed_jp, rep_count, _, is_full_hit = apply_term_replace(
                original_jp, term_list, forward_map, log_callback, idx+1
            )
            
            total_replace += rep_count
            if is_full_hit:
                full_hit_count += 1
            
            processed_segs.append({
                "start": seg["start"],
                "end": seg["end"],
                "jp_original": original_jp,
                "jp_processed": processed_jp,
                "is_full_hit": is_full_hit
            })

        log_callback(f"\n✅ 术语替换完成：共替换 {total_replace} 处术语，整句命中 {full_hit_count} 句\n")

        # ========== 阶段3：翻译 + 择优 ==========
        log_callback("\n【阶段3】翻译 + 智能择优融合\n")
        translator_map = {
            "youdao": translate_youdao,
            "deeplx": translate_deeplx,
            "baidu": translate_baidu,
            "siliconflow_translate": translate_siliconflow
        }

        if not translators_enabled:
            log_callback("⚠️ 未启用任何翻译引擎，仅输出日文原文\n")
            final_results = []
            for seg in processed_segs:
                final_results.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "jp": seg["jp_original"],
                    "cn": restore_placeholders(seg["jp_processed"], reverse_map),
                    "source": "原文",
                    "raw": {},
                    "statuses": {}
                })
        else:
            prev_cn = ""
            final_results = []

            for idx, seg in enumerate(processed_segs):
                if stop_event.is_set():
                    break
                
                # 整句命中的直接用结果，跳过翻译
                if seg["is_full_hit"]:
                    final_cn = seg["jp_processed"]
                    final_results.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "jp": seg["jp_original"],
                        "cn": final_cn,
                        "source": "本地术语直出",
                        "raw": {name: final_cn for name in translators_enabled},
                        "statuses": {name: "整句命中" for name in translators_enabled}
                    })
                    prev_cn = final_cn
                    continue

                jp_for_trans = seg["jp_processed"]
                results = {}
                statuses = {}

                with ThreadPoolExecutor(max_workers=len(translators_enabled)) as executor:
                    futures = {}
                    for name in translators_enabled:
                        if name == "siliconflow_translate":
                            futures[executor.submit(translate_siliconflow, jp_for_trans, context, prev_cn)] = name
                        else:
                            futures[executor.submit(translator_map[name], jp_for_trans)] = name
                    
                    for future in as_completed(futures):
                        if stop_event.is_set():
                            break
                        name = futures[future]
                        try:
                            text, status = future.result()
                            # 还原占位符
                            if text:
                                text = restore_placeholders(text, reverse_map)
                            results[name] = text
                            statuses[name] = status
                        except Exception as e:
                            results[name] = None
                            statuses[name] = "异常"

                best, source, statuses = merge_translation(seg["jp_original"], results, statuses)
                final_results.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "jp": seg["jp_original"],
                    "cn": best,
                    "source": source,
                    "raw": results,
                    "statuses": statuses
                })

                if CONFIG["params"].get("enable_context", True):
                    prev_cn = best

                # 逐句打印状态
                status_parts = []
                for name in translators_enabled:
                    display_name = engine_name_map.get(name, name)
                    if results[name]:
                        status_parts.append(f"{display_name}:✅")
                    else:
                        err = statuses.get(name, "失败")
                        status_parts.append(f"{display_name}:❌({err})")
                status_line = " | ".join(status_parts)
                log_callback(f"翻译进度: {len(final_results)}/{len(processed_segs)} | {status_line}\n")

            final_results.sort(key=lambda x: x["start"])

        if stop_event.is_set():
            log_callback("⏹️ 用户停止\n")

        # 预览前5条
        log_callback("\n翻译结果预览（前5条）：\n")
        for i, res in enumerate(final_results[:5]):
            log_callback(f"{i+1}. 【日】{res['jp']}\n   【中】{res['cn']}\n   来源:{res['source']}\n\n")

        # 生成SRT
        out_dir = os.path.dirname(video_path)
        filename = os.path.splitext(os.path.basename(video_path))[0]
        suffix = "_部分" if stop_event.is_set() else ""
        srt_path = os.path.join(out_dir, f"{filename}{suffix}_字幕.srt")
        log_callback("【阶段4】生成 SRT 字幕文件...\n")
        with open(srt_path, "w", encoding="utf-8") as f:
            for idx, res in enumerate(final_results, start=1):
                f.write(f"{idx}\n")
                f.write(f"{format_srt_time(res['start'])} --> {format_srt_time(res['end'])}\n")
                f.write(f"{res['cn']}\n\n")

        # 汇总统计
        if translators_enabled:
            stats = {}
            for name in translators_enabled:
                stats[name] = {"total": 0, "success": 0, "fail_reasons": {}}
            
            for res in final_results:
                for name in translators_enabled:
                    stats[name]["total"] += 1
                    if res["raw"].get(name):
                        stats[name]["success"] += 1
                    else:
                        reason = res["statuses"].get(name, "未知")
                        if reason != "整句命中":  # 排除整句命中的情况
                            stats[name]["fail_reasons"][reason] = stats[name]["fail_reasons"].get(reason, 0) + 1
            
            log_callback("\n📊 翻译引擎汇总统计\n")
            for name in translators_enabled:
                display_name = engine_name_map.get(name, name)
                s = stats[name]
                total = s["total"]
                success = s["success"]
                rate = success / total * 100 if total else 0
                log_callback(f"  {display_name}: {success}/{total} 成功率 {rate:.1f}%\n")
                if s["fail_reasons"]:
                    top_reasons = sorted(s["fail_reasons"].items(), key=lambda x: -x[1])[:2]
                    for reason, cnt in top_reasons:
                        if cnt > 0:
                            log_callback(f"    主要失败原因: {reason} ({cnt}次)\n")

        log_callback(f"\n🎉 全部完成！字幕文件：{srt_path}\n")
    except Exception as e:
        log_callback(f"❌ 处理异常: {e}\n")
    finally:
        finish_callback()

# ---------------------------- 一键部署环境 ----------------------------
def install_dependencies(log_callback):
    log_callback("🔧 部署环境...\n")
    log_callback(f"安装依赖: {', '.join(REQUIRED_PACKAGES)}\n")
    log_callback(f"镜像源: {PIP_MIRROR}\n")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], capture_output=True, timeout=60)
        log_callback("pip 升级完成\n")
    except Exception as e:
        log_callback(f"⚠️ pip 升级失败: {e}\n")
    cmd = [sys.executable, "-m", "pip", "install", "-i", PIP_MIRROR] + REQUIRED_PACKAGES
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log_callback(line)
        proc.wait()
        if proc.returncode == 0:
            log_callback("✅ 部署成功\n")
        else:
            log_callback(f"❌ 部署失败，错误代码 {proc.returncode}\n")
            log_callback(f"💡 手动安装命令: pip install -i {PIP_MIRROR} " + " ".join(REQUIRED_PACKAGES) + "\n")
    except Exception as e:
        log_callback(f"❌ 异常: {e}\n")

# ---------------------------- GUI 界面 ----------------------------
class SubtitleApp:
    def __init__(self, root):
        self.root = root
        root.title("日语视频自动字幕生成工具 (独立术语替换版)")
        root.geometry("850x750")

        self.log_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 10))
        self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        
        self.select_btn = tk.Button(btn_frame, text="📂 选择视频", command=self.on_select,
                                    width=20, height=2, bg="#4CAF50", fg="white", font=("Arial", 10))
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="⏹️ 停止", command=self.on_stop,
                                  width=10, height=2, bg="#f44336", fg="white", font=("Arial", 10),
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.settings_btn = tk.Button(btn_frame, text="⚙️ 设置", command=self.open_settings,
                                      width=10, height=2, bg="#2196F3", fg="white", font=("Arial", 10))
        self.settings_btn.pack(side=tk.LEFT, padx=5)
        
        self.deploy_btn = tk.Button(btn_frame, text="🔧 部署环境", command=self.on_deploy,
                                    width=12, height=2, bg="#FF9800", fg="white", font=("Arial", 10))
        self.deploy_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = tk.Button(btn_frame, text="清空日志", command=self.clear_log,
                                   width=10, height=2, bg="#9E9E9E", fg="white", font=("Arial", 10))
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(root, text="引擎: 加载中", font=("Arial", 9), fg="gray")
        self.status_label.pack(pady=2)
        
        self.processing = False
        self.stop_event = threading.Event()
        self.log_queue = queue.Queue()
        self.root.after(100, self.process_log_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_status()
        self.video_context = ""

    def update_status(self):
        primary = CONFIG["speech_engine"]["primary"]
        fallback = "启用" if CONFIG["speech_engine"]["fallback_enabled"] else "禁用"
        self.status_label.config(text=f"主引擎: {primary} | 备用: {fallback}")

    def log(self, text):
        self.log_queue.put(text)

    def process_log_queue(self):
        while not self.log_queue.empty():
            text = self.log_queue.get()
            if text == "__FINISH__":
                self.on_process_finish()
            else:
                self.log_text.insert(tk.END, text)
                self.log_text.see(tk.END)
        self.root.after(100, self.process_log_queue)

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)

    def open_background_window(self):
        win = tk.Toplevel(self.root)
        win.title("填写翻译术语背景（提升翻译准确度）")
        win.geometry("580x620")
        win.transient(self.root)
        win.grab_set()

        tk.Label(win, text="动画/作品名称：", font=("Arial", 10, "bold")).pack(anchor="w", padx=15, pady=(10, 2))
        anime_entry = ttk.Entry(win)
        anime_entry.insert(0, "间谍过家家")
        anime_entry.pack(fill="x", padx=15, pady=2)

        tk.Label(win, text="角色名对照表（日文→中文，每行一个）：", font=("Arial", 10, "bold")).pack(anchor="w", padx=15, pady=(10, 2))
        char_text = tk.Text(win, height=6, wrap="word")
        char_text.insert("1.0", """ロイド・フォージャー → 劳埃德·福杰（黄昏）
ヨル・フォージャー → 约尔·福杰（荆棘公主）
アーニャ → 阿尼亚""")
        char_text.pack(fill="x", padx=15, pady=2)

        tk.Label(win, text="特殊术语对照表（日文→中文，每行一个）：", font=("Arial", 10, "bold")).pack(anchor="w", padx=15, pady=(10, 2))
        term_text = tk.Text(win, height=4, wrap="word")
        term_text.insert("1.0", """西国 → 西国
東国 → 东国""")
        term_text.pack(fill="x", padx=15, pady=2)

        tk.Label(win, text="固定短句/翻译偏好（日文→中文，每行一个）：", font=("Arial", 10, "bold")).pack(anchor="w", padx=15, pady=(10, 2))
        pref_text = tk.Text(win, height=4, wrap="word")
        pref_text.insert("1.0", """おはよう → 早安
おやすみ → 晚安""")
        pref_text.pack(fill="x", padx=15, pady=2)

        tk.Label(win, text="💡 所有带 → 的行都会在阶段二本地强制替换，所有翻译引擎生效。", fg="gray", font=("Arial", 9)).pack(anchor="w", padx=15, pady=(5, 0))

        def on_ok():
            anime = anime_entry.get().strip()
            chars = char_text.get("1.0", tk.END).strip()
            terms = term_text.get("1.0", tk.END).strip()
            prefs = pref_text.get("1.0", tk.END).strip()

            context_parts = []
            if anime:
                context_parts.append(f"作品：{anime}")
            if chars:
                context_parts.append(f"角色名对照表：\n{chars}")
            if terms:
                context_parts.append(f"特殊术语对照表：\n{terms}")
            if prefs:
                context_parts.append(f"固定短句：\n{prefs}")
            
            self.video_context = "\n\n".join(context_parts)
            win.destroy()

        def on_cancel():
            self.video_context = ""
            win.destroy()

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="确认使用", command=on_ok, width=15).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="跳过", command=on_cancel, width=10).pack(side=tk.LEFT, padx=10)

        self.root.wait_window(win)

    def on_select(self):
        if self.processing:
            return
        file_path = filedialog.askopenfilename(
            title="选择日语视频/音频文件",
            filetypes=[
                ("视频音频", "*.mkv *.mp4 *.avi *.mov *.flv *.m4v *.wmv *.mp3 *.wav *.m4a *.flac"),
                ("所有文件", "*.*")
            ]
        )
        if not file_path:
            return

        self.video_context = ""
        if messagebox.askyesno("翻译优化", "是否填写角色名/术语表来提升翻译准确度？"):
            self.open_background_window()

        self.stop_event.clear()
        self.select_btn.config(state=tk.DISABLED, text="⏳ 处理中...")
        self.stop_btn.config(state=tk.NORMAL)
        self.settings_btn.config(state=tk.DISABLED)
        self.deploy_btn.config(state=tk.DISABLED)
        self.log("开始处理...\n")
        self.processing = True
        thread = threading.Thread(target=self.run_process, args=(file_path, self.video_context), daemon=True)
        thread.start()

    def on_stop(self):
        if self.processing:
            self.stop_event.set()
            self.stop_btn.config(state=tk.DISABLED, text="⏹️ 停止中...")
            self.log("⏹️ 停止请求已发送，正在安全中断...\n")

    def on_deploy(self):
        if self.processing:
            messagebox.showinfo("提示", "请先停止当前处理")
            return
        if not messagebox.askyesno("确认部署", f"将安装以下依赖包：\n{', '.join(REQUIRED_PACKAGES)}\n\n确认继续？"):
            return
        thread = threading.Thread(target=self.run_deploy, daemon=True)
        thread.start()

    def run_deploy(self):
        self.deploy_btn.config(state=tk.DISABLED, text="⏳ 安装中...")
        self.log("🔧 开始部署环境...\n")
        install_dependencies(self.log)
        self.root.after(0, lambda: self.deploy_btn.config(state=tk.NORMAL, text="🔧 部署环境"))

    def run_process(self, video_path, context):
        def log_callback(text):
            self.log(text)
        def finish_callback():
            self.log("__FINISH__")
        process_video(video_path, log_callback, finish_callback, self.stop_event, context)

    def on_process_finish(self):
        self.select_btn.config(state=tk.NORMAL, text="📂 选择视频")
        self.stop_btn.config(state=tk.DISABLED, text="⏹️ 停止")
        self.settings_btn.config(state=tk.NORMAL)
        self.deploy_btn.config(state=tk.NORMAL)
        self.processing = False
        self.log("\n✅ 处理结束。\n")

    def on_close(self):
        if self.processing:
            self.stop_event.set()
            time.sleep(0.5)
        self.root.destroy()

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("API 设置")
        win.geometry("650x750")
        win.transient(self.root)
        win.grab_set()
        temp = json.loads(json.dumps(CONFIG))

        notebook = ttk.Notebook(win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        api_info = {
            "youdao": {"label": "有道智云", "fields": ["appid", "secret"], "url": "https://ai.youdao.com/"},
            "baidu": {"label": "百度翻译", "fields": ["appid", "key"], "url": "https://fanyi-api.baidu.com/"},
            "deeplx": {"label": "DeepLX (需自建)", "fields": ["url"], "url": "https://github.com/OwO-Network/DeepLX"},
            "siliconflow_translate": {"label": "SiliconFlow 翻译", "fields": ["api_key", "model", "url"], "url": "https://siliconflow.cn/"},
            "siliconflow_stt": {"label": "SiliconFlow 语音", "fields": ["api_key", "model", "url"], "url": "https://siliconflow.cn/"}
        }
        entry_vars = {}
        check_vars = {}
        
        for key, info in api_info.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=info["label"])
            var = tk.BooleanVar(value=temp[key].get("enabled", False))
            check_vars[key] = var
            ttk.Checkbutton(frame, text="启用此 API", variable=var).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
            entry_vars[key] = {}
            row = 1
            for field in info["fields"]:
                label = field.replace("_", " ").title()
                ttk.Label(frame, text=f"{label}:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
                var_entry = tk.StringVar(value=temp[key].get(field, ""))
                entry_vars[key][field] = var_entry
                show = "*" if "secret" in field or "key" in field or "api_key" in field else None
                entry = ttk.Entry(frame, textvariable=var_entry, show=show)
                entry.grid(row=row, column=1, sticky="w", padx=5, pady=5, ipadx=100)
                row += 1
            ttk.Button(frame, text="🔗 访问官网", command=lambda url=info["url"]: webbrowser.open(url)).grid(row=row, column=0, columnspan=2, pady=5)

        # 语音引擎标签页
        speech_frame = ttk.Frame(notebook)
        notebook.add(speech_frame, text="🎤 语音引擎")
        row = 0
        
        ttk.Label(speech_frame, text="主引擎:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        primary_var = tk.StringVar(value=temp["speech_engine"]["primary"])
        ttk.Combobox(speech_frame, textvariable=primary_var,
                     values=["siliconflow_stt", "whisper"], 
                     state="readonly", width=25).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1
        
        ttk.Label(speech_frame, text="云端STT模型:", foreground="gray").grid(row=row, column=0, sticky="e", padx=5)
        stt_model_var = tk.StringVar(value=temp["siliconflow_stt"]["model"])
        ttk.Combobox(speech_frame, textvariable=stt_model_var,
                     values=["FunAudioLLM/SenseVoiceSmall", "TeleAI/TeleSpeechASR"],
                     state="readonly", width=30).grid(row=row, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(speech_frame, text="推荐日语用 SenseVoiceSmall", foreground="gray", font=("Arial", 8)).grid(row=row+1, column=1, sticky="w", padx=5)
        row += 2
        
        fallback_var = tk.BooleanVar(value=temp["speech_engine"]["fallback_enabled"])
        ttk.Checkbutton(speech_frame, text="启用备用引擎（主引擎失败自动切换）",
                        variable=fallback_var).grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        row += 1
        
        ttk.Separator(speech_frame).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        
        ttk.Label(speech_frame, text="Whisper 本地配置", font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", padx=5)
        row += 1
        
        w_enabled = tk.BooleanVar(value=temp["speech_engine"]["whisper"]["enabled"])
        ttk.Checkbutton(speech_frame, text="启用 Whisper（本地运行，精度最高）", variable=w_enabled).grid(row=row, column=0, columnspan=2, sticky="w", padx=5)
        row += 1
        
        ttk.Label(speech_frame, text="设备:").grid(row=row, column=0, sticky="e", padx=5)
        w_device = tk.StringVar(value=temp["speech_engine"]["whisper"]["device"])
        ttk.Combobox(speech_frame, textvariable=w_device,
                     values=["cuda", "cpu"], state="readonly", width=10).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        ttk.Label(speech_frame, text="模型大小:").grid(row=row, column=0, sticky="e", padx=5)
        w_model = tk.StringVar(value=temp["speech_engine"]["whisper"]["model_size"])
        ttk.Combobox(speech_frame, textvariable=w_model,
                     values=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
                     state="readonly", width=15).grid(row=row, column=1, sticky="w", padx=5)

        def save():
            for key in check_vars:
                temp[key]["enabled"] = check_vars[key].get()
                for field, var in entry_vars[key].items():
                    temp[key][field] = var.get().strip()
            temp["siliconflow_stt"]["model"] = stt_model_var.get().strip()
            temp["speech_engine"]["primary"] = primary_var.get()
            temp["speech_engine"]["fallback_enabled"] = fallback_var.get()
            temp["speech_engine"]["whisper"]["enabled"] = w_enabled.get()
            temp["speech_engine"]["whisper"]["device"] = w_device.get()
            temp["speech_engine"]["whisper"]["model_size"] = w_model.get()
            save_config(temp)
            global CONFIG
            CONFIG = temp
            self.update_status()
            messagebox.showinfo("成功", "设置已保存")
            win.destroy()
        
        def cancel():
            win.destroy()
        
        btnf = ttk.Frame(win)
        btnf.pack(pady=10)
        ttk.Button(btnf, text="保存", command=save).pack(side=tk.LEFT, padx=10)
        ttk.Button(btnf, text="取消", command=cancel).pack(side=tk.LEFT, padx=10)

# ---------------------------- 主入口 ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SubtitleApp(root)
    root.mainloop()
