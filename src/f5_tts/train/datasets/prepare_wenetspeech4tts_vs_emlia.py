# 兼容emlia的vocab
# generate audio text map for WenetSpeech4TTS
# evaluate for vocab size

import os
import sys


sys.path.append(os.getcwd())

import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files

import torchaudio
from datasets import Dataset
from tqdm import tqdm

from f5_tts.model.utils import convert_char_to_pinyin


# 修改 1: 增加 vocab_set 参数
def deal_with_sub_path_files(dataset_path, sub_path, vocab_set=None):
    print(f"Dealing with: {sub_path}")

    text_dir = os.path.join(dataset_path, sub_path, "txts")
    audio_dir = os.path.join(dataset_path, sub_path, "wavs")
    text_files = os.listdir(text_dir)

    audio_paths, texts, durations = [], [], []
    for text_file in tqdm(text_files):
        with open(os.path.join(text_dir, text_file), "r", encoding="utf-8") as file:
            first_line = file.readline().split("\t")
        audio_nm = first_line[0]
        audio_path = os.path.join(audio_dir, audio_nm + ".wav")
        text = first_line[1].strip()

        # 逻辑调整: 先处理文本，检查通过后再加入列表
        converted_text = ""
        if tokenizer == "pinyin":
            # 获取转换后的拼音字符串
            converted_list = convert_char_to_pinyin([text], polyphone=polyphone)
            converted_text = converted_list[0]
        elif tokenizer == "char":
            converted_text = text

        # 修改 2: 核心过滤逻辑
        # 如果提供了 vocab_set，且转换后的文本包含不在词表中的字符，则跳过
        if vocab_set is not None:
            # 检查 converted_text 的字符集合是否是 vocab_set 的子集
            if not set(converted_text).issubset(vocab_set):
                # 包含禁用字，直接跳过该样本
                continue

        # 检查通过，数据对齐加入
        audio_paths.append(audio_path)
        texts.append(converted_text)

        # 加载音频计算时长 (只有通过了检查才加载，节省IO)
        audio, sample_rate = torchaudio.load(audio_path)
        durations.append(audio.shape[-1] / sample_rate)

    return audio_paths, texts, durations


def main():
    assert tokenizer in ["pinyin", "char"]
    # 修改 3: 加载 Emilia vocab (如果在配置里定义了路径)
    vocab_set = None
    if emilia_vocab_path and os.path.exists(emilia_vocab_path):
        print(f"Loading Emilia vocab from: {emilia_vocab_path}")
        with open(emilia_vocab_path, "r", encoding="utf-8") as f:
            # 读取所有字符，去除换行符
            vocab_set = set(line.strip() for line in f if line.strip())
        vocab_set.add(" ") # 确保允许空格
        assert len(vocab_set) == 2545, f'print{len(vocab_set)} error len'
    else:
        print("Warning: Emilia vocab path not found or invalid, skipping filter.")
    audio_path_list, text_list, duration_list = [], [], []

    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []
    for dataset_path in dataset_paths:
        sub_items = os.listdir(dataset_path)
        sub_paths = [item for item in sub_items if os.path.isdir(os.path.join(dataset_path, item))]
        for sub_path in sub_paths:
            futures.append(executor.submit(deal_with_sub_path_files, dataset_path, sub_path, vocab_set))
    for future in tqdm(futures, total=len(futures)):
        audio_paths, texts, durations = future.result()
        audio_path_list.extend(audio_paths)
        text_list.extend(texts)
        duration_list.extend(durations)
    executor.shutdown()

    if not os.path.exists("data"):
        os.makedirs("data")

    print(f"\nSaving to {save_dir} ...")
    dataset = Dataset.from_dict({"audio_path": audio_path_list, "text": text_list, "duration": duration_list})
    dataset.save_to_disk(f"{save_dir}/raw", max_shard_size="2GB")  # arrow format

    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump(
            {"duration": duration_list}, f, ensure_ascii=False
        )  # dup a json separately saving duration in case for DynamicBatchSampler ease

    print("\nEvaluating vocab size (all characters and symbols / all phonemes) ...")
    text_vocab_set = set()
    for text in tqdm(text_list):
        text_vocab_set.update(list(text))

    # add alphabets and symbols (optional, if plan to ft on de/fr etc.)
    if tokenizer == "pinyin":
        text_vocab_set.update([chr(i) for i in range(32, 127)] + [chr(i) for i in range(192, 256)])

    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")
    print(f"\nFor {dataset_name}, sample count: {len(text_list)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}\n")


if __name__ == "__main__":
    max_workers = 32
    emilia_vocab_path = "/inspire/hdd/project/video-generation/chenxie-25019/hyr/F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt"
    tokenizer = "pinyin"  # "pinyin" | "char"
    polyphone = True
    dataset_choice = 1  # 1: Premium, 2: Standard, 3: Basic

    dataset_name = (
        ["WenetSpeech4TTS_Premium", "WenetSpeech4TTS_Standard", "WenetSpeech4TTS_Basic"][dataset_choice - 1]
        + "_"
        + tokenizer
    )
    dataset_paths = [
        "/inspire/hdd/global_public/public_datas/speech/wenetspeech4tts/Basic",
        "/inspire/hdd/global_public/public_datas/speech/wenetspeech4tts/Standard",
        "/inspire/hdd/global_public/public_datas/speech/wenetspeech4tts/Premium",
    ][-dataset_choice:]
    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nChoose Dataset: {dataset_name}, will save to {save_dir}\n")

    main()

    # Results (if adding alphabets with accents and symbols):
    # WenetSpeech4TTS       Basic     Standard     Premium
    # samples count       3932473      1941220      407494
    # pinyin vocab size      1349         1348        1344   (no polyphone)
    #                           -            -        1459   (polyphone)
    # char   vocab size      5264         5219        5042

    # vocab size may be slightly different due to jieba tokenizer and pypinyin (e.g. way of polyphoneme)
    # please be careful if using pretrained model, make sure the vocab.txt is same
