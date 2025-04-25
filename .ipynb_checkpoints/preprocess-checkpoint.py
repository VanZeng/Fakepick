import raw_dataset as dataset
from feature_extraction import *
import os
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model,Wav2Vec2FeatureExtractor,Wav2Vec2Config
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cuda = torch.cuda.is_available()
print('Cuda device available: ', cuda)

def pad_dataset(wav):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = 64600
    if waveform_len >= cut:
        waveform = waveform[:cut]
        return waveform
    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]
    return padded_waveform

# prprocess asvspoof2019LA

for part_ in ["train", "dev"]:
    codecspoof_raw = dataset.codecfake("./Codecfake", "./Codecfake/label/", part=part_)
    target_dir = os.path.join("./Codecfake/preprocess_xls-r-5", part_,
                              "xls-r-5")
    config = Wav2Vec2Config.from_json_file("facebook/wav2vec2-xls-r-300m/config.json")                          
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").cuda()
    model.config.output_hidden_states = True

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for idx in tqdm(range(len(codecspoof_raw))):
        waveform, filename, label = codecspoof_raw[idx]
        
        # ================ 断点续传核心逻辑 ================
        # 构造目标文件路径
        file_id = "%06d_%s_%s" % (idx, filename, label)
        target_path = os.path.join(target_dir, f"{file_id}.pt")
        
        # 检查文件是否已存在
        if os.path.exists(target_path):
            # print(f"跳过已存在文件: {file_id}")
            continue
        # ================ 断点续传逻辑结束 ================
        
        waveform = pad_dataset(waveform)
        
        input_values = processor(waveform, sampling_rate=16000,
                                    return_tensors="pt").input_values.cuda()
        with torch.no_grad():
            wav2vec2 = model(input_values).hidden_states[5].cpu()
        torch.save(wav2vec2.float(), target_path)

    print("Done")