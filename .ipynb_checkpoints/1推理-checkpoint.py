import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from model import W2VAASIST  # 确保从你的模型文件导入

# 音频预处理函数
def preprocess_audio(audio_path, max_len=64600, feat_layer=5):
    """
    音频预处理流程（与训练时保持一致）
    :param audio_path: 音频文件路径
    :param max_len: 最大音频长度（与训练参数一致）
    :param feat_layer: 使用的Wav2Vec2隐藏层（5对应xls-r-5）
    :return: 预处理后的特征张量
    """
    # 加载音频文件（自动重采样到16kHz）
    waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
    waveform = torch.from_numpy(waveform).float()
    
    # 截断或填充到固定长度
    if len(waveform) > max_len:
        waveform = waveform[:max_len]
    else:
        repeats = (max_len // len(waveform)) + 1
        waveform = waveform.repeat(repeats)[:max_len]
    
    # 初始化特征提取器和模型（使用与训练时相同的配置）
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").eval()
    
    # 提取特征
    inputs = feature_extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        features = outputs.hidden_states[feat_layer].squeeze(0)  # 取第5层特征
    
    # 调整形状为 [1, 1, 128, 201]（与训练输入一致）
    features = features.transpose(0, 1).unsqueeze(0).unsqueeze(0)
    return features

# 预测函数
def predict(audio_path, model_path='./models/try/anti-spoofing_feat_model.pt'):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device).to(device)
    model.eval()
    
    # 预处理音频
    try:
        features = preprocess_audio(audio_path).to(device)
    except Exception as e:
        return f"预处理失败: {str(e)}"
    
    # 模型推理
    with torch.no_grad():
        _, outputs = model(features)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    # 解析结果
    real_prob = probs[0]
    fake_prob = probs[1]
    prediction = "Real" if real_prob > fake_prob else "Fake"
    
    return {
        "real_probability": float(f"{real_prob:.4f}"),
        "fake_probability": float(f"{fake_prob:.4f}"),
        "prediction": prediction
    }

# 使用示例
if __name__ == "__main__":
    # 配置参数
    audio_file = "chun.wav"  # 替换为你的测试音频路径
    model_path = "./models/try/anti-spoofing_feat_model.pt"  # 模型路径
    
    # 执行预测
    result = predict(audio_file, model_path)
    
    # 输出结果
    print("=" * 40)
    print(f"音频文件: {audio_file}")
    print(f"真实概率: {result['real_probability']}")
    print(f"伪造概率: {result['fake_probability']}")
    print(f"最终判断: {result['prediction']}")
    print("=" * 40)