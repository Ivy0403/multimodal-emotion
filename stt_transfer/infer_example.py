import os
import numpy as np
import torch
from funasr import AutoModel
from loguru import logger
import librosa.display

class Emotion2vecPredict(object):
    def __init__(self):
        """
        :param audio: 音频文件路径
        """
        self.audio_file_path =None

        # 固定的模型文件目录和GPU设置
        self.model_dir = 'models/iic/emotion2vec_plus_base'  # 模型文件目录
        self.use_gpu = True  # 是否使用GPU进行推理

        # 检查模型目录是否存在
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"模型目录不存在：{self.model_dir}")

        # 加载模型
        self.model = AutoModel(model=self.model_dir,
                               log_level="ERROR",
                               device='cuda' if self.use_gpu else 'cpu',
                               disable_pbar=True,
                               disable_log=True,
                               disable_update=True)

        logger.info(f"成功加载本地模型：{self.model_dir}")

    def extract_features(self, audio_file):
        """
        提取音频特征（MFCC），并确保特征维度一致
        :param audio_file: 音频文件路径
        :return: 音频数据、采样率和提取的 MFCC 特征
        """
        # 读取音频文件
        X, sample_rate = librosa.load(audio_file, res_type='kaiser_fast',
                                      duration=2.5, sr=22050 * 2, offset=0.5)

        # 提取 MFCC 特征
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)

        # 如果 MFCC 特征长度不足 216，进行零填充
        if len(mfccs) < 216:
            mfccs = np.pad(mfccs, (0, 216 - len(mfccs)), 'constant')
        elif len(mfccs) > 216:
            mfccs = mfccs[:216]  # 如果超过 216，裁剪到 216

        return X, sample_rate, mfccs

    def predict(self, audio_path):
        """
        对输入的音频进行情感预测
        """
        # 读取音频并进行预测
        res = self.model.generate(audio_path, granularity="utterance", extract_embedding=False)

        # 初始化字典，用来存放每个标签和其对应的概率
        result_dict = {}

        # 遍历预测结果
        for result in res:
            label, score = result["labels"], result["scores"]

            # 遍历每个标签及其对应的概率
            for l, s in zip(label, score):
                # 提取英文标签，去除"/"后面的部分
                l = l.split("/")[1] if "/" in l else l  # 确保只取英文标签部分
                result_dict[l] = round(float(s), 2)  # 保留两位小数并存入字典

        # 打印最终结果
        print(f"预测结果: {result_dict}")
        return result_dict


if __name__ == "__main__":
    # 音频路径：指定本地音频文件路径
    audio_file_path = 'test.m4a'  # 替换为你的音频文件路径

    # 创建Emotion2vecPredict实例，并进行情感预测
    predictor = Emotion2vecPredict()
    predictor.predict(audio_file_path)
