import os
import numpy as np
import torch
from funasr import AutoModel
from loguru import logger
import re
from faster_whisper import WhisperModel
import joblib
import pandas as pd


class VideoAudioAnalyzer:
    def __init__(self):
        """
        初始化语音情感识别、语音转文字模型和文本情感分类器
        """
        # 初始化语音情感识别模型
        self.emotion_model = self._init_emotion_model()

        # 初始化语音转文字模型
        self.transcriber = self._init_transcriber()

        # 初始化文本情感分类器
        self.text_classifier = self._init_text_classifier()

    def _init_emotion_model(self):
        """
        初始化语音情感识别模型
        """
        model_dir = 'stt_transfer/models/iic/emotion2vec_plus_base'
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目录不存在：{model_dir}")

        try:
            model = AutoModel(model=model_dir,
                              log_level="ERROR",
                              device='cuda' if torch.cuda.is_available() else 'cpu',
                              disable_pbar=True,
                              disable_log=True,
                              disable_update=True)
            logger.info(f"成功加载本地情感识别模型：{model_dir}")
            return model
        except Exception as e:
            raise RuntimeError(f"情感识别模型初始化失败: {e}")

    def _init_transcriber(self):
        """
        初始化语音转文字模型
        """
        try:
            transcriber = WhisperModel(
                "tiny",
                device="cpu" if torch.cuda.is_available() else "cpu",
                compute_type="default",
                download_root="stt_transfer/models",
                local_files_only=True
            )
            logger.info("成功加载语音转文字模型")
            return transcriber
        except Exception as e:
            raise RuntimeError(f"语音转文字模型初始化失败: {e}")

    def _init_text_classifier(self):
        """
        初始化文本情感分类器
        """
        try:
            classifier_path = "stt_transfer/models/emotion_classifier_pipe_lr.pkl"
            if not os.path.exists(classifier_path):
                raise FileNotFoundError(f"文本情感分类器不存在：{classifier_path}")

            classifier = joblib.load(open(classifier_path, "rb"))
            logger.info("成功加载文本情感分类器")
            return classifier
        except Exception as e:
            raise RuntimeError(f"文本情感分类器初始化失败: {e}")

    def extract_audio_from_video(self, video_path):
        """
        从视频中提取音频
        :param video_path: 视频文件路径
        :return: 音频文件路径
        """
        try:
            import moviepy.editor as mp
            audio_path = os.path.splitext(video_path)[0] + ".wav"

            # 如果音频文件已存在，直接返回
            if os.path.exists(audio_path):
                return audio_path

            # 提取音频
            clip = mp.VideoFileClip(video_path)
            clip.audio.write_audiofile(audio_path)
            return audio_path
        except Exception as e:
            raise RuntimeError(f"从视频提取音频失败: {e}")

    def analyze_audio_emotion(self, audio_path):
        """
        分析音频情感并返回所有情感类别的置信度
        :param audio_path: 音频文件路径
        :return: 字典 {emotion1: confidence1, emotion2: confidence2, ...}
        """
        try:
            res = self.emotion_model.generate(audio_path, granularity="utterance", extract_embedding=False)
            result_dict = {}

            for result in res:
                label, score = result["labels"], result["scores"]
                for l, s in zip(label, score):
                    l = l.split("/")[1] if "/" in l else l
                    result_dict[l] = round(float(s), 2)

            logger.info(f"语音情感分析结果: {result_dict}")

            return result_dict
        except Exception as e:
            raise RuntimeError(f"语音情感分析失败: {e}")

    def analyze_text_emotion_from_audio(self, audio_path):
        """
        从音频转文字并分析文本情感
        :param audio_path: 音频文件路径
        :return: 字典 {emotion1: confidence1, emotion2: confidence2, ...}
        """
        try:
            # 1. 语音转文字
            segments, info = self.transcriber.transcribe(
                audio_path,
                beam_size=5,
                best_of=5,
                language=None
            )

            transcription = []
            for segment in segments:
                text = segment.text.strip().replace('&#39;', "'")
                text = re.sub(r'&#\d+;', '', text)

                if text and not re.match(r'^[，。、？‘’“”；：（｛｝【】）:;"\'\s \d`!@#$%^&*()_+=.,?/\\-]*$', text):
                    transcription.append(text)

            text = "\n".join(transcription)
            logger.info(f"语音转写结果: {text}")

            # 2. 文本情感分析
            probability = self.text_classifier.predict_proba([text])[0]
            proba_dict = dict(zip(self.text_classifier.classes_, probability))

            # 四舍五入保留4位小数
            proba_dict = {k: round(float(v), 4) for k, v in proba_dict.items()}

            logger.info(f"文本情感分析结果: {proba_dict}")

            return proba_dict
        except Exception as e:
            raise RuntimeError(f"文本情感分析失败: {e}")




    def analyze_video(self, video_path):
        """
        分析视频文件，返回所有情感类别的置信度
        :param video_path: 视频文件路径
        :return: 字典 {
            audio_emotion: {emotion1: confidence1, emotion2: confidence2, ...},
            text_emotion: {emotion1: confidence1, emotion2: confidence2, ...}
        }
        """
        try:
            # 1. 从视频提取音频
            audio_path = self.extract_audio_from_video(video_path)

            # 2. 并行执行语音情感分析和文本情感分析
            audio_emotion = self.analyze_audio_emotion(audio_path)
            text_emotion = self.analyze_text_emotion_from_audio(audio_path)

            # 3. 返回结果
            return {
                "audio_emotion": audio_emotion,
                "text_emotion": text_emotion
            }
        except Exception as e:
            logger.error(f"视频分析失败: {e}")
            raise


if __name__ == "__main__":
    # 使用示例
    try:
        analyzer = VideoAudioAnalyzer()
        video_path = "yfz168_1300547xhs.mp4"  # 替换为您的视频文件路径

        results = analyzer.analyze_video(video_path)

        print("\n语音情感分析结果:")
        for emotion, confidence in results["audio_emotion"].items():
            print(f"{emotion}: {confidence:.4f}")
        # {'angry': 0.47, 'disgusted': 0.0, 'fearful': 0.0, 'happy': 0.0, 'neutral': 0.52, 'other': 0.0, 'sad': 0.0, 'surprised': 0.0, '<unk>': 0.0}

        
        print("\n文本情感分析结果:")
        for emotion, confidence in results["text_emotion"].items():
            print(f"{emotion}: {confidence:.4f}")
    # {'anger': 0.0, 'disgust': 0.0, 'fear': 0.996, 'joy': 0.0001, 'neutral': 0.0, 'sadness': 0.0, 'shame': 0.0, 'surprise': 0.0039}
    except Exception as e:
        print(f"发生错误: {e}")