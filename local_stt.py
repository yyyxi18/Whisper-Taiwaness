#!/usr/bin/env python3
"""
本地台語語音辨識模組
可以直接在Python中使用，無需啟動HTTP服務
"""

import os
import tempfile
import logging
from transformers import pipeline
import torch
import soundfile as sf
import numpy as np

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalSTT:
    """本地台語語音辨識類"""
    
    def __init__(self, model_path=None):
        """
        初始化STT模型
        
        Args:
            model_path: 模型路徑，如果為None則使用Hugging Face模型
        """
        self.model = None
        self.model_path = model_path
        self.device = None
        self.load_model()
    
    def load_model(self):
        """載入Whisper模型"""
        try:
            logger.info("正在載入Whisper-Taiwanese模型...")
            
            # 檢查是否有本地模型文件
            if self.model_path and os.path.exists(self.model_path):
                model_path = self.model_path
                logger.info(f"使用本地模型: {model_path}")
            else:
                # 如果沒有本地模型，使用Hugging Face模型
                model_path = "NUTN-KWS/Whisper-Taiwanese-model-v0.5"
                logger.info(f"使用Hugging Face模型: {model_path}")
            
            # 優化GPU配置
            if torch.cuda.is_available():
                # 檢查GPU內存
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"檢測到GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU內存: {gpu_memory:.1f} GB")
                
                # 根據GPU內存選擇最佳配置
                if gpu_memory >= 8:  # 8GB以上使用float16
                    self.device = 0
                    torch_dtype = torch.float16
                    logger.info("使用GPU + float16 (最佳性能)")
                elif gpu_memory >= 4:  # 4-8GB使用float16但啟用內存優化
                    self.device = 0
                    torch_dtype = torch.float16
                    logger.info("使用GPU + float16 (內存優化)")
                    # 啟用內存優化
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                else:  # 4GB以下使用float32
                    self.device = 0
                    torch_dtype = torch.float32
                    logger.info("使用GPU + float32 (內存節省)")
            else:
                self.device = -1
                torch_dtype = torch.float32
                logger.info("使用CPU")
            
            # 設置CUDA內存分配策略
            if self.device >= 0:
                # 啟用CUDA內存優化
                torch.cuda.empty_cache()
                # 設置內存分配策略
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                logger.info("已啟用CUDA內存優化")
            
            self.model = pipeline(
                "automatic-speech-recognition",
                model=model_path,
                device=self.device,
                torch_dtype=torch_dtype,
                model_kwargs={
                    "low_cpu_mem_usage": True,
                    "attn_implementation": "flash_attention_2" if self.device >= 0 and self._check_flash_attention() else "eager"
                }
            )
            
            # 如果使用GPU，將模型移到GPU並優化
            if self.device >= 0:
                # 確保模型在GPU上
                if hasattr(self.model.model, 'to'):
                    self.model.model = self.model.model.to(f'cuda:{self.device}')
                # 啟用混合精度推理
                if torch_dtype == torch.float16:
                    self.model.model = self.model.model.half()
                logger.info("模型已優化並移至GPU")
            
            logger.info("模型載入完成！")
            return True
            
        except Exception as e:
            logger.error(f"模型載入失敗: {str(e)}")
            return False
    
    def _check_flash_attention(self):
        """檢查Flash Attention是否可用"""
        try:
            import flash_attn
            logger.info("Flash Attention 2 可用，啟用性能優化")
            return True
        except ImportError:
            logger.info("Flash Attention 2 未安裝，使用標準attention")
            return False
    
    def check_ffmpeg(self):
        """檢查ffmpeg是否可用"""
        try:
            import subprocess
            import os
            
            # 首先檢查系統PATH中的ffmpeg
            try:
                result = subprocess.run(['ffmpeg', '-version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logger.info("找到系統PATH中的ffmpeg")
                    return True
            except:
                pass
            
            # 檢查本地ffmpeg_bin文件夾
            local_ffmpeg_path = os.path.join(os.path.dirname(__file__), 'ffmpeg_bin', 'ffmpeg.exe')
            if os.path.exists(local_ffmpeg_path):
                try:
                    result = subprocess.run([local_ffmpeg_path, '-version'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        logger.info("找到本地ffmpeg_bin中的ffmpeg")
                        # 將本地ffmpeg路徑添加到環境變量
                        os.environ['PATH'] = os.path.join(os.path.dirname(__file__), 'ffmpeg_bin') + os.pathsep + os.environ.get('PATH', '')
                        return True
                except:
                    pass
            
            # 檢查當前目錄下的ffmpeg_bin
            current_ffmpeg_path = os.path.join('ffmpeg_bin', 'ffmpeg.exe')
            if os.path.exists(current_ffmpeg_path):
                try:
                    result = subprocess.run([current_ffmpeg_path, '-version'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        logger.info("找到當前目錄下的ffmpeg_bin中的ffmpeg")
                        # 將ffmpeg_bin路徑添加到環境變量
                        os.environ['PATH'] = os.path.abspath('ffmpeg_bin') + os.pathsep + os.environ.get('PATH', '')
                        return True
                except:
                    pass
            
            logger.warning("未找到ffmpeg，將使用替代方法處理音頻")
            return False
            
        except Exception as e:
            logger.warning(f"檢查ffmpeg時發生錯誤: {e}")
            return False
    
    def preprocess_audio(self, audio_file_path):
        """預處理音頻文件"""
        try:
            # 檢查ffmpeg是否可用
            if not self.check_ffmpeg():
                logger.warning("ffmpeg未安裝，嘗試使用替代方法...")
                return self.preprocess_audio_alternative(audio_file_path)
            
            # 使用librosa讀取音頻（需要ffmpeg）
            try:
                import librosa
                audio, sr = librosa.load(audio_file_path, sr=16000)
                
                # 確保是單聲道
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # 標準化音量
                audio = librosa.util.normalize(audio)
                
                # 保存為臨時文件
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                sf.write(temp_file.name, audio, sr)
                
                return temp_file.name, sr
                
            except ImportError:
                logger.warning("librosa未安裝，使用替代方法...")
                return self.preprocess_audio_alternative(audio_file_path)
                
        except Exception as e:
            logger.error(f"音頻預處理失敗: {str(e)}")
            raise e
    
    def preprocess_audio_alternative(self, audio_file_path):
        """替代的音頻預處理方法（不依賴ffmpeg）"""
        try:
            # 直接使用soundfile讀取（僅支持有限格式）
            audio, sr = sf.read(audio_file_path)
            
            # 轉換為float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # 確保是單聲道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # 標準化音量
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # 重採樣到16kHz（如果需要）
            if sr != 16000:
                logger.warning(f"音頻採樣率 {sr}Hz 不是16kHz，可能影響辨識效果")
            
            # 保存為臨時文件
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, audio, sr)
            
            return temp_file.name, sr
            
        except Exception as e:
            logger.error(f"替代音頻預處理失敗: {str(e)}")
            raise e
    
    def transcribe_file(self, audio_file_path):
        """
        辨識音頻文件
        
        Args:
            audio_file_path: 音頻文件路徑
            
        Returns:
            dict: 包含辨識結果的字典
        """
        if self.model is None:
            return {"error": "模型尚未載入"}
        
        try:
            # 檢查文件是否存在
            if not os.path.exists(audio_file_path):
                return {"error": f"音頻文件不存在: {audio_file_path}"}
            
            # 檢查文件類型
            supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
            file_ext = os.path.splitext(audio_file_path.lower())[1]
            
            if file_ext not in supported_formats:
                return {"error": f"不支援的音頻格式: {file_ext}，請使用 {', '.join(supported_formats)}"}
            
            # 檢查ffmpeg依賴
            if file_ext in {'.mp3', '.m4a', '.ogg'} and not self.check_ffmpeg():
                logger.warning(f"處理 {file_ext} 格式需要ffmpeg，建議安裝ffmpeg以獲得最佳支持")
            
            # 預處理音頻
            temp_file_path, sample_rate = self.preprocess_audio(audio_file_path)
            
            try:
                # 執行語音辨識
                logger.info("開始語音辨識...")
                
                # 如果使用GPU，清理GPU內存
                if self.device >= 0:
                    torch.cuda.empty_cache()
                
                result = self.model(
                    temp_file_path,
                    generate_kwargs={
                        "language": "zh",  # 中文
                        "task": "transcribe"
                    }
                )
                
                transcription = result["text"]
                logger.info(f"辨識結果: {transcription}")
                
                return {
                    'success': True,
                    'transcription': transcription,
                    'language': 'zh',
                    'sample_rate': sample_rate,
                    'file_path': audio_file_path
                }
                
            finally:
                # 清理臨時文件
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"語音辨識失敗: {str(e)}")
            return {"error": f"語音辨識失敗: {str(e)}"}
    
    def transcribe_url(self, audio_url):
        """
        通過URL進行語音辨識
        
        Args:
            audio_url: 音頻文件URL
            
        Returns:
            dict: 包含辨識結果的字典
        """
        if self.model is None:
            return {"error": "模型尚未載入"}
        
        try:
            # 下載音頻文件
            import requests
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()
            
            # 保存為臨時文件
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.close()
            
            try:
                # 預處理音頻
                temp_file_path, sample_rate = self.preprocess_audio(temp_file.name)
                
                # 執行語音辨識
                logger.info("開始語音辨識...")
                
                # 如果使用GPU，清理GPU內存
                if self.device >= 0:
                    torch.cuda.empty_cache()
                
                result = self.model(
                    temp_file_path,
                    generate_kwargs={
                        "language": "zh",
                        "task": "transcribe"
                    }
                )
                
                transcription = result["text"]
                logger.info(f"辨識結果: {transcription}")
                
                return {
                    'success': True,
                    'transcription': transcription,
                    'language': 'zh',
                    'sample_rate': sample_rate,
                    'url': audio_url
                }
                
            finally:
                # 清理臨時文件
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"語音辨識失敗: {str(e)}")
            return {"error": f"語音辨識失敗: {str(e)}"}
    
    def get_model_info(self):
        """獲取模型信息"""
        if self.model is None:
            return {"error": "模型尚未載入"}
        
        device_info = "GPU" if self.device >= 0 else "CPU"
        if self.device >= 0:
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            device_info = f"GPU ({gpu_name}, {gpu_memory:.1f}GB)"
        
        return {
            'model_name': 'Whisper-Taiwanese-model-v0.5',
            'base_model': 'openai/whisper-large-v3-turbo',
            'language': 'Taiwanese (Taiwanese Hokkien)',
            'device': device_info,
            'sample_rate': 16000,
            'model_loaded': True,
            'ffmpeg_available': self.check_ffmpeg(),
            'gpu_optimized': self.device >= 0
        }
    
    def is_ready(self):
        """檢查模型是否準備就緒"""
        return self.model is not None
    
    def clear_gpu_memory(self):
        """清理GPU內存"""
        if self.device >= 0:
            torch.cuda.empty_cache()
            logger.info("GPU內存已清理")

# 全局實例
_stt_instance = None

def get_stt_instance(model_path=None):
    """獲取STT實例（單例模式）"""
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = LocalSTT(model_path)
    return _stt_instance

def transcribe_audio_file(audio_file_path, model_path=None):
    """
    快速函數：辨識音頻文件
    
    Args:
        audio_file_path: 音頻文件路徑
        model_path: 模型路徑（可選）
        
    Returns:
        dict: 辨識結果
    """
    stt = get_stt_instance(model_path)
    return stt.transcribe_file(audio_file_path)

def transcribe_audio_url(audio_url, model_path=None):
    """
    快速函數：通過URL辨識音頻
    
    Args:
        audio_url: 音頻文件URL
        model_path: 模型路徑（可選）
        
    Returns:
        dict: 辨識結果
    """
    stt = get_stt_instance(model_path)
    return stt.transcribe_audio_url(audio_url)
