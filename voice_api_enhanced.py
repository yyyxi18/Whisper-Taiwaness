#!/usr/bin/env python3
"""
增強版語音輸入API服務
支持麥克風輸入、文件上傳和手機錄音等多種方式
支持手機和電腦訪問
"""

import os
import tempfile
import logging
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
from local_stt import LocalSTT
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import time
import socket

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 全局變量
stt_instance = None
recording = False
audio_queue = queue.Queue()
sample_rate = 16000

def get_local_ip():
    """獲取本機IP地址"""
    try:
        # 連接到外部地址來獲取本機IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🎤 台語語音辨識 - 多種輸入方式</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { 
            font-family: 'Microsoft JhengHei', Arial, sans-serif; 
            max-width: 1000px; 
            margin: 0 auto; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { 
            background: rgba(255, 255, 255, 0.95); 
            padding: 25px; 
            border-radius: 15px; 
            margin: 20px 0; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #4a5568;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            color: #718096;
            font-size: 1.1em;
        }
        .network-info {
            background: #e6fffa;
            border: 1px solid #81e6d9;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            text-align: center;
        }
        .input-methods {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .method-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            border: 2px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        .method-card:hover {
            border-color: #667eea;
            transform: translateY(-2px);
        }
        .method-card h3 {
            color: #4a5568;
            margin-top: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .button { 
            background: linear-gradient(45deg, #667eea, #764ba2); 
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer; 
            margin: 8px; 
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            width: 100%;
            max-width: 200px;
        }
        .button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .button:disabled { 
            background: #cbd5e0; 
            cursor: not-allowed;
            transform: none;
        }
        .record-btn { background: #dc3545; }
        .record-btn:hover { background: #c82333; }
        .result { 
            background: white; 
            padding: 20px; 
            border-radius: 15px; 
            margin: 15px 0; 
            border-left: 5px solid #48bb78;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            animation: slideIn 0.5s ease;
        }
        .result.error {
            border-left-color: #f56565;
        }
        .status { 
            padding: 15px; 
            border-radius: 10px; 
            margin: 15px 0; 
            text-align: center;
            font-weight: bold;
        }
        .recording { background: #fff3cd; border: 1px solid #ffeaa7; }
        .success { background: #d4edda; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; border: 1px solid #f5c6cb; }
        .info { background: #bee3f8; border: 1px solid #90cdf4; }
        .upload-area { 
            border: 3px dashed #667eea; 
            padding: 40px; 
            text-align: center; 
            border-radius: 20px; 
            margin: 20px 0; 
            background: rgba(102, 126, 234, 0.05);
            transition: all 0.3s ease;
        }
        .upload-area:hover { 
            border-color: #764ba2; 
            background: rgba(118, 75, 162, 0.1);
        }
        .upload-area.dragover {
            border-color: #764ba2;
            background: rgba(118, 75, 162, 0.15);
        }
        .tips {
            background: #e6fffa;
            border: 1px solid #81e6d9;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }
        .tips h4 {
            margin-top: 0;
            color: #2d3748;
        }
        .mobile-tips {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 768px) {
            .input-methods {
                grid-template-columns: 1fr;
            }
            .button {
                font-size: 14px;
                padding: 10px 20px;
            }
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎤 台語語音辨識</h1>
        <p>多種輸入方式的智能語音辨識工具</p>
    </div>
    
    <div class="network-info">
        <h3>🌐 網絡連接信息</h3>
        <p><strong>本機IP:</strong> <span id="localIP">正在獲取...</span></p>
        <p><strong>端口:</strong> 5000</p>
        <p><strong>手機訪問地址:</strong> <span id="mobileURL">正在獲取...</span></p>
        <p><small>請確保手機和電腦在同一個WiFi網絡下</small></p>
    </div>
    
    <div class="container">
        <h2>🔧 輸入方式選擇</h2>
        <div class="input-methods">
            <!-- 麥克風輸入 -->
            <div class="method-card">
                <h3>🎙️ 麥克風輸入</h3>
                <p>使用電腦麥克風進行實時語音輸入</p>
                <button id="testMic" class="button">🔍 測試麥克風</button>
                <div id="micStatus" class="status" style="display: none;"></div>
                <button id="startRecord" class="button record-btn">🎤 開始錄音</button>
                <button id="stopRecord" class="button" disabled>⏹️ 停止錄音</button>
                <div id="recordingStatus" class="status recording" style="display: none;">
                    🎵 正在錄音... 請說話
                </div>
            </div>
            
            <!-- 文件上傳 -->
            <div class="method-card">
                <h3>📁 文件上傳</h3>
                <p>上傳音頻文件進行辨識</p>
                <div class="upload-area" id="uploadArea">
                    <p>拖拽音頻文件到這裡，或點擊選擇文件</p>
                    <input type="file" id="audioFile" accept="audio/*" style="display: none;">
                    <button class="button" onclick="document.getElementById('audioFile').click()">選擇文件</button>
                </div>
            </div>
            
            <!-- 手機錄音 -->
            <div class="method-card">
                <h3>📱 手機錄音</h3>
                <p>用手機錄音後傳輸到電腦</p>
                <div class="tips">
                    <h4>📋 使用步驟：</h4>
                    <ol>
                        <li>用手機錄製語音</li>
                        <li>通過微信/QQ傳輸到電腦</li>
                        <li>或直接連接USB傳輸</li>
                        <li>拖拽到上方文件上傳區域</li>
                    </ol>
                </div>
            </div>
        </div>
    </div>

    <div class="mobile-tips">
        <h3>📱 手機用戶特別提示</h3>
        <p>如果您在手機上訪問此頁面：</p>
        <ul>
            <li>✅ 可以使用文件上傳功能</li>
            <li>✅ 可以拖拽音頻文件</li>
            <li>❌ 麥克風功能僅在電腦上可用</li>
            <li>💡 建議在電腦上使用麥克風功能</li>
        </ul>
    </div>

    <div class="container">
        <h2>🎵 支持的音頻格式</h2>
        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <span style="background: #667eea; color: white; padding: 8px 16px; border-radius: 20px;">WAV</span>
            <span style="background: #667eea; color: white; padding: 8px 16px; border-radius: 20px;">MP3</span>
            <span style="background: #667eea; color: white; padding: 8px 16px; border-radius: 20px;">M4A</span>
            <span style="background: #667eea; color: white; padding: 8px 16px; border-radius: 20px;">FLAC</span>
            <span style="background: #667eea; color: white; padding: 8px 16px; border-radius: 20px;">OGG</span>
        </div>
    </div>

    <div class="container">
        <h2>📋 辨識結果</h2>
        <div id="results"></div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;

        // 頁面加載時獲取網絡信息
        window.addEventListener('load', async () => {
            try {
                const response = await fetch('/network_info');
                const data = await response.json();
                document.getElementById('localIP').textContent = data.local_ip;
                document.getElementById('mobileURL').textContent = `http://${data.local_ip}:5000`;
            } catch (error) {
                console.error('獲取網絡信息失敗:', error);
            }
        });

        // 檢測是否為移動設備
        function isMobile() {
            return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        }

        // 如果是移動設備，隱藏麥克風相關功能
        if (isMobile()) {
            document.querySelector('.method-card:first-child').style.display = 'none';
        }

        // 測試麥克風
        document.getElementById('testMic').addEventListener('click', async () => {
            const statusDiv = document.getElementById('micStatus');
            try {
                // 檢查瀏覽器支持
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    throw new Error('瀏覽器不支持麥克風訪問');
                }
                
                // 檢查權限
                if (navigator.permissions) {
                    const permission = await navigator.permissions.query({ name: 'microphone' });
                    if (permission.state === 'denied') {
                        throw new Error('麥克風權限被拒絕');
                    }
                }
                
                // 測試麥克風訪問
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        sampleRate: 16000
                    } 
                });
                
                // 獲取音頻軌道信息
                const audioTrack = stream.getAudioTracks()[0];
                const settings = audioTrack.getSettings();
                
                statusDiv.innerHTML = `
                    ✅ 麥克風測試成功！<br>
                    📊 採樣率: ${settings.sampleRate || '未知'} Hz<br>
                    🎵 聲道數: ${settings.channelCount || '未知'}<br>
                    🔧 設備: ${audioTrack.label || '未知'}
                `;
                statusDiv.className = 'status success';
                statusDiv.style.display = 'block';
                
                // 停止測試流
                stream.getTracks().forEach(track => track.stop());
                
            } catch (error) {
                console.error('麥克風測試錯誤:', error);
                let errorMessage = '❌ 麥克風測試失敗: ';
                
                if (error.name === 'NotAllowedError') {
                    errorMessage += '權限被拒絕，請允許瀏覽器訪問麥克風';
                } else if (error.name === 'NotFoundError') {
                    errorMessage += '未找到麥克風設備，請檢查麥克風連接';
                } else if (error.name === 'NotSupportedError') {
                    errorMessage += '瀏覽器不支持麥克風功能';
                } else {
                    errorMessage += error.message;
                }
                
                statusDiv.innerHTML = errorMessage;
                statusDiv.className = 'status error';
                statusDiv.style.display = 'block';
            }
        });

        // 開始錄音
        document.getElementById('startRecord').addEventListener('click', async () => {
            try {
                // 檢查瀏覽器支持
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    throw new Error('瀏覽器不支持麥克風訪問');
                }
                
                // 獲取麥克風權限
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        sampleRate: 16000
                    } 
                });
                
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus'
                });
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    await uploadAudio(audioBlob, '麥克風錄音');
                    
                    // 停止所有音頻軌道
                    stream.getTracks().forEach(track => track.stop());
                };
                
                mediaRecorder.start();
                isRecording = true;
                document.getElementById('startRecord').disabled = true;
                document.getElementById('stopRecord').disabled = false;
                document.getElementById('recordingStatus').style.display = 'block';
                
            } catch (error) {
                console.error('錄音錯誤:', error);
                let errorMessage = '無法訪問麥克風: ';
                
                if (error.name === 'NotAllowedError') {
                    errorMessage += '權限被拒絕，請允許瀏覽器訪問麥克風';
                } else if (error.name === 'NotFoundError') {
                    errorMessage += '未找到麥克風設備，請檢查麥克風連接或使用文件上傳方式';
                } else if (error.name === 'NotSupportedError') {
                    errorMessage += '瀏覽器不支持麥克風功能';
                } else {
                    errorMessage += error.message;
                }
                
                alert(errorMessage);
            }
        });

        // 停止錄音
        document.getElementById('stopRecord').addEventListener('click', () => {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                document.getElementById('startRecord').disabled = false;
                document.getElementById('stopRecord').disabled = true;
                document.getElementById('recordingStatus').style.display = 'none';
            }
        });

        // 文件上傳
        document.getElementById('audioFile').addEventListener('change', async (event) => {
            const files = event.target.files;
            if (files.length > 0) {
                for (let file of files) {
                    await uploadAudio(file, file.name);
                }
            }
        });

        // 拖拽上傳
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', async (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                for (let file of files) {
                    await uploadAudio(file, file.name);
                }
            }
        });

        // 上傳音頻並辨識
        async function uploadAudio(audioBlob, filename) {
            const formData = new FormData();
            formData.append('audio', audioBlob, filename);
            
            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                displayResult(result, filename);
                
            } catch (error) {
                displayResult({ error: '上傳失敗: ' + error.message }, filename);
            }
        }

        // 顯示結果
        function displayResult(result, filename) {
            const resultsDiv = document.getElementById('results');
            const resultDiv = document.createElement('div');
            resultDiv.className = 'result';
            
            if (result.success) {
                resultDiv.innerHTML = `
                    <h3>✅ ${filename}</h3>
                    <p><strong>辨識結果:</strong> ${result.transcription}</p>
                    <p><small>處理時間: ${result.processing_time || '未知'} 秒</small></p>
                `;
            } else {
                resultDiv.innerHTML = `
                    <h3>❌ ${filename}</h3>
                    <p><strong>錯誤:</strong> ${result.error}</p>
                `;
            }
            
            resultsDiv.insertBefore(resultDiv, resultsDiv.firstChild);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """主頁面"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/network_info')
def network_info():
    """獲取網絡信息"""
    return jsonify({
        'local_ip': get_local_ip(),
        'port': 5000
    })

@app.route('/health')
def health():
    """健康檢查"""
    return jsonify({
        'status': 'healthy',
        'local_ip': get_local_ip(),
        'gpu_available': torch.cuda.is_available(),
        'model_loaded': stt_instance is not None and stt_instance.is_ready()
    })

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """語音辨識API端點"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': '沒有音頻文件'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': '沒有選擇文件'}), 400
        
        # 保存上傳的文件
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_file.save(temp_file.name)
        temp_file.close()
        
        try:
            # 記錄開始時間
            start_time = time.time()
            
            # 執行語音辨識
            result = stt_instance.transcribe_file(temp_file.name)
            
            # 計算處理時間
            processing_time = time.time() - start_time
            
            # 添加處理時間到結果
            if result.get('success'):
                result['processing_time'] = round(processing_time, 2)
                result['filename'] = audio_file.filename
            
            return jsonify(result)
            
        finally:
            # 清理臨時文件
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
                
    except Exception as e:
        logger.error(f"語音辨識失敗: {str(e)}")
        return jsonify({'error': f'語音辨識失敗: {str(e)}'}), 500

@app.route('/model_info')
def model_info():
    """獲取模型信息"""
    if stt_instance is None:
        return jsonify({'error': '模型尚未載入'}), 500
    
    return jsonify(stt_instance.get_model_info())

def init_stt():
    """初始化STT模型"""
    global stt_instance
    try:
        logger.info("正在初始化STT模型...")
        stt_instance = LocalSTT()
        if stt_instance.is_ready():
            logger.info("STT模型初始化成功！")
        else:
            logger.error("STT模型初始化失敗！")
    except Exception as e:
        logger.error(f"STT模型初始化失敗: {str(e)}")

if __name__ == '__main__':
    # 初始化STT模型
    init_stt()
    
    # 獲取本機IP
    local_ip = get_local_ip()
    
    # 啟動Flask服務
    host = "0.0.0.0"  # 綁定到所有網絡接口
    port = 5000
    
    logger.info(f"啟動增強版語音輸入API服務")
    logger.info(f"本機IP: {local_ip}")
    logger.info(f"本地訪問: http://localhost:{port}")
    logger.info(f"手機訪問: http://{local_ip}:{port}")
    logger.info(f"健康檢查: http://{local_ip}:{port}/health")
    
    app.run(host=host, port=port, debug=False)
