# 🎯 台語語音辨識工具

基於 Whisper 的台語語音辨識工具，支持 GPU 加速和多種使用方式。

## ✨ 功能特點

- 🎵 **多種輸入方式**：麥克風輸入、文件上傳、手機錄音
- 🚀 **GPU 加速**：支持 CUDA，大幅提升處理速度
- 🌐 **Web API**：美觀的網頁界面，支持手機和電腦訪問
- 💻 **命令行工具**：簡單易用的 CLI 工具
- 📱 **響應式設計**：支持手機和電腦瀏覽器
- 🔧 **智能配置**：自動檢測 GPU 和依賴

## 🛠️ 安裝要求

- Python 3.8+
- CUDA (可選，用於 GPU 加速)
- 音頻處理庫

## 📦 安裝依賴

```bash
pip install -r requirements.txt
```

## 🚀 快速開始

### 1. 啟動 Web API 服務

```bash
# 啟動服務（支持手機和電腦訪問）
python start_api.py

# 僅本地訪問
python start_api.py --local

# 指定端口
python start_api.py --port 8080
```

### 2. 使用命令行工具

```bash
# 簡單模式（交互式）
python stt_cli.py

# 直接處理音頻文件
python stt_cli.py audio.wav

# 保存結果到文件
python stt_cli.py audio.wav -o result.txt

# 批量處理目錄
python stt_cli.py --batch audio_folder/ -o results/

# 顯示模型信息
python stt_cli.py --info
```

### 3. 使用統一啟動腳本

```bash
# 啟動 Web API（默認）
python start_api.py

# 啟動 CLI 工具
python start_api.py --cli

# 檢查依賴和模型
python start_api.py --check

# 詳細模式
python start_api.py --verbose
```

## 🌐 Web API 使用

### 訪問地址

- **本地訪問**: `http://localhost:5000`
- **網絡訪問**: `http://<本機IP>:5000`
- **健康檢查**: `http://localhost:5000/health`

### 功能說明

1. **🎙️ 麥克風輸入**：使用電腦內建麥克風進行實時語音輸入
2. **📁 文件上傳**：上傳音頻文件進行辨識
3. **📱 手機錄音**：用手機錄音後傳輸到電腦處理

### 支持的音頻格式

- WAV, MP3, M4A, FLAC, OGG

## 💻 命令行工具使用

### 簡單模式

直接運行 `python stt_cli.py`，會自動檢測當前目錄的音頻文件並提供交互式選擇。

### 進階模式

```bash
# 處理單個文件
python stt_cli.py audio.wav

# 批量處理
python stt_cli.py --batch audio_folder/ -o results/

# 詳細輸出
python stt_cli.py audio.wav -v

# 自定義模型路徑
python stt_cli.py audio.wav --model-path custom_model/
```

## 🔧 配置說明

### GPU 配置

工具會自動檢測 CUDA 可用性：

- 如果檢測到 GPU，會自動使用 GPU 加速
- 如果沒有 GPU，會使用 CPU 模式
- 支持 Flash Attention 2 優化（如果安裝了相關包）

### 網絡配置

- 默認綁定到 `0.0.0.0:5000`，支持外部訪問
- 使用 `--local` 參數僅綁定到 `127.0.0.1:5000`
- 支持自定義主機地址和端口

## 📁 項目結構

```
Whisper-Taiwaness/
├── voice_api_enhanced.py    # Web API 服務（主要）
├── stt_cli.py              # 命令行工具（主要）
├── start_api.py            # 統一啟動腳本（主要）
├── local_stt.py            # 核心 STT 功能
├── requirements.txt         # Python 依賴
├── README.md               # 說明文檔
├── ffmpeg_bin/             # FFmpeg 二進制文件
└── Whisper-Taiwanese-model-v0.5/  # 模型文件
```

## 🎯 使用場景

### 個人使用
- 語音筆記轉文字
- 音頻文件轉錄
- 學習台語發音

### 開發者使用
- 語音辨識 API 服務
- 批量音頻處理
- 語音數據預處理

### 教育用途
- 台語教學輔助
- 發音練習評估
- 語音資料庫建立

## 🔍 故障排除

### 常見問題

1. **麥克風無法訪問**
   - 檢查瀏覽器權限設置
   - 確認麥克風設備已啟用
   - 檢查防火牆設置

2. **GPU 不可用**
   - 安裝 CUDA 版本的 PyTorch
   - 檢查 NVIDIA 驅動
   - 確認 CUDA 版本兼容性

3. **模型載入失敗**
   - 檢查模型文件路徑
   - 確認模型文件完整性
   - 檢查磁盤空間

4. **依賴缺失**
   - 運行 `python start_api.py --check`
   - 手動安裝缺失的包
   - 檢查 Python 版本

### 日誌查看

使用 `--verbose` 參數可以查看詳細的日誌信息：

```bash
python start_api.py --verbose
python stt_cli.py -v
```

## 📞 技術支持

如遇問題，請檢查：

1. 日誌輸出信息
2. 依賴包版本
3. 系統配置
4. 網絡連接狀態

## 📄 授權

本項目基於開源技術構建，遵循相關開源協議。

---

🎵 **享受台語語音辨識的樂趣！** ✨
