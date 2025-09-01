#!/usr/bin/env python3
"""
台語語音辨識統一啟動腳本
支持多種啟動模式和配置
"""

import os
import sys
import argparse
import logging
from pathlib import Path

def setup_logging(verbose=False):
    """設置日誌"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_dependencies():
    """檢查依賴"""
    required_packages = [
        'flask', 'flask-cors', 'torch', 'transformers', 
        'librosa', 'soundfile', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logging.info(f"✅ {package} 已安裝")
        except ImportError:
            missing_packages.append(package)
            logging.warning(f"❌ {package} 未安裝")
    
    if missing_packages:
        logging.error(f"缺少以下依賴包: {', '.join(missing_packages)}")
        logging.info("正在嘗試安裝...")
        
        for package in missing_packages:
            try:
                import subprocess
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logging.info(f"✅ {package} 安裝成功")
            except Exception as e:
                logging.error(f"❌ {package} 安裝失敗: {e}")
                return False
    
    return True

def check_model():
    """檢查模型文件"""
    model_path = "NUTN-KWS/Whisper-Taiwanese-model-v0.5"
    if os.path.exists(model_path):
        logging.info(f"✅ 模型文件存在: {model_path}")
        return True
    else:
        logging.warning(f"⚠️ 模型文件不存在: {model_path}")
        logging.info("請確保已下載模型文件")
        return False

def start_web_api(host="0.0.0.0", port=5000, verbose=False):
    """啟動Web API服務"""
    logging.info("🌐 啟動Web API服務...")
    
    try:
        from voice_api_enhanced import app, init_stt
        
        # 初始化STT模型
        init_stt()
        
        # 啟動Flask服務
        app.run(host=host, port=port, debug=verbose)
        
    except ImportError as e:
        logging.error(f"❌ 無法導入API模組: {e}")
        return False
    except Exception as e:
        logging.error(f"❌ 啟動API失敗: {e}")
        return False
    
    return True

def start_cli_tool():
    """啟動CLI工具"""
    logging.info("💻 啟動CLI工具...")
    
    try:
        from stt_cli import simple_mode
        simple_mode()
        return True
    except ImportError as e:
        logging.error(f"❌ 無法導入CLI模組: {e}")
        return False
    except Exception as e:
        logging.error(f"❌ 啟動CLI失敗: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='🎯 台語語音辨識統一啟動腳本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 啟動Web API服務 (默認)
  python start_api.py
  
  # 啟動Web API服務 (指定端口)
  python start_api.py --port 8080
  
  # 啟動Web API服務 (僅本地訪問)
  python start_api.py --local
  
  # 啟動CLI工具
  python start_api.py --cli
  
  # 詳細模式
  python start_api.py --verbose
  
  # 檢查依賴
  python start_api.py --check
        """
    )
    
    parser.add_argument('--cli', action='store_true', help='啟動CLI工具而不是Web API')
    parser.add_argument('--host', default='0.0.0.0', help='綁定主機地址 (默認: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='綁定端口 (默認: 5000)')
    parser.add_argument('--local', action='store_true', help='僅本地訪問 (綁定到127.0.0.1)')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細輸出')
    parser.add_argument('--check', action='store_true', help='僅檢查依賴和模型')
    
    args = parser.parse_args()
    
    # 設置日誌
    setup_logging(args.verbose)
    
    logging.info("🎯 台語語音辨識統一啟動腳本")
    logging.info("=" * 50)
    
    # 檢查依賴
    if not check_dependencies():
        logging.error("❌ 依賴檢查失敗")
        sys.exit(1)
    
    # 檢查模型
    if not check_model():
        logging.warning("⚠️ 模型檢查失敗，但繼續執行...")
    
    # 僅檢查模式
    if args.check:
        logging.info("✅ 檢查完成")
        return
    
    # 設置主機地址
    if args.local:
        args.host = '127.0.0.1'
    
    # 啟動服務
    if args.cli:
        # 啟動CLI工具
        success = start_cli_tool()
    else:
        # 啟動Web API
        logging.info(f"🌐 服務配置:")
        logging.info(f"   主機地址: {args.host}")
        logging.info(f"   端口: {args.port}")
        if args.host == '0.0.0.0':
            logging.info(f"   本地訪問: http://localhost:{args.port}")
            logging.info(f"   網絡訪問: http://<本機IP>:{args.port}")
        else:
            logging.info(f"   訪問地址: http://{args.host}:{args.port}")
        
        success = start_web_api(args.host, args.port, args.verbose)
    
    if not success:
        logging.error("❌ 啟動失敗")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("🛑 程序被用戶中斷")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ 程序異常: {e}")
        sys.exit(1)
