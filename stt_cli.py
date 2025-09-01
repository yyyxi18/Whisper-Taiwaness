#!/usr/bin/env python3
"""
台語語音辨識統一命令行工具
整合了簡單和進階功能，支持多種使用方式
"""

import argparse
import sys
import os
from local_stt import LocalSTT

def main():
    parser = argparse.ArgumentParser(
        description='🎯 台語語音辨識統一命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 簡單使用 - 直接辨識音頻文件
  python stt_cli.py audio.wav
  
  # 進階使用 - 保存結果到文件
  python stt_cli.py audio.wav -o result.txt
  
  # 批量處理目錄中的音頻文件
  python stt_cli.py --batch audio_folder/ -o results/
  
  # 顯示模型信息
  python stt_cli.py --info
  
  # 詳細輸出模式
  python stt_cli.py audio.wav -v
        """
    )
    
    # 主要參數
    parser.add_argument('audio_file', nargs='?', help='音頻文件路徑')
    parser.add_argument('--output', '-o', help='輸出文件路徑')
    parser.add_argument('--batch', help='批量處理目錄路徑')
    parser.add_argument('--info', action='store_true', help='顯示模型信息')
    parser.add_argument('--model-path', help='自定義模型路徑')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細輸出')
    parser.add_argument('--simple', action='store_true', help='簡單模式（無需參數）')
    
    args = parser.parse_args()
    
    # 簡單模式 - 如果沒有參數，顯示幫助
    if not args.simple and not any([args.audio_file, args.batch, args.info]):
        parser.print_help()
        print("\n💡 提示：使用 --simple 參數進入簡單模式")
        return
    
    # 設置日誌級別
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 初始化STT
        print("🎯 台語語音辨識統一工具")
        print("=" * 50)
        
        stt = LocalSTT(args.model_path)
        
        if not stt.is_ready():
            print("❌ 模型載入失敗！")
            sys.exit(1)
        
        print("✅ 模型載入成功！")
        model_info = stt.get_model_info()
        print(f"📱 使用設備: {'GPU' if model_info['device'] == 'GPU' else 'CPU'}")
        print(f"🔧 模型路徑: {model_info['model_path']}")
        print()
        
        # 顯示模型信息
        if args.info:
            print("📋 詳細模型信息:")
            for key, value in model_info.items():
                print(f"   {key}: {value}")
            return
        
        # 批量處理
        if args.batch:
            process_batch(args.batch, stt, args.output)
            return
        
        # 單個文件處理
        if args.audio_file:
            process_single_file(args.audio_file, stt, args.output, args.verbose)
            return
            
    except KeyboardInterrupt:
        print("\n🛑 操作已取消")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def process_single_file(audio_file, stt, output_file, verbose=False):
    """處理單個音頻文件"""
    # 檢查文件是否存在
    if not os.path.exists(audio_file):
        print(f"❌ 文件不存在: {audio_file}")
        return
    
    print(f"🎵 正在辨識: {audio_file}")
    if verbose:
        print(f"📁 文件大小: {os.path.getsize(audio_file) / 1024:.1f} KB")
    print("-" * 50)
    
    # 執行辨識
    result = stt.transcribe_file(audio_file)
    
    if result.get('success'):
        transcription = result['transcription']
        print("✅ 辨識成功！")
        print(f"🎵 辨識結果: {transcription}")
        
        if verbose:
            print(f"📊 採樣率: {result.get('sample_rate', '未知')} Hz")
            print(f"📁 文件路徑: {result.get('file_path', '未知')}")
            if 'processing_time' in result:
                print(f"⏱️ 處理時間: {result['processing_time']:.2f} 秒")
        
        # 保存到文件
        if output_file:
            try:
                os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                print(f"💾 結果已保存到: {output_file}")
            except Exception as e:
                print(f"⚠️ 保存文件失敗: {e}")
    else:
        print(f"❌ 辨識失敗: {result.get('error', '未知錯誤')}")

def process_batch(folder_path, stt, output_folder):
    """批量處理目錄中的音頻文件"""
    if not os.path.exists(folder_path):
        print(f"❌ 目錄不存在: {folder_path}")
        return
    
    # 支持的音頻格式
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.m4a'}
    
    # 獲取所有音頻文件
    audio_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"❌ 在目錄 {folder_path} 中沒有找到音頻文件")
        return
    
    print(f"📁 找到 {len(audio_files)} 個音頻文件")
    print(f"🎵 支持的格式: {', '.join(audio_extensions)}")
    print()
    
    # 創建輸出目錄
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        print(f"📂 輸出目錄: {output_folder}")
        print()
    
    # 處理每個文件
    success_count = 0
    total_time = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] 處理: {os.path.basename(audio_file)}")
        
        result = stt.transcribe_file(audio_file)
        
        if result.get('success'):
            success_count += 1
            transcription = result['transcription']
            
            if 'processing_time' in result:
                total_time += result['processing_time']
            
            # 保存結果
            if output_folder:
                output_file = os.path.join(
                    output_folder, 
                    os.path.splitext(os.path.basename(audio_file))[0] + '.txt'
                )
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(transcription)
                    print(f"   ✅ 已保存到: {output_file}")
                except Exception as e:
                    print(f"   ⚠️ 保存失敗: {e}")
            else:
                print(f"   ✅ 辨識結果: {transcription[:50]}{'...' if len(transcription) > 50 else ''}")
        else:
            print(f"   ❌ 失敗: {result.get('error', '未知錯誤')}")
        
        print()
    
    # 顯示統計信息
    print(f"🎯 批量處理完成！")
    print(f"📊 統計信息:")
    print(f"   總文件數: {len(audio_files)}")
    print(f"   成功數量: {success_count}")
    print(f"   失敗數量: {len(audio_files) - success_count}")
    print(f"   成功率: {success_count/len(audio_files)*100:.1f}%")
    if total_time > 0:
        print(f"   總處理時間: {total_time:.2f} 秒")
        print(f"   平均處理時間: {total_time/success_count:.2f} 秒")

def simple_mode():
    """簡單模式 - 無需參數的交互式使用"""
    print("🎯 台語語音辨識 - 簡單模式")
    print("=" * 40)
    
    # 檢查是否有音頻文件
    audio_files = []
    for file in os.listdir('.'):
        if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
            audio_files.append(file)
    
    if audio_files:
        print("📁 發現以下音頻文件:")
        for i, file in enumerate(audio_files, 1):
            print(f"   {i}. {file}")
        print()
        
        try:
            choice = input("請選擇要辨識的文件編號 (或按Enter退出): ").strip()
            if choice and choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(audio_files):
                    selected_file = audio_files[idx]
                    print(f"\n🎵 選擇了: {selected_file}")
                    
                    # 創建STT實例並處理
                    stt = LocalSTT()
                    if stt.is_ready():
                        process_single_file(selected_file, stt, None, verbose=True)
                    else:
                        print("❌ 模型載入失敗！")
                else:
                    print("❌ 無效的選擇")
            else:
                print("👋 退出程序")
        except KeyboardInterrupt:
            print("\n👋 退出程序")
    else:
        print("📁 當前目錄中沒有找到音頻文件")
        print("💡 請將音頻文件放在此目錄中，或使用命令行參數指定文件路徑")
        print("\n使用示例:")
        print("  python stt_cli.py audio.wav")
        print("  python stt_cli.py --batch audio_folder/")

if __name__ == "__main__":
    # 檢查是否為簡單模式
    if len(sys.argv) == 1:
        simple_mode()
    else:
        main()
