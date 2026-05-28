import os
from src.pipeline import run_pipeline
from src.config import load_config

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(THIS_DIR, "input")
POINTS_FILE = os.path.join(THIS_DIR, "point", "points.txt")


def main():
    config = load_config()
    api_url = config.get("api_url", "")
    api_key = config.get("api_key", "")

    if not api_url or not api_key:
        print("=" * 60)
        print("请配置 API URL 和 API Key！")
        print("  方式 1: 设置环境变量 DEEPSEEK_API_URL 和 DEEPSEEK_API_KEY")
        print("  方式 2: 在 Web UI (http://127.0.0.1:5000) 中配置，自动保存到 config.json")
        print("=" * 60)
        return

    def progress(pct: int, status: str = ""):
        print(f"[{pct}%] {status}" if status else f"[{pct}%]")

    print(f"启动分析...")
    print(f"  输入目录: {INPUT_DIR}")
    print(f"  输出文件: {POINTS_FILE}")

    try:
        run_pipeline(
            api_url=api_url,
            api_key=api_key,
            input_dir=INPUT_DIR,
            output_path=POINTS_FILE,
            progress_callback=progress,
            debug_callback=lambda msg: print(f"[DEBUG] {msg}"),
        )
        print("\n完成！")
    except Exception as e:
        print(f"\n错误: {e}")


if __name__ == "__main__":
    main()
