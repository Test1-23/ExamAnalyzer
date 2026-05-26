import os
import json
from src.pipeline import run_pipeline

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(THIS_DIR, "input")
POINTS_FILE = os.path.join(THIS_DIR, "point", "points.txt")
CONFIG_FILE = os.path.join(THIS_DIR, "config.json")


def _load_config() -> dict:
    """Load API config from file, with env var overrides."""
    config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass
    # Environment variables take precedence
    config["api_url"] = os.environ.get("DEEPSEEK_API_URL", config.get("api_url", ""))
    config["api_key"] = os.environ.get("DEEPSEEK_API_KEY", config.get("api_key", ""))
    return config


def main():
    config = _load_config()
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
