import json
from pathlib import Path
from datetime import datetime
from dateutil.rrule import rrule, DAILY, MO, TU, WE, TH, FR
from state_detector import process_videos
from data_processer import process_data
from state_counter import count_state

def generate_workdays(start_date, end_date):
    """生成工作日列表"""
    return rrule(
        DAILY,
        dtstart=start_date,
        until=end_date,
        byweekday=(MO, TU, WE, TH, FR)
    )

def main():
    # 创建必要目录
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/time").mkdir(parents=True, exist_ok=True)

    # 设置日期范围
    start_date = datetime(2025, 3, 8)
    end_date = datetime(2025, 4, 8)

    # 遍历每个工作日
    for day in generate_workdays(start_date, end_date):
        date_str = day.strftime("%Y%m%d")
        print(f"\nProcessing {date_str}...")

        try:
            # 步骤1: 视频处理
            raw_path = Path(f"data/raw/{date_str}.json")
            if not raw_path.exists():
                print("Processing video data...")
                daily_start = day.replace(hour=7, minute=0)
                daily_end = day.replace(hour=8, minute=5)
                
                raw_data = process_videos(
                    start_time=daily_start,
                    end_time=daily_end,
                )
                
                with open(raw_path, "w") as f:
                    json.dump(raw_data, f, indent=2, ensure_ascii=False)

            # 步骤2: 数据加工
            processed_path = Path(f"data/processed/{date_str}.json")
            if not processed_path.exists():
                print("Processing raw data...")
                processed_data = process_data(raw_path)
                
                with open(processed_path, "w") as f:
                    json.dump(processed_data, f, indent=2, ensure_ascii=False)

            # 步骤3: 时间统计
            time_path = Path(f"data/time/{date_str}.json")
            if not time_path.exists():
                print("Generating time statistics...")
                time_data = count_state(processed_path)
                
                if time_data:
                    with open(time_path, "w") as f:
                        json.dump(time_data, f, indent=4)

        except Exception as e:
            print(f"Error processing {date_str}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
