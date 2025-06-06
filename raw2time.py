import json
from pathlib import Path
from datetime import datetime
from dateutil.rrule import rrule, DAILY
from data_processer import process_data
from state_counter import count_state

def generate_days(start_date, end_date):
    return rrule(
        DAILY,
        dtstart=start_date,
        until=end_date,
    )

def main():
    # 创建必要目录
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/time").mkdir(parents=True, exist_ok=True)

    # 设置日期范围
    start_date = datetime(2025, 3, 7)
    end_date = datetime(2025, 5, 30)

    # 遍历每个工作日
    for day in generate_days(start_date, end_date):
        date_str = day.strftime("%Y%m%d")

        try:
            raw_path = Path(f"data/raw/{date_str}.json")

            if raw_path.exists():
                print(f"Processing {date_str}...")

                # data processer
                processed_path = Path(f"data/processed/{date_str}.json")
                print("Processing raw data...")
                processed_data = process_data(raw_path)
                
                with open(processed_path, "w") as f:
                    json.dump(processed_data, f, indent=2, ensure_ascii=False)

                # state detector
                time_path = Path(f"data/time/{date_str}.json")
                print("Generating time statistics...")
                time_data = count_state(processed_path)
                    
                if time_data:
                    with open(time_path, "w") as f:
                        json.dump(time_data, f, indent=4)
            
            else:
                print(f"No raw data on {date_str}, skipped")

        except Exception as e:
            print(f"Error processing {date_str}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
