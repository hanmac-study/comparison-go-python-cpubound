import time
import pandas as pd
import requests
import os

def fetch_upbit_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    headers = {"Accept": "application/json"}
    params = {"isDetails": "false"}  
    try:
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        markets = res.json()
        krw_markets = [market['market'] for market in markets if market['market'].startswith('KRW-')]
        print(f"Fetched {len(krw_markets)} KRW markets from Upbit.")
        return krw_markets
    except requests.exceptions.RequestException as e:
        print(f"Error fetching KRW markets: {e}")
        return [] 

def fetch_upbit_candles(market, to=None, count=200):
    url = "https://api.upbit.com/v1/candles/minutes/5"
    headers = {"Accept": "application/json"}
    params = {
        "market": market,
        "count": count
    }
    if to:
        params["to"] = to
    res = requests.get(url, headers=headers, params=params)
    res.raise_for_status()
    return res.json()


def get_all_candles(market, max_count=52000):
    all_data = []
    to = None
    batch_count = 0

    while len(all_data) < max_count:
        try:
            candles = fetch_upbit_candles(market, to=to, count=min(200, max_count - len(all_data)))
        except requests.exceptions.RequestException as e:
            print(f"Error fetching candles for {market}: {e}")
            break # 오류 발생 시 중단
        
        if not candles:
            break
        all_data.extend(candles)
        # API 응답의 마지막 캔들 시간을 다음 요청의 'to' 파라미터로 사용합니다.
        # Upbit API는 'to' 이전의 캔들을 반환합니다.
        to = candles[-1]['candle_date_time_utc'] 

        batch_count += 1
        if batch_count % 5 == 0: # 진행 상황 더 자주 출력
            print(f"{market}: {len(all_data)}개 캔들 데이터 수집 중... ({min(len(all_data) / max_count * 100, 100):.1f}%)")

        time.sleep(0.2) # API 속도 제한 방지

    if not all_data:
        print(f"{market}: 수집된 데이터가 없습니다.")
        return pd.DataFrame() # 빈 DataFrame 반환

    print(f"{market}: 총 {len(all_data)}개 캔들 데이터 수집 완료")
    df_raw = pd.DataFrame(all_data) # 원본 API 데이터를 DataFrame으로 변환

    # API 응답 컬럼명과 표준 컬럼명 매핑 정의
    column_mapping = {
        'candle_date_time_utc': 'timestamp', # UTC 시간을 timestamp로 사용
        'opening_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'trade_price': 'close',  # Upbit 캔들 데이터에서 trade_price가 해당 봉의 종가임
        'candle_acc_trade_volume': 'volume'
    }

    # API 응답에 필요한 모든 원본 컬럼이 있는지 확인
    source_api_columns = list(column_mapping.keys())
    missing_api_cols = [col for col in source_api_columns if col not in df_raw.columns]
    if missing_api_cols:
        print(f"오류: API 응답에 다음 필수 컬럼이 없습니다: {missing_api_cols}. 사용 가능한 컬럼: {df_raw.columns.tolist()}")
        return pd.DataFrame()

    # 필요한 API 컬럼만 선택하여 새 DataFrame 생성 후, 컬럼명 변경
    df = df_raw[source_api_columns].copy() # .copy()를 사용하여 SettingWithCopyWarning 방지
    df.rename(columns=column_mapping, inplace=True)

    # 이제 'df'는 중복 없는 표준 컬럼명을 가짐
    # 'timestamp' 컬럼을 datetime 객체로 변환
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        print(f"오류: 'timestamp' 컬럼을 datetime으로 변환 중 예외 발생: {e}")
        print(f"변환 전 'timestamp' 컬럼 데이터 샘플 (최대 5개):\n{df['timestamp'].head().to_string()}")
        return pd.DataFrame() # 오류 발생 시 빈 DataFrame 반환
        
    df.sort_values('timestamp', inplace=True, ascending=True)

    # 숫자형 데이터 타입 변환
    for col in ['open', 'high', 'low', 'close', 'volume']:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception as e:
            print(f"오류: '{col}' 컬럼을 숫자로 변환 중 예외 발생: {e}")
            print(f"변환 전 '{col}' 컬럼 데이터 샘플 (최대 5개):\n{df[col].head().to_string()}")
            return pd.DataFrame()
            
    # 최종적으로 필요한 컬럼 순서대로 DataFrame 반환 (선택 사항, 현재는 rename 순서 따름)
    # final_columns_order = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    # df = df[final_columns_order]
            
    return df


if __name__ == '__main__':
    output_dir = ".venv/data"  # 데이터를 저장할 디렉터리
    candle_interval_value = "5m" # 5분봉을 의미

    # 1년치 5분봉 캔들 수 계산
    # 1일 = 24시간 * 60분/시간 = 1440분
    # 1일 5분봉 개수 = 1440분 / 5분 = 288개
    # 1년 5분봉 개수 = 288개/일 * 365일 = 105,120개
    candles_per_year = 288 * 365

    print(f"Upbit의 모든 KRW 마켓에 대해 1년치 5분봉 데이터(약 {candles_per_year}개/마켓)를 수집합니다.")
    print(f"데이터는 '{output_dir}/{{MARKET_CODE}}_{candle_interval_value}.csv' 형식으로 저장됩니다.")
    print("주의: 이 작업은 매우 많은 API 호출을 필요로 하며, 수 시간이 소요될 수 있습니다.")
    print("-" * 50)

    # 출력 디렉터리 생성 (없는 경우)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"디렉터리 '{output_dir}' 생성됨.")

    krw_market_codes = fetch_upbit_krw_markets()

    if not krw_market_codes:
        print("오류: KRW 마켓 목록을 가져오는 데 실패했습니다. 스크립트를 종료합니다.")
        exit()
    
    print(f"총 {len(krw_market_codes)}개의 KRW 마켓에 대해 데이터 수집을 시작합니다: {krw_market_codes}")

    for i, market_code in enumerate(krw_market_codes):
        print(f"\n[{i+1}/{len(krw_market_codes)}] '{market_code}' 마켓 데이터 수집 시작...")
        
        # 파일명 형식: {MARKET_CODE}_{INTERVAL}.csv, 예: KRW-BTC_5m.csv
        # CSVDataProvider가 사용하는 market_code는 'KRW-BTC'와 같은 형식이므로 그대로 사용합니다.
        output_filename = f"{market_code}_{candle_interval_value}.csv"
        output_path = os.path.join(output_dir, output_filename)

        # 이미 파일이 존재하면 건너뛸 수 있도록 옵션 추가 (사용자 선택)
        # 여기서는 무조건 새로 받도록 구현, 필요시 주석 해제하고 로직 추가
        # if os.path.exists(output_path):
        #     print(f"'{output_path}' 파일이 이미 존재합니다. 건너뜁니다.")
        #     continue

        df_candles = get_all_candles(market_code, max_count=candles_per_year)

        if not df_candles.empty:
            try:
                df_candles.to_csv(output_path, index=False)
                print(f"성공: '{market_code}' 마켓의 {len(df_candles)}개 캔들 데이터가 '{output_path}' 파일에 저장되었습니다.")
            except Exception as e:
                print(f"오류: '{output_path}' 파일 저장 중 예외 발생: {e}")
        else:
            print(f"경고: '{market_code}' 마켓에 대한 데이터를 수집하지 못했거나 데이터가 없습니다.")
        
        # 각 마켓 처리 후 짧은 추가 지연시간 (선택적)
        # time.sleep(1)

    print("\n모든 지정된 KRW 마켓에 대한 데이터 수집 작업이 완료되었습니다.")
