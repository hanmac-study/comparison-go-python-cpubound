#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 구현 벤치마크 스크립트.
Go 언어 구현과의 성능 비교를 위한 스크립트입니다.
"""

import argparse
import time
import pandas as pd
import numpy as np
import talib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import statistics
import logging
from datetime import datetime
import platform

# 로깅 설정
def setup_logger():
    """로깅 설정 함수"""
    logger = logging.getLogger('benchmark')
    logger.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 파일 핸들러
    file_handler = logging.FileHandler('benchmark.log')
    file_handler.setLevel(logging.INFO)
    
    # 포맷 설정
    formatter = logging.Formatter('[benchmark.py][%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 로거 초기화
logger = setup_logger()

# 벤치마크 티커 목록
BENCHMARK_TICKERS = [
    "KRW-BTC", "KRW-ETH", "KRW-ETC", "KRW-GRS", "KRW-ICX",
    "KRW-XRP", "KRW-WAVES", "KRW-QTUM", "KRW-MTL", "KRW-ADA"
]

# 데이터베이스 연결 설정
import os
import psycopg2
from sqlalchemy import create_engine

from dotenv import load_dotenv

load_dotenv()

# 데이터베이스 연결 정보
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_SSLMODE = os.getenv("DB_SSLMODE")

# 데이터베이스 연결 문자열
CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={DB_SSLMODE}"


def get_ticker_data(ticker, period=200):
    """데이터베이스에서 티커 데이터를 가져옵니다."""
    logger.info(f"{ticker} 데이터 로드 시작 (기간: {period})")
    try:
        # SQLAlchemy 엔진 생성
        engine = create_engine(CONNECTION_STRING)

        # 쿼리 실행
        query = f"""
        SELECT timestamp, code, trade_price, opening_price, high_price, low_price, candle_acc_trade_volume, candle_acc_trade_price
        FROM upbit_candle_1m
        WHERE code = '{ticker}'
        ORDER BY timestamp
        LIMIT 30000
        """

        df = pd.read_sql(query, engine)
        engine.dispose()
        
        logger.info(f"{ticker} 데이터 로드 완료: {len(df)} 행")
        return df
    except Exception as e:
        logger.error(f"{ticker} 데이터 로드 실패: {e}")
        return pd.DataFrame()


def calculate_adx(df, period=14):
    """ADX 지표를 계산합니다."""
    logger.info(f"ADX 계산 시작 (기간: {period})")
    if df.empty:
        logger.warning("ADX 계산 실패: 빈 데이터프레임")
        return pd.DataFrame()

    # 결과 DataFrame 생성
    result = pd.DataFrame()
    result['timestamp'] = df['timestamp']

    # ADX 계산
    result['adx'] = talib.ADX(df['high_price'].values, df['low_price'].values, df['trade_price'].values, timeperiod=period)

    # +DI, -DI 계산
    result['plus_di'] = talib.PLUS_DI(df['high_price'].values, df['low_price'].values, df['trade_price'].values, timeperiod=period)
    result['minus_di'] = talib.MINUS_DI(df['high_price'].values, df['low_price'].values, df['trade_price'].values, timeperiod=period)

    # 추가 계산 (Go 구현과 동일하게)
    result['di_difference'] = result['plus_di'] - result['minus_di']
    result['di_sum'] = result['plus_di'] + result['minus_di']

    # ADX 트렌드 강도
    result['trend_strength'] = 0  # 기본값
    result.loc[result['adx'] < 25, 'trend_strength'] = 0
    result.loc[(result['adx'] >= 25) & (result['adx'] < 50), 'trend_strength'] = 1
    result.loc[(result['adx'] >= 50) & (result['adx'] < 75), 'trend_strength'] = 2
    result.loc[result['adx'] >= 75, 'trend_strength'] = 3

    # 방향성 이동
    result['directional_movement'] = 0  # 기본값
    plus_dominant = result['plus_di'] > result['minus_di'] * 1.2
    minus_dominant = result['minus_di'] > result['plus_di'] * 1.2
    result.loc[plus_dominant, 'directional_movement'] = 1
    result.loc[minus_dominant, 'directional_movement'] = -1

    logger.info("ADX 계산 완료")
    return result


def calculate_atr(df, periods=[7, 14, 21]):
    """ATR 지표를 계산합니다."""
    logger.info(f"ATR 계산 시작 (기간: {periods})")
    if df.empty:
        logger.warning("ATR 계산 실패: 빈 데이터프레임")
        return pd.DataFrame()

    # 결과 DataFrame 생성
    result = pd.DataFrame()
    result['timestamp'] = df['timestamp']

    # 각 기간별 ATR 계산
    for period in periods:
        result[f'atr_{period}'] = talib.ATR(df['high_price'].values, df['low_price'].values, df['trade_price'].values, timeperiod=period)
        result[f'atr_{period}_ratio'] = result[f'atr_{period}'] / df['trade_price']

    # 진정한 범위
    result['true_range'] = talib.TRANGE(df['high_price'].values, df['low_price'].values, df['trade_price'].values)

    # HL 범위
    result['hl_range'] = df['high_price'] - df['low_price']

    # HC, LC 범위 계산 (첫 행 제외)
    result['hc_range'] = np.abs(df['high_price'] - df['trade_price'].shift(1))
    result['lc_range'] = np.abs(df['low_price'] - df['trade_price'].shift(1))

    logger.info("ATR 계산 완료")
    return result


def calculate_bollinger(df, period=20, std_multiplier=2.0):
    """볼린저 밴드 지표를 계산합니다."""
    logger.info(f"볼린저 밴드 계산 시작 (기간: {period}, 표준편차: {std_multiplier})")
    if df.empty:
        logger.warning("볼린저 밴드 계산 실패: 빈 데이터프레임")
        return pd.DataFrame()

    # 결과 DataFrame 생성
    result = pd.DataFrame()
    result['timestamp'] = df['timestamp']

    # 볼린저 밴드 계산
    result['bb_upper'], result['bb_middle'], result['bb_lower'] = talib.BBANDS(
        df['trade_price'].values,
        timeperiod=period,
        nbdevup=std_multiplier,
        nbdevdn=std_multiplier,
        matype=0  # Simple Moving Average
    )

    # 밴드 폭과 위치 계산
    result['bb_width'] = result['bb_upper'] - result['bb_lower']
    result['bb_width_ratio'] = result['bb_width'] / result['bb_middle']

    # %B 계산 (가격 위치)
    result['bb_position'] = (df['trade_price'] - result['bb_lower']) / result['bb_width']

    # 기타 상태 계산
    result['price_above_middle'] = df['trade_price'] > result['bb_middle']
    result['price_distance_from_middle'] = (df['trade_price'] - result['bb_middle']) / result['bb_middle']

    # 터치 및 돌파 상태
    result['touch_upper'] = False
    result['touch_lower'] = False
    result['break_upper'] = False
    result['break_lower'] = False

    # 근접 기준 (밴드 폭의 1%)
    proximity_threshold = result['bb_width'] * 0.01

    result.loc[np.abs(df['high_price'] - result['bb_upper']) <= proximity_threshold, 'touch_upper'] = True
    result.loc[np.abs(df['low_price'] - result['bb_lower']) <= proximity_threshold, 'touch_lower'] = True
    result.loc[df['high_price'] > result['bb_upper'], 'break_upper'] = True
    result.loc[df['low_price'] < result['bb_lower'], 'break_lower'] = True

    logger.info("볼린저 밴드 계산 완료")
    return result


def generate_fibonacci_numbers(count):
    """피보나치 수열을 생성합니다."""
    # 기본 피보나치 비율
    base_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618, 3.618, 4.236]
    
    # 피보나치 수열 계산 (CPU 부하 증가를 위해 많은 수 계산)
    fib_numbers = np.zeros(100)
    fib_numbers[0] = 0
    fib_numbers[1] = 1
    
    for i in range(2, 100):
        fib_numbers[i] = fib_numbers[i-1] + fib_numbers[i-2]
    
    # 결과 배열 생성
    result = np.zeros(count)
    
    # 기본 비율과 계산된 수열을 조합
    for i in range(count):
        if i < len(base_ratios):
            result[i] = base_ratios[i]
        else:
            idx = i % len(fib_numbers)
            # 정규화
            result[i] = fib_numbers[idx] / fib_numbers[-1]
    
    return result


def calculate_fibonacci(df, period=30):
    """피보나치 리트레이스먼트와 확장 레벨을 계산합니다."""
    logger.info(f"피보나치 계산 시작 (기간: {period})")
    if df.empty or len(df) < period:
        logger.warning("피보나치 계산 실패: 데이터 부족")
        return pd.DataFrame()
    
    # 피보나치 수열 생성 (CPU 연산을 많이 사용하기 위해 깊게 생성)
    fib_numbers = generate_fibonacci_numbers(20)
    
    # 결과 DataFrame 생성
    result = pd.DataFrame()
    result['timestamp'] = df['timestamp']
    
    # 수준, 리트레이스 포인트, 확장 포인트 컬럼 미리 생성
    result['levels'] = None
    result['retrace_points'] = None
    result['extension_points'] = None
    result['trend_direction'] = 0
    result['strength_score'] = 0.0
    
    # 각 캔들마다 계산
    for i in range(period, len(df)):
        # 현재 윈도우
        window = df.iloc[i-period:i+1]
        
        # 고가와 저가 찾기
        high = window['high_price'].max()
        low = window['low_price'].min()
        
        # 추세 방향 결정 (계산 부하를 위해 복잡한 계산 추가)
        avg_start = window.iloc[:period//2]['trade_price'].mean()
        avg_end = window.iloc[-period//2:]['trade_price'].mean()
        
        direction = 0
        if avg_end > avg_start:
            direction = 1
        elif avg_end < avg_start:
            direction = -1
            
        result.at[i, 'trend_direction'] = direction
        
        # 피보나치 레벨 계산 (계산 부하를 위해 많은 레벨 계산)
        diff = high - low
        levels = []
        
        for j in range(len(fib_numbers)):
            if direction >= 0:
                level = low + diff * fib_numbers[j]
            else:
                level = high - diff * fib_numbers[j]
            levels.append(level)
            
        # 리트레이스먼트 포인트 계산 (추가 계산 부하)
        retrace_points = []
        for j in range(period):
            if j >= len(window):
                break
                
            price = window.iloc[-j-1]['trade_price']
            price_ratio = (price - low) / diff if diff != 0 else 0
            
            # 가장 가까운 피보나치 레벨 찾기 (계산 부하를 위한 루프)
            closest_level = 0.0
            min_diff = float('inf')
            
            for k in range(len(levels)):
                level_ratio = fib_numbers[k]
                if abs(price_ratio - level_ratio) < min_diff:
                    min_diff = abs(price_ratio - level_ratio)
                    closest_level = level_ratio
                    
            retrace_points.append(closest_level)
            
        # 확장 포인트 계산 (추가 계산 부하)
        extension_points = []
        for j in range(period):
            # 복잡한 계산 로직 (CPU 부하 증가용)
            base = (high + low) / 2
            amplitude = (high - low) / 2
            angle = 2 * np.pi * j / period
            
            # 사인, 코사인 함수 사용하여 계산 부하 증가
            value = base + amplitude * np.sin(angle)
            
            # 피보나치 수열의 값을 가중치로 사용
            weight = fib_numbers[j % len(fib_numbers)]
            
            extension_points.append(value * weight)
            
        # 강도 점수 계산 (복잡한 계산 추가)
        strength_score = 0.0
        for j in range(min(period, len(window))):
            price = window.iloc[-j-1]['trade_price']
            distance = 0.0
            
            for k in range(len(levels)):
                if levels[k] == 0:  # 0으로 나누기 방지
                    continue
                level_distance = abs(price - levels[k]) / levels[k]
                weight = np.exp(-level_distance * 10)  # 지수 함수로 가중치 계산
                distance += weight * fib_numbers[k]
                
            strength_score += distance
            
        strength_score /= period
        
        # 결과 저장 (리스트를 문자열로 변환하지 않고 객체 그대로 저장)
        result.at[i, 'levels'] = levels
        result.at[i, 'retrace_points'] = retrace_points
        result.at[i, 'extension_points'] = extension_points
        result.at[i, 'strength_score'] = strength_score
        
    logger.info("피보나치 계산 완료")
    return result


def load_all_data():
    """모든 티커 데이터를 한 번에 로드합니다."""
    logger.info("데이터베이스에서 모든 티커 데이터 로드 시작")
    print("Loading data from database...")
    all_data = {}

    for ticker in BENCHMARK_TICKERS:
        df = get_ticker_data(ticker)
        if not df.empty:
            all_data[ticker] = df
        else:
            logger.warning(f"{ticker} 데이터 없음")
            print(f"Warning: No data for {ticker}")

    logger.info(f"총 {len(all_data)}개 티커 데이터 로드 완료")
    print(f"Loaded data for {len(all_data)} tickers")
    return all_data


def process_ticker_from_memory(ticker_data, indicator_type):
    """메모리에 있는 데이터로 지표를 계산합니다."""
    if ticker_data.empty:
        logger.warning("빈 데이터프레임으로 처리 불가")
        return False
    
    ticker = ticker_data['code'].iloc[0] if not ticker_data.empty else "Unknown"
    logger.info(f"{ticker} - {indicator_type} 처리 시작")
    
    try:
        start_time = time.time()
        
        if indicator_type == "adx":
            result = calculate_adx(ticker_data)
            processed = not result.empty
        elif indicator_type == "atr":
            result = calculate_atr(ticker_data)
            processed = not result.empty
        elif indicator_type == "bollinger":
            result = calculate_bollinger(ticker_data)
            processed = not result.empty
        elif indicator_type == "fibonacci":
            result = calculate_fibonacci(ticker_data, period=30)
            processed = not result.empty
        else:
            logger.warning(f"알 수 없는 지표 유형: {indicator_type}")
            return False
        
        duration = time.time() - start_time
        logger.info(f"{ticker} - {indicator_type} 처리 완료: {duration:.4f}초")
        
        return processed
    except Exception as e:
        logger.error(f"{ticker} - {indicator_type} 처리 오류: {e}")
        return False


def process_ticker_with_args(args):
    """병렬 처리를 위한 도우미 함수"""
    ticker, data, indicator_type = args
    return ticker, process_ticker_from_memory(data, indicator_type)


def process_parallel_from_memory(all_data, indicator_type, workers=None):
    """메모리 데이터로 병렬 처리 방식으로 지표를 계산합니다."""
    if workers is None:
        workers = multiprocessing.cpu_count()
    
    logger.info(f"병렬 처리 시작: {indicator_type} 지표 (워커: {workers}개)")
    start_time = time.time()
    success_count = 0
    
    # 티커와 데이터 리스트 생성
    tasks = [(ticker, data, indicator_type) for ticker, data in all_data.items()]
    
    # ProcessPoolExecutor로 병렬 처리
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # 작업 제출 - lambda 대신 분리된 함수 사용
        future_to_ticker = {
            executor.submit(process_ticker_with_args, task): task[0] 
            for task in tasks
        }
        
        # 결과 수집
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                ticker_name, success = future.result()
                if success:
                    success_count += 1
                    logger.info(f"{ticker_name} 병렬 처리 성공")
                else:
                    logger.warning(f"{ticker_name} 병렬 처리 실패")
            except Exception as e:
                logger.error(f"{ticker} 처리 중 오류 발생: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"병렬 처리 완료: {success_count}/{len(all_data)} 성공, 소요 시간: {elapsed_time:.4f}초")
    return elapsed_time, success_count


def process_traditional_from_memory(all_data, indicator_type):
    """메모리 데이터로 전통적인 방식(순차 처리)으로 지표를 계산합니다."""
    logger.info(f"전통적 순차 처리 시작: {indicator_type} 지표")
    start_time = time.time()
    success_count = 0

    for ticker, data in all_data.items():
        logger.info(f"{ticker} 처리 시작")
        if process_ticker_from_memory(data, indicator_type):
            success_count += 1
        logger.info(f"{ticker} 처리 완료")

    elapsed_time = time.time() - start_time
    logger.info(f"전통적 순차 처리 완료: {success_count}/{len(all_data)} 성공, 소요 시간: {elapsed_time:.4f}초")
    return elapsed_time, success_count


def measure_resources():
    """시스템 리소스를 측정합니다."""
    logger.info("시스템 리소스 측정")
    
    # 전체 시스템 CPU 사용량
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # 현재 프로세스 정보
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # MB 단위로 변환
    
    # 프로세스별 CPU 시간 (사용자 모드 + 시스템 모드)
    cpu_times = process.cpu_times()
    process_cpu_user = cpu_times.user
    process_cpu_system = cpu_times.system
    process_cpu_total = process_cpu_user + process_cpu_system
    
    logger.info(f"CPU 사용량: {cpu_percent:.2f}%, 메모리 사용량: {memory_mb:.2f}MB")
    logger.info(f"프로세스 CPU 시간 - 사용자: {process_cpu_user:.4f}s, 시스템: {process_cpu_system:.4f}s, 총합: {process_cpu_total:.4f}s")
    
    return {
        'cpu_percent': cpu_percent,
        'memory_mb': memory_mb,
        'process_cpu_user': process_cpu_user,
        'process_cpu_system': process_cpu_system,
        'process_cpu_total': process_cpu_total
    }


def calculate_stats(durations):
    """통계를 계산합니다."""
    logger.info("실행 시간 통계 계산 시작")
    min_duration = min(durations)
    max_duration = max(durations)
    avg_duration = sum(durations) / len(durations)
    
    # 백분위수 계산
    sorted_durations = sorted(durations)
    p95_idx = int(len(sorted_durations) * 0.95)
    p99_idx = int(len(sorted_durations) * 0.99)
    p95 = sorted_durations[p95_idx]
    p99 = sorted_durations[p99_idx]
    
    logger.info(f"통계 결과 - 최소: {min_duration:.4f}초, 최대: {max_duration:.4f}초, 평균: {avg_duration:.4f}초, P95: {p95:.4f}초, P99: {p99:.4f}초")
    return min_duration, max_duration, avg_duration, p95, p99


def run_benchmark(use_parallel=True, workers=None, iterations=100):
    """벤치마크를 실행합니다."""
    logger.info(f"벤치마크 시작 (병렬: {use_parallel}, 워커: {workers}, 반복: {iterations})")
    
    # 최대 워커 수 설정
    if workers is None:
        workers = multiprocessing.cpu_count()
        logger.info(f"워커 수 자동 설정: {workers}")
    
    # 모든 데이터를 한 번에 로드
    all_data = load_all_data()
    if not all_data:
        logger.error("데이터 로드 실패")
        return
    
    print(f"\n=== Python Implementation Performance ===")
    print(f"Workers: {workers}, Parallel mode: {use_parallel}, Iterations: {iterations}\n")
    
    indicators = ["adx", "atr", "bollinger", "fibonacci"]
    
    for indicator in indicators:
        logger.info(f"=== {indicator.upper()} 지표 벤치마크 시작 ===")
        print(f"\n--- {indicator} ---")
        
        execution_times = []
        cpu_usages = []
        memory_usages = []
        
        # 프로세스 CPU 시간 측정을 위한 리스트
        pre_resources_list = []
        post_resources_list = []
        
        for i in range(iterations):
            logger.info(f"{indicator} 지표: {i}/{iterations} 반복 완료")
            if i % 10 == 0 or iterations < 10:
                print(f"Running iteration {i+1}/{iterations}...", end="\r")
            
            # CPU, 메모리 사용량 측정
            pre_resources = measure_resources()
            pre_resources_list.append(pre_resources)
            
            # 처리 방식에 따라 실행
            if use_parallel:
                duration, _ = process_parallel_from_memory(all_data, indicator, workers)
            else:
                duration, _ = process_traditional_from_memory(all_data, indicator)
            
            # 실행 후 리소스 측정
            post_resources = measure_resources()
            post_resources_list.append(post_resources)
            
            # 결과 기록
            execution_times.append(duration)
            cpu_usages.append(post_resources['cpu_percent'] - pre_resources['cpu_percent'] if post_resources['cpu_percent'] > pre_resources['cpu_percent'] else post_resources['cpu_percent'])
            memory_usages.append(post_resources['memory_mb'])
        
        print(f"\nCompleted {iterations} iterations for {indicator}.")
        
        # 실행 시간 통계
        logger.info(f"{indicator} 실행 시간 통계 계산")
        min_time, max_time, avg_time, p95, p99 = calculate_stats(execution_times)
        
        print("\nExecution Time Statistics:")
        print(f"Min: {min_time:.4f} seconds")
        print(f"Max: {max_time:.4f} seconds")
        print(f"Avg: {avg_time:.4f} seconds")
        print(f"P95: {p95:.4f} seconds")
        print(f"P99: {p99:.4f} seconds")
        
        # CPU 사용량 통계
        logger.info(f"{indicator} CPU 사용량 통계 계산")
        if cpu_usages:
            min_cpu = min(cpu_usages)
            max_cpu = max(cpu_usages)
            avg_cpu = sum(cpu_usages) / len(cpu_usages)
            
            print("\nCPU Usage:")
            print(f"Min: {min_cpu:.2f}%")
            print(f"Max: {max_cpu:.2f}%")
            print(f"Avg: {avg_cpu:.2f}%")
        
        # 메모리 사용량 통계
        logger.info(f"{indicator} 메모리 사용량 통계 계산")
        if memory_usages:
            min_mem = min(memory_usages)
            max_mem = max(memory_usages)
            avg_mem = sum(memory_usages) / len(memory_usages)
            
            print("\nMemory Usage:")
            print(f"Min: {min_mem:.2f} MB")
            print(f"Max: {max_mem:.2f} MB")
            print(f"Avg: {avg_mem:.2f} MB")
        
        # 프로세스별 CPU 시간 통계
        logger.info(f"{indicator} 프로세스별 CPU 시간 통계 계산")
        if pre_resources_list and post_resources_list:
            process_cpu_user_diff = [post['process_cpu_user'] - pre['process_cpu_user'] 
                                    for post, pre in zip(post_resources_list, pre_resources_list)]
            process_cpu_system_diff = [post['process_cpu_system'] - pre['process_cpu_system'] 
                                      for post, pre in zip(post_resources_list, pre_resources_list)]
            process_cpu_total_diff = [post['process_cpu_total'] - pre['process_cpu_total'] 
                                     for post, pre in zip(post_resources_list, pre_resources_list)]
            
            if process_cpu_user_diff:
                min_cpu_user = min(process_cpu_user_diff)
                max_cpu_user = max(process_cpu_user_diff)
                avg_cpu_user = sum(process_cpu_user_diff) / len(process_cpu_user_diff)
                
                print("\nProcess CPU Time (User):")
                print(f"Min: {min_cpu_user:.4f} seconds")
                print(f"Max: {max_cpu_user:.4f} seconds")
                print(f"Avg: {avg_cpu_user:.4f} seconds")
            
            if process_cpu_system_diff:
                min_cpu_system = min(process_cpu_system_diff)
                max_cpu_system = max(process_cpu_system_diff)
                avg_cpu_system = sum(process_cpu_system_diff) / len(process_cpu_system_diff)
                
                print("\nProcess CPU Time (System):")
                print(f"Min: {min_cpu_system:.4f} seconds")
                print(f"Max: {max_cpu_system:.4f} seconds")
                print(f"Avg: {avg_cpu_system:.4f} seconds")
            
            if process_cpu_total_diff:
                min_cpu_total = min(process_cpu_total_diff)
                max_cpu_total = max(process_cpu_total_diff)
                avg_cpu_total = sum(process_cpu_total_diff) / len(process_cpu_total_diff)
                
                print("\nProcess CPU Time (Total):")
                print(f"Min: {min_cpu_total:.4f} seconds")
                print(f"Max: {max_cpu_total:.4f} seconds")
                print(f"Avg: {avg_cpu_total:.4f} seconds")
        
        logger.info(f"=== {indicator.upper()} 지표 벤치마크 완료 ===")


def main():
    """메인 함수"""
    logger.info("Python 벤치마크 시작")
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='Python implementation benchmark')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations for each test')
    parser.add_argument('--workers', type=int, help='Number of worker processes for parallel execution')
    parser.add_argument('--traditional', action='store_true', help='Use traditional sequential processing')
    
    args = parser.parse_args()
    
    logger.info(f"설정 - 반복: {args.iterations}, 워커: {args.workers}, 전통적 처리: {args.traditional}")
    print(f"Running benchmark with {args.iterations} iterations")
    if args.traditional:
        print("Using traditional sequential processing")
    else:
        workers = args.workers if args.workers else multiprocessing.cpu_count()
        print(f"Using parallel processing with {workers} workers")
    
    # 벤치마크 실행
    run_benchmark(
        use_parallel=not args.traditional,
        workers=args.workers,
        iterations=args.iterations
    )
    
    logger.info("Python 벤치마크 완료")


if __name__ == "__main__":
    main()