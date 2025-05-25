#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python과 Go 구현의 성능을 비교하는 스크립트
"""

import subprocess
import re
import sys
import os
import json
import logging
from datetime import datetime

# 로깅 설정
def setup_logger():
    """로깅 설정 함수"""
    logger = logging.getLogger('run_comparison')
    logger.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 파일 핸들러
    file_handler = logging.FileHandler('run_comparison.log')
    file_handler.setLevel(logging.INFO)
    
    # 포맷 설정
    formatter = logging.Formatter('[run_comparison.py][%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 로거 초기화
logger = setup_logger()

def parse_python_output(output):
    """Python/Go 출력에서 통계 파싱"""
    lines = output.strip().split('\n')
    results = {}
    current_indicator = None

    for i, line in enumerate(lines):
        # 지표 이름 찾기
        if '---' in line:
            match = re.search(r'--- (\w+) ---', line)
            if match:
                current_indicator = match.group(1).lower()
                logger.info(f"지표 발견: {current_indicator}")
                results[current_indicator] = {
                    'execution_time': {},
                    'cpu_usage': {},
                    'memory_usage': {}
                }

        # 실행 시간 통계
        if 'Execution Time Statistics:' in line and current_indicator:
            logger.info(f"{current_indicator} 실행 시간 통계 파싱")
            for j in range(1, 6):
                if i + j < len(lines):
                    stat_line = lines[i + j]
                    if 'Min:' in stat_line:
                        results[current_indicator]['execution_time']['min'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'Max:' in stat_line:
                        results[current_indicator]['execution_time']['max'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'Avg:' in stat_line:
                        results[current_indicator]['execution_time']['avg'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'P95:' in stat_line:
                        results[current_indicator]['execution_time']['p95'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'P99:' in stat_line:
                        results[current_indicator]['execution_time']['p99'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))

        # CPU 사용량 - System CPU Usage 또는 단순히 CPU Usage
        if ('System CPU Usage:' in line or 'CPU Usage:' in line) and current_indicator:
            logger.info(f"{current_indicator} CPU 사용량 파싱")
            for j in range(1, 4):
                if i + j < len(lines):
                    stat_line = lines[i + j]
                    if 'Min:' in stat_line:
                        results[current_indicator]['cpu_usage']['min'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'Max:' in stat_line:
                        results[current_indicator]['cpu_usage']['max'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'Avg:' in stat_line:
                        results[current_indicator]['cpu_usage']['avg'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))

        # 메모리 사용량
        if 'Memory Usage:' in line and current_indicator:
            logger.info(f"{current_indicator} 메모리 사용량 파싱")
            for j in range(1, 4):
                if i + j < len(lines):
                    stat_line = lines[i + j]
                    if 'Min:' in stat_line:
                        results[current_indicator]['memory_usage']['min'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'Max:' in stat_line:
                        results[current_indicator]['memory_usage']['max'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'Avg:' in stat_line:
                        results[current_indicator]['memory_usage']['avg'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))

    logger.info("Python 출력 파싱 완료")
    return results

def parse_go_output(output):
    """Go 출력에서 통계 파싱 (Python과 동일한 형식)"""
    logger.info("Go 출력 파싱 시작")
    lines = output.strip().split('\n')
    results = {}
    current_indicator = None

    for i, line in enumerate(lines):
        # 지표 이름 찾기 (--- adx ---, --- atr ---, --- bollinger ---, --- fibonacci ---)
        if '---' in line:
            match = re.search(r'--- (\w+) ---', line)
            if match:
                current_indicator = match.group(1).lower()
                logger.info(f"지표 발견: {current_indicator}")
                results[current_indicator] = {
                    'execution_time': {},
                    'cpu_usage': {},
                    'memory_usage': {}
                }

        # 실행 시간 통계
        if 'Execution Time Statistics:' in line and current_indicator:
            logger.info(f"{current_indicator} 실행 시간 통계 파싱")
            for j in range(1, 6):
                if i + j < len(lines):
                    stat_line = lines[i + j]
                    if 'Min:' in stat_line:
                        results[current_indicator]['execution_time']['min'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'Max:' in stat_line:
                        results[current_indicator]['execution_time']['max'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'Avg:' in stat_line:
                        results[current_indicator]['execution_time']['avg'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'P95:' in stat_line:
                        results[current_indicator]['execution_time']['p95'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'P99:' in stat_line:
                        results[current_indicator]['execution_time']['p99'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))

        # CPU 사용량 - Go는 System CPU Usage와 Process CPU Usage를 모두 출력함
        if 'System CPU Usage:' in line and current_indicator:
            logger.info(f"{current_indicator} CPU 사용량 파싱")
            for j in range(1, 4):
                if i + j < len(lines):
                    stat_line = lines[i + j]
                    if 'Min:' in stat_line:
                        results[current_indicator]['cpu_usage']['min'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'Max:' in stat_line:
                        results[current_indicator]['cpu_usage']['max'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'Avg:' in stat_line:
                        results[current_indicator]['cpu_usage']['avg'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))

        # 메모리 사용량
        if 'Memory Usage:' in line and current_indicator:
            logger.info(f"{current_indicator} 메모리 사용량 파싱")
            for j in range(1, 4):
                if i + j < len(lines):
                    stat_line = lines[i + j]
                    if 'Min:' in stat_line:
                        results[current_indicator]['memory_usage']['min'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'Max:' in stat_line:
                        results[current_indicator]['memory_usage']['max'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))
                    elif 'Avg:' in stat_line:
                        results[current_indicator]['memory_usage']['avg'] = float(re.search(r'(\d+\.\d+)', stat_line).group(1))

    logger.info("Go 출력 파싱 완료")
    return results

def run_benchmark(command):
    """벤치마크 실행"""
    logger.info(f"벤치마크 실행: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        logger.error(f"벤치마크 실행 중 오류 발생: {stderr}")
        return None

    logger.info("벤치마크 실행 완료")
    return stdout

def print_comparison(python_results, go_results):
    """결과 비교 출력"""
    logger.info("결과 비교 출력 시작")
    print("\n" + "="*80)
    print(" "*25 + "성능 비교 결과")
    print("="*80)

    indicators = ['adx', 'atr', 'bollinger', 'fibonacci']

    for indicator in indicators:
        if indicator not in python_results or indicator not in go_results:
            logger.warning(f"{indicator} 지표가 결과에 없습니다")
            continue

        logger.info(f"{indicator.upper()} 지표 비교 출력")
        print(f"\n{indicator.upper()} 지표 비교:")
        print("-"*60)

        # 실행 시간 비교
        print("\n실행 시간 (초):")
        print(f"{'메트릭':<10} {'Python':>15} {'Go':>15} {'성능 향상':>15}")
        print("-"*60)

        metrics = ['min', 'max', 'avg', 'p95', 'p99']
        for metric in metrics:
            py_val = python_results[indicator]['execution_time'].get(metric, 0)
            go_val = go_results[indicator]['execution_time'].get(metric, 0)
            if go_val > 0:
                speedup = py_val / go_val
                print(f"{metric.upper():<10} {py_val:>15.4f} {go_val:>15.4f} {speedup:>14.2f}x")

        # CPU 사용량 비교
        print("\nCPU 사용률 (%):")
        print(f"{'메트릭':<10} {'Python':>15} {'Go':>15}")
        print("-"*40)

        for metric in ['min', 'max', 'avg']:
            py_val = python_results[indicator]['cpu_usage'].get(metric, 0)
            go_val = go_results[indicator]['cpu_usage'].get(metric, 0)
            print(f"{metric.upper():<10} {py_val:>15.2f} {go_val:>15.2f}")

        # 메모리 사용량 비교
        print("\n메모리 사용량 (MB):")
        print(f"{'메트릭':<10} {'Python':>15} {'Go':>15}")
        print("-"*40)

        for metric in ['min', 'max', 'avg']:
            py_val = python_results[indicator]['memory_usage'].get(metric, 0)
            go_val = go_results[indicator]['memory_usage'].get(metric, 0)
            print(f"{metric.upper():<10} {py_val:>15.2f} {go_val:>15.2f}")
    
    logger.info("결과 비교 출력 완료")

def save_results(python_results, go_results):
    """결과를 JSON 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"

    logger.info(f"결과를 {filename}에 저장 시작")
    results = {
        "timestamp": timestamp,
        "python": python_results,
        "go": go_results
    }

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"결과가 {filename}에 저장되었습니다")
    print(f"\n결과가 {filename}에 저장되었습니다.")

def main():
    logger.info("Python vs Go 벤치마크 비교 시작")
    print("Python vs Go 기술 분석 벤치마크 비교")
    print("="*80)

    # 명령줄 인자 파싱
    iterations = 100
    workers = None
    parallel = True

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('--iterations='):
                iterations = int(arg.split('=')[1])
                logger.info(f"반복 횟수 설정: {iterations}")
            elif arg.startswith('--workers='):
                workers = int(arg.split('=')[1])
                logger.info(f"워커 수 설정: {workers}")
            elif arg == '--traditional':
                parallel = False
                logger.info("전통적 순차 처리 모드 설정")

    # Python 벤치마크 실행
    logger.info("Python 벤치마크 실행 시작")
    print("\nPython 벤치마크 실행 중...")
    python_cmd = ['python', 'benchmark.py', f'--iterations={iterations}']
    if workers:
        python_cmd.append(f'--workers={workers}')
    if not parallel:
        python_cmd.append('--traditional')

    python_output = run_benchmark(python_cmd)
    if python_output is None:
        logger.error("Python 벤치마크 실행 실패")
        print("Python 벤치마크 실행 실패")
        return

    # Go 벤치마크 실행
    logger.info("Go 벤치마크 실행 시작")
    print("\nGo 벤치마크 실행 중...")

    # Go 빌드 확인
    if not os.path.exists('./benchmark'):
        logger.info("Go 바이너리 빌드 시작")
        print("Go 바이너리를 빌드합니다...")
        build_result = subprocess.run(['go', 'build', 'benchmark.go'], capture_output=True)
        if build_result.returncode != 0:
            error_msg = build_result.stderr.decode()
            logger.error(f"Go 빌드 실패: {error_msg}")
            print(f"Go 빌드 실패: {error_msg}")
            return
        logger.info("Go 바이너리 빌드 완료")

    go_cmd = ['./benchmark', f'-iterations={iterations}']
    if workers:
        go_cmd.append(f'-workers={workers}')
    if not parallel:
        go_cmd.append('-traditional')

    go_output = run_benchmark(go_cmd)
    if go_output is None:
        logger.error("Go 벤치마크 실행 실패")
        print("Go 벤치마크 실행 실패")
        return

    # 결과 파싱
    logger.info("벤치마크 결과 파싱 시작")
    python_results = parse_python_output(python_output)
    go_results = parse_go_output(go_output)

    # 결과 비교 출력
    print_comparison(python_results, go_results)

    # 결과 저장
    save_results(python_results, go_results)
    
    logger.info("벤치마크 비교 완료")

if __name__ == "__main__":
    main()