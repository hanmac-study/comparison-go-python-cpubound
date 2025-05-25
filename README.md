# Technical Analysis Benchmark: Python vs Go

이 프로젝트는 Python과 Go로 구현된 기술적 지표 계산의 성능을 비교합니다.

## 구성 요소

- `benchmark.py`: Python 구현
- `benchmark.go`: Go 구현
- `go.mod`: Go 모듈 파일
- `.env`: 데이터베이스 연결 정보
- `run_comparison.py`: 자동 비교 스크립트

## 벤치마크 설계

### DB 호출 최소화
- 벤치마크 시작 시 모든 데이터를 한 번에 메모리로 로드
- 이후 모든 반복 실행에서는 메모리에 있는 데이터 사용
- 순수하게 지표 계산 알고리즘의 성능만 측정

### 측정 방식
- 100회 반복 실행으로 통계적 신뢰성 확보
- 각 실행마다 CPU 및 메모리 사용량 측정
- p95, p99 백분위수로 이상치 영향 최소화

## 필요 사항

### Python
```bash
pip install pandas numpy talib psycopg2-binary sqlalchemy python-dotenv psutil
```

### Go
```bash
go mod download
```

## 실행 방법

### Python 벤치마크
```bash
# 병렬 처리 (기본값)
python benchmark.py

# 순차 처리
python benchmark.py --traditional

# 워커 수 지정
python benchmark.py --workers 4

# 반복 횟수 지정 (기본값: 100)
python benchmark.py --iterations 50
```

### Go 벤치마크
```bash
# 먼저 빌드
go build benchmark.go

# 병렬 처리 (기본값)
./benchmark

# 순차 처리
./benchmark -traditional

# 워커 수 지정
./benchmark -workers 4

# 반복 횟수 지정 (기본값: 100)
./benchmark -iterations 50
```

### 자동 비교 실행
```bash
# 두 언어의 벤치마크를 자동으로 실행하고 비교
python run_comparison.py

# 커스텀 설정
python run_comparison.py --iterations=50 --workers=4 --traditional
```

## 측정 항목

### 지표 종류
- **ADX (Average Directional Index)**: 트렌드 강도 측정
- **ATR (Average True Range)**: 변동성 측정
- **Bollinger Bands**: 가격 밴드 계산

### 성능 메트릭
- **실행 시간**
  - 최소값 (Min)
  - 최대값 (Max)
  - 평균값 (Avg)
  - 95 백분위수 (P95)
  - 99 백분위수 (P99)

- **리소스 사용량**
  - CPU 사용률 (%)
  - 메모리 사용량 (MB)

## 벤치마크 데이터

10개의 암호화폐 티커에 대해 각 지표를 계산:
- KRW-BTC, KRW-ETH, KRW-ETC, KRW-GRS, KRW-ICX
- KRW-XRP, KRW-WAVES, KRW-QTUM, KRW-MTL, KRW-ADA

각 티커당 200개 기간(약 12,000개 캔들)의 데이터를 처리합니다.

## 예상 결과

일반적으로 Go 구현이 Python 구현보다 빠른 성능을 보입니다:
- **순차 처리**: Go가 3-5배 빠름
- **병렬 처리**: Go가 2-3배 빠름
- **메모리 사용**: Go가 더 적은 메모리 사용

하지만 Python은 TA-Lib 같은 최적화된 라이브러리를 사용하므로, 복잡한 지표의 경우 성능 차이가 줄어들 수 있습니다.

## 주의사항

1. **데이터베이스 연결**: `.env` 파일에 올바른 DB 정보가 있어야 합니다.
2. **TA-Lib 설치**: Python에서 TA-Lib은 시스템 라이브러리가 필요합니다:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ta-lib
   
   # macOS
   brew install ta-lib
   ```
3. **충분한 데이터**: 테이블에 충분한 데이터가 있어야 정확한 지표 계산이 가능합니다.
4. **메모리 요구사항**: 모든 데이터를 메모리에 로드하므로 충분한 RAM이 필요합니다 (약 1-2GB).

## 결과 해석

- **실행 시간**: 낮을수록 좋음
- **P95/P99**: 평균보다 안정성을 잘 나타냄 (이상치 제외)
- **CPU 사용률**: 병렬 처리 시 높은 것이 정상
- **메모리 사용량**: 지표 계산 중 추가로 사용된 메모리량