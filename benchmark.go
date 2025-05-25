package main

import (
	"database/sql"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/joho/godotenv"
	_ "github.com/lib/pq"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/process"
)

// 벤치마크 티커 목록
var BENCHMARK_TICKERS = []string{
	"KRW-BTC", "KRW-ETH", "KRW-ETC", "KRW-GRS", "KRW-ICX",
	"KRW-XRP", "KRW-WAVES", "KRW-QTUM", "KRW-MTL", "KRW-ADA",
}

// CandleData 구조체
type CandleData struct {
	Timestamp     time.Time
	Code          string
	TradePrice    float64
	OpeningPrice  float64
	HighPrice     float64
	LowPrice      float64
	AccVolume     float64
	AccTradePrice float64
}

// 지표 결과 구조체
type ADXResult struct {
	Timestamp           time.Time
	ADX                 float64
	PlusDI              float64
	MinusDI             float64
	DIDeference         float64
	DISum               float64
	TrendStrength       int
	DirectionalMovement int
}

type ATRResult struct {
	Timestamp  time.Time
	ATR7       float64
	ATR14      float64
	ATR21      float64
	ATR7Ratio  float64
	ATR14Ratio float64
	ATR21Ratio float64
	TrueRange  float64
	HLRange    float64
	HCRange    float64
	LCRange    float64
}

type BollingerResult struct {
	Timestamp               time.Time
	Upper                   float64
	Middle                  float64
	Lower                   float64
	Width                   float64
	WidthRatio              float64
	Position                float64
	PriceAboveMiddle        bool
	PriceDistanceFromMiddle float64
	TouchUpper              bool
	TouchLower              bool
	BreakUpper              bool
	BreakLower              bool
}

type FibonacciResult struct {
	Timestamp       time.Time
	Levels          []float64
	RetracePoints   []float64
	ExtensionPoints []float64
	TrendDirection  int     // 1: 상승, -1: 하락, 0: 중립
	StrengthScore   float64 // 피보나치 지표 강도
}

// 데이터베이스 연결
func getDB() (*sql.DB, error) {
	// .env 파일 로드
	err := godotenv.Load()
	if err != nil {
		log.Printf("Error loading .env file: %v", err)
	}

	// DATABASE_URL이 있으면 우선 사용
	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL != "" {
		db, err := sql.Open("postgres", databaseURL)
		if err != nil {
			return nil, err
		}

		// 연결 테스트
		err = db.Ping()
		if err != nil {
			return nil, err
		}

		return db, nil
	}

	// 개별 환경 변수 사용
	dbHost := os.Getenv("DB_HOST")
	dbPort := os.Getenv("DB_PORT")
	dbName := os.Getenv("DB_NAME")
	dbUser := os.Getenv("DB_USER")
	dbPassword := os.Getenv("DB_PASSWORD")
	dbSSLMode := os.Getenv("DB_SSLMODE")

	// 연결 문자열 생성
	connStr := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=%s",
		dbHost, dbPort, dbUser, dbPassword, dbName, dbSSLMode)

	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, err
	}

	// 연결 테스트
	err = db.Ping()
	if err != nil {
		return nil, err
	}

	return db, nil
}

// 티커 데이터 가져오기
func getTickerData(db *sql.DB, ticker string, period int) ([]CandleData, error) {
	query := `
		SELECT timestamp, code, trade_price, opening_price, high_price, low_price, 
		       candle_acc_trade_volume, candle_acc_trade_price
		FROM upbit_candle_1m
		WHERE code = $1
		ORDER BY timestamp
		LIMIT $2
	`

	rows, err := db.Query(query, ticker, period*60)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var candles []CandleData
	for rows.Next() {
		var c CandleData
		err := rows.Scan(&c.Timestamp, &c.Code, &c.TradePrice, &c.OpeningPrice,
			&c.HighPrice, &c.LowPrice, &c.AccVolume, &c.AccTradePrice)
		if err != nil {
			return nil, err
		}
		candles = append(candles, c)
	}

	return candles, nil
}

// ADX 계산
func calculateADX(candles []CandleData, period int) []ADXResult {
	if len(candles) < period+1 {
		return []ADXResult{}
	}

	results := make([]ADXResult, len(candles))

	// True Range 계산
	tr := make([]float64, len(candles))
	for i := 1; i < len(candles); i++ {
		h := candles[i].HighPrice
		l := candles[i].LowPrice
		pc := candles[i-1].TradePrice

		hl := h - l
		hc := math.Abs(h - pc)
		lc := math.Abs(l - pc)

		tr[i] = math.Max(hl, math.Max(hc, lc))
	}

	// Directional Movement 계산
	plusDM := make([]float64, len(candles))
	minusDM := make([]float64, len(candles))

	for i := 1; i < len(candles); i++ {
		upMove := candles[i].HighPrice - candles[i-1].HighPrice
		downMove := candles[i-1].LowPrice - candles[i].LowPrice

		if upMove > downMove && upMove > 0 {
			plusDM[i] = upMove
		}
		if downMove > upMove && downMove > 0 {
			minusDM[i] = downMove
		}
	}

	// ATR, +DI, -DI 계산
	atr := sma(tr, period)
	plusDI := make([]float64, len(candles))
	minusDI := make([]float64, len(candles))

	for i := period; i < len(candles); i++ {
		if atr[i] > 0 {
			plusDI[i] = 100 * sma(plusDM, period)[i] / atr[i]
			minusDI[i] = 100 * sma(minusDM, period)[i] / atr[i]
		}
	}

	// DX 계산
	dx := make([]float64, len(candles))
	for i := period; i < len(candles); i++ {
		diSum := plusDI[i] + minusDI[i]
		if diSum > 0 {
			dx[i] = 100 * math.Abs(plusDI[i]-minusDI[i]) / diSum
		}
	}

	// ADX 계산 (DX의 이동평균)
	adx := sma(dx, period)

	// 결과 생성
	for i := 0; i < len(candles); i++ {
		r := ADXResult{
			Timestamp:   candles[i].Timestamp,
			ADX:         adx[i],
			PlusDI:      plusDI[i],
			MinusDI:     minusDI[i],
			DIDeference: plusDI[i] - minusDI[i],
			DISum:       plusDI[i] + minusDI[i],
		}

		// Trend Strength
		if r.ADX < 25 {
			r.TrendStrength = 0
		} else if r.ADX < 50 {
			r.TrendStrength = 1
		} else if r.ADX < 75 {
			r.TrendStrength = 2
		} else {
			r.TrendStrength = 3
		}

		// Directional Movement
		if r.PlusDI > r.MinusDI*1.2 {
			r.DirectionalMovement = 1
		} else if r.MinusDI > r.PlusDI*1.2 {
			r.DirectionalMovement = -1
		}

		results[i] = r
	}

	return results
}

// ATR 계산
func calculateATR(candles []CandleData, periods []int) []ATRResult {
	if len(candles) < 2 {
		return []ATRResult{}
	}

	results := make([]ATRResult, len(candles))

	// True Range 계산
	tr := make([]float64, len(candles))
	for i := 1; i < len(candles); i++ {
		h := candles[i].HighPrice
		l := candles[i].LowPrice
		pc := candles[i-1].TradePrice

		hl := h - l
		hc := math.Abs(h - pc)
		lc := math.Abs(l - pc)

		tr[i] = math.Max(hl, math.Max(hc, lc))

		results[i].TrueRange = tr[i]
		results[i].HLRange = hl
		results[i].HCRange = hc
		results[i].LCRange = lc
	}

	// 각 기간별 ATR 계산
	atr7 := sma(tr, 7)
	atr14 := sma(tr, 14)
	atr21 := sma(tr, 21)

	for i := 0; i < len(candles); i++ {
		results[i].Timestamp = candles[i].Timestamp
		results[i].ATR7 = atr7[i]
		results[i].ATR14 = atr14[i]
		results[i].ATR21 = atr21[i]

		if candles[i].TradePrice > 0 {
			results[i].ATR7Ratio = atr7[i] / candles[i].TradePrice
			results[i].ATR14Ratio = atr14[i] / candles[i].TradePrice
			results[i].ATR21Ratio = atr21[i] / candles[i].TradePrice
		}
	}

	return results
}

// 볼린저 밴드 계산
func calculateBollinger(candles []CandleData, period int, stdMultiplier float64) []BollingerResult {
	if len(candles) < period {
		return []BollingerResult{}
	}

	results := make([]BollingerResult, len(candles))

	// 가격 추출
	prices := make([]float64, len(candles))
	for i, c := range candles {
		prices[i] = c.TradePrice
	}

	// 이동평균 계산
	ma := sma(prices, period)

	// 표준편차 계산
	for i := period - 1; i < len(candles); i++ {
		// 표준편차 계산
		sum := 0.0
		for j := i - period + 1; j <= i; j++ {
			diff := prices[j] - ma[i]
			sum += diff * diff
		}
		std := math.Sqrt(sum / float64(period))

		// 밴드 계산
		upper := ma[i] + stdMultiplier*std
		lower := ma[i] - stdMultiplier*std
		middle := ma[i]
		width := upper - lower

		r := BollingerResult{
			Timestamp: candles[i].Timestamp,
			Upper:     upper,
			Middle:    middle,
			Lower:     lower,
			Width:     width,
		}

		if middle > 0 {
			r.WidthRatio = width / middle
		}

		if width > 0 {
			r.Position = (prices[i] - lower) / width
		}

		r.PriceAboveMiddle = prices[i] > middle

		if middle > 0 {
			r.PriceDistanceFromMiddle = (prices[i] - middle) / middle
		}

		// 터치 및 돌파
		r.TouchUpper = math.Abs(prices[i]-upper)/upper < 0.001
		r.TouchLower = math.Abs(prices[i]-lower)/lower < 0.001
		r.BreakUpper = prices[i] > upper
		r.BreakLower = prices[i] < lower

		results[i] = r
	}

	return results
}

// 피보나치 레벨 계산
func calculateFibonacci(candles []CandleData, period int) []FibonacciResult {
	if len(candles) < period {
		return []FibonacciResult{}
	}

	results := make([]FibonacciResult, len(candles))
	fibNumbers := generateFibonacciNumbers(20) // 계산 부하를 높이기 위해 깊은 수준까지 계산

	// 각 캔들마다 계산
	for i := period; i < len(candles); i++ {
		// 피보나치 계산 범위 설정 (계산 부하를 높이기 위해 큰 범위 사용)
		start := i - period
		end := i

		// 고가와 저가 찾기
		high := candles[start].HighPrice
		low := candles[start].LowPrice
		for j := start + 1; j <= end; j++ {
			if candles[j].HighPrice > high {
				high = candles[j].HighPrice
			}
			if candles[j].LowPrice < low {
				low = candles[j].LowPrice
			}
		}

		// 상승 또는 하락 추세 결정 (계산 부하를 위해 복잡한 계산 추가)
		direction := 0
		sumStart := 0.0
		sumEnd := 0.0
		for j := 0; j < period/2; j++ {
			sumStart += candles[start+j].TradePrice
			sumEnd += candles[end-j].TradePrice
		}
		avgStart := sumStart / float64(period/2)
		avgEnd := sumEnd / float64(period/2)

		if avgEnd > avgStart {
			direction = 1
		} else if avgEnd < avgStart {
			direction = -1
		}

		// 피보나치 레벨 계산 (계산 부하를 위해 많은 레벨 계산)
		diff := high - low
		levels := make([]float64, len(fibNumbers))
		for j := 0; j < len(fibNumbers); j++ {
			if direction >= 0 {
				levels[j] = low + diff*fibNumbers[j]
			} else {
				levels[j] = high - diff*fibNumbers[j]
			}
		}

		// 리트레이스먼트 포인트 계산 (추가적인 계산 부하)
		retracePoints := make([]float64, period)
		for j := 0; j < period; j++ {
			priceRatio := (candles[end-j].TradePrice - low) / diff
			closestLevel := 0.0
			minDiff := math.MaxFloat64

			// 가장 가까운 피보나치 레벨 찾기 (계산 부하를 위한 반복문)
			for k := 0; k < len(levels); k++ {
				levelRatio := fibNumbers[k]
				if math.Abs(priceRatio-levelRatio) < minDiff {
					minDiff = math.Abs(priceRatio - levelRatio)
					closestLevel = levelRatio
				}
			}
			retracePoints[j] = closestLevel
		}

		// 확장 포인트 계산 (추가적인 계산 부하)
		extensionPoints := make([]float64, period)
		for j := 0; j < period; j++ {
			// 복잡한 계산 로직 추가 (CPU 부하 증가용)
			base := (high + low) / 2
			amplitude := (high - low) / 2
			angle := 2 * math.Pi * float64(j) / float64(period)
			
			// 사인, 코사인 함수 사용하여 계산 부하 증가
			value := base + amplitude*math.Sin(angle)
			
			// 피보나치 수열의 값을 가중치로 사용
			weight := fibNumbers[j%len(fibNumbers)]
			
			extensionPoints[j] = value * weight
		}

		// 강도 점수 계산 (복잡한 계산 추가)
		strengthScore := 0.0
		for j := 0; j < period; j++ {
			// CPU 연산을 위한 복잡한 계산
			price := candles[end-j].TradePrice
			distance := 0.0
			
			for k := 0; k < len(levels); k++ {
				levelDistance := math.Abs(price-levels[k]) / levels[k]
				weight := math.Exp(-levelDistance * 10) // 지수 함수로 가중치 계산
				distance += weight * fibNumbers[k]
			}
			
			strengthScore += distance
		}
		strengthScore /= float64(period)

		// 결과 저장
		results[i] = FibonacciResult{
			Timestamp:       candles[i].Timestamp,
			Levels:          levels,
			RetracePoints:   retracePoints,
			ExtensionPoints: extensionPoints,
			TrendDirection:  direction,
			StrengthScore:   strengthScore,
		}
	}

	return results
}

// 피보나치 수열 생성 (고의적으로 재귀 대신 반복문 사용하여 효율적으로 구현)
func generateFibonacciNumbers(count int) []float64 {
	fib := make([]float64, count)
	
	// 기본 피보나치 비율
	baseRatios := []float64{0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618, 3.618, 4.236}
	
	// 피보나치 수열 계산 (추가 연산을 위해 많은 수 계산)
	a, b := 0.0, 1.0
	fibNumbers := make([]float64, 100)
	fibNumbers[0] = a
	fibNumbers[1] = b
	
	for i := 2; i < 100; i++ {
		c := a + b
		fibNumbers[i] = c
		a, b = b, c
	}
	
	// 기본 비율과 계산된 수열을 조합하여 더 많은 레벨 생성
	for i := 0; i < count; i++ {
		if i < len(baseRatios) {
			fib[i] = baseRatios[i]
		} else {
			idx := i % len(fibNumbers)
			normalizedValue := fibNumbers[idx] / fibNumbers[99] // 정규화
			fib[i] = normalizedValue
		}
	}
	
	return fib
}

// Simple Moving Average
func sma(data []float64, period int) []float64 {
	result := make([]float64, len(data))

	for i := period - 1; i < len(data); i++ {
		sum := 0.0
		for j := i - period + 1; j <= i; j++ {
			sum += data[j]
		}
		result[i] = sum / float64(period)
	}

	return result
}

// 모든 티커 데이터를 한 번에 로드
func loadAllData(db *sql.DB) (map[string][]CandleData, error) {
	fmt.Println("Loading data from database...")
	allData := make(map[string][]CandleData)

	for _, ticker := range BENCHMARK_TICKERS {
		candles, err := getTickerData(db, ticker, 30000)
		if err != nil {
			log.Printf("Error loading data for %s: %v", ticker, err)
			continue
		}
		if len(candles) > 0 {
			allData[ticker] = candles
		} else {
			log.Printf("Warning: No data for %s", ticker)
		}
	}

	fmt.Printf("Loaded data for %d tickers\n", len(allData))
	return allData, nil
}

// 메모리에서 데이터로 지표 처리
func processTickerFromMemory(ticker string, candles []CandleData, indicatorType string) bool {
	var processed bool
	var start time.Time
	var duration time.Duration

	switch indicatorType {
	case "adx":
		start = time.Now()
		adxResult := calculateADX(candles, 14)
		duration = time.Since(start)
		processed = len(adxResult) > 0
	case "atr":
		start = time.Now()
		atrResult := calculateATR(candles, []int{7, 14, 21})
		duration = time.Since(start)
		processed = len(atrResult) > 0
	case "bollinger":
		start = time.Now()
		bollingerResult := calculateBollinger(candles, 20, 2.0)
		duration = time.Since(start)
		processed = len(bollingerResult) > 0
	case "fibonacci":
		start = time.Now()
		fibonacciResult := calculateFibonacci(candles, 30)
		duration = time.Since(start)
		processed = len(fibonacciResult) > 0
	default:
		return false
	}

	if processed {
		log.Printf("Processed %s - %s in %v", ticker, indicatorType, duration)
	}
	return processed
}

// 메모리 데이터로 전통적인 순차 처리
func processTraditionalFromMemory(allData map[string][]CandleData, indicatorType string) (time.Duration, int) {
	start := time.Now()
	success := 0

	for ticker, candles := range allData {
		if processTickerFromMemory(ticker, candles, indicatorType) {
			success++
		}
	}

	elapsed := time.Since(start)
	return elapsed, success
}

// 메모리 데이터로 병렬 처리
func processParallelFromMemory(allData map[string][]CandleData, indicatorType string, workers int) (time.Duration, int) {
	start := time.Now()
	success := 0
	successMutex := sync.Mutex{}

	type workItem struct {
		ticker  string
		candles []CandleData
	}

	var wg sync.WaitGroup
	ch := make(chan workItem, len(allData))

	// 워커 시작
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range ch {
				if processTickerFromMemory(item.ticker, item.candles, indicatorType) {
					successMutex.Lock()
					success++
					successMutex.Unlock()
				}
			}
		}()
	}

	// 작업 분배
	for ticker, candles := range allData {
		ch <- workItem{
			ticker:  ticker,
			candles: candles,
		}
	}
	close(ch)

	wg.Wait()

	elapsed := time.Since(start)
	return elapsed, success
}

// 통계 계산
func calculateStats(durations []float64) (min, max, avg, p95, p99 float64) {
	if len(durations) == 0 {
		return
	}

	sort.Float64s(durations)

	min = durations[0]
	max = durations[len(durations)-1]

	sum := 0.0
	for _, d := range durations {
		sum += d
	}
	avg = sum / float64(len(durations))

	p95Index := int(float64(len(durations)) * 0.95)
	p99Index := int(float64(len(durations)) * 0.99)

	if p95Index < len(durations) {
		p95 = durations[p95Index]
	}
	if p99Index < len(durations) {
		p99 = durations[p99Index]
	}

	return
}

// 시스템 리소스 측정
type ResourceUsage struct {
	CPUPercent       float64
	MemoryMB         float64
	ProcessCPUUser   float64
	ProcessCPUSystem float64
	ProcessCPUTotal  float64
}

// 시스템 리소스 측정
func measureResources() ResourceUsage {
	v, _ := mem.VirtualMemory()
	cpuPercent, _ := cpu.Percent(0, false)
	
	// 프로세스별 CPU 사용량 측정
	pid := os.Getpid()
	proc, err := process.NewProcess(int32(pid))
	procCPUUser := 0.0
	procCPUSystem := 0.0
	procCPUTotal := 0.0
	
	if err == nil {
		times, err := proc.Times()
		if err == nil {
			procCPUUser = times.User
			procCPUSystem = times.System
			procCPUTotal = times.User + times.System
		}
	}
	
	cpuPercentVal := 0.0
	if len(cpuPercent) > 0 {
		cpuPercentVal = cpuPercent[0]
	}

	return ResourceUsage{
		CPUPercent:       cpuPercentVal,
		MemoryMB:         float64(v.Used) / 1024 / 1024,
		ProcessCPUUser:   procCPUUser,
		ProcessCPUSystem: procCPUSystem,
		ProcessCPUTotal:  procCPUTotal,
	}
}

// 벤치마크 실행
func runBenchmark(parallel bool, workers int, iterations int) {
	fmt.Printf("\n=== Go Implementation Performance ===\n")
	fmt.Printf("Workers: %d, Parallel mode: %v, Iterations: %d\n\n", workers, parallel, iterations)

	// DB 연결 - 한 번만
	db, err := getDB()
	if err != nil {
		log.Fatal("Failed to connect to database:", err)
	}
	defer db.Close()

	// 모든 데이터를 한 번에 로드
	allData, err := loadAllData(db)
	if err != nil || len(allData) == 0 {
		log.Fatal("Failed to load data:", err)
	}

	indicators := []string{"adx", "atr", "bollinger", "fibonacci"}

	for _, indicator := range indicators {
		var durations []float64
		var cpuUsages []float64
		var memUsages []float64
		var processCPUUsages []float64  // 프로세스 CPU 사용량

		fmt.Printf("\n--- %s ---\n", indicator)

		for i := 0; i < iterations; i++ {
			// 리소스 측정 시작
			startResources := measureResources()

			// 실행
			var elapsed time.Duration
			var success int
			if parallel {
				elapsed, success = processParallelFromMemory(allData, indicator, workers)
			} else {
				elapsed, success = processTraditionalFromMemory(allData, indicator)
			}

			// 리소스 측정 종료
			endResources := measureResources()

			durations = append(durations, elapsed.Seconds())
			cpuUsages = append(cpuUsages, endResources.CPUPercent-startResources.CPUPercent)
			memUsages = append(memUsages, endResources.MemoryMB-startResources.MemoryMB)
			
			// 프로세스 CPU 사용량 계산
			processCPUTime := endResources.ProcessCPUTotal - startResources.ProcessCPUTotal
			processCPUPercent := (processCPUTime / elapsed.Seconds()) * 100
			processCPUUsages = append(processCPUUsages, processCPUPercent)

			// 진행 상황 표시 (10회마다)
			if (i+1)%10 == 0 {
				fmt.Printf("Progress: %d/%d - Last execution: %.4fs (%d/%d tickers)\n",
					i+1, iterations, elapsed.Seconds(), success, len(allData))
			}
		}

		// 통계 계산 및 출력
		min, max, avg, p95, p99 := calculateStats(durations)
		fmt.Printf("\nExecution Time Statistics:\n")
		fmt.Printf("  Min: %.4f seconds\n", min)
		fmt.Printf("  Max: %.4f seconds\n", max)
		fmt.Printf("  Avg: %.4f seconds\n", avg)
		fmt.Printf("  P95: %.4f seconds\n", p95)
		fmt.Printf("  P99: %.4f seconds\n", p99)

		cpuMin, cpuMax, cpuAvg, _, _ := calculateStats(cpuUsages)
		fmt.Printf("\nSystem CPU Usage:\n")
		fmt.Printf("  Min: %.2f%%\n", cpuMin)
		fmt.Printf("  Max: %.2f%%\n", cpuMax)
		fmt.Printf("  Avg: %.2f%%\n", cpuAvg)
		
		procCPUMin, procCPUMax, procCPUAvg, _, _ := calculateStats(processCPUUsages)
		fmt.Printf("\nProcess CPU Usage:\n")
		fmt.Printf("  Min: %.2f%%\n", procCPUMin)
		fmt.Printf("  Max: %.2f%%\n", procCPUMax)
		fmt.Printf("  Avg: %.2f%%\n", procCPUAvg)

		memMin, memMax, memAvg, _, _ := calculateStats(memUsages)
		fmt.Printf("\nMemory Usage:\n")
		fmt.Printf("  Min: %.2f MB\n", memMin)
		fmt.Printf("  Max: %.2f MB\n", memMax)
		fmt.Printf("  Avg: %.2f MB\n", memAvg)
	}
}

func main() {
	traditional := flag.Bool("traditional", false, "Use traditional (non-parallel) processing")
	workers := flag.Int("workers", runtime.NumCPU(), "Number of workers")
	iterations := flag.Int("iterations", 100, "Number of iterations")
	flag.Parse()

	runBenchmark(!*traditional, *workers, *iterations)
}
