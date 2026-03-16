#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}   Polars vs Spark: Stratified Sampling Performance Test${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Create logs directory
mkdir -p logs

# Log files
POLARS_LOG="logs/polars_run.log"
SPARK_LOG="logs/spark_run.log"
COMPARISON_LOG="logs/comparison_results.txt"

# Clear previous logs
> "$POLARS_LOG"
> "$SPARK_LOG"
> "$COMPARISON_LOG"

# ============================================
# RUN POLARS METHOD
# ============================================
echo -e "${YELLOW}[1/2] Running POLARS method...${NC}"
echo "Starting at: $(date)"
echo ""
source /Users/eugene/Documents/Task/SSG/.venv/bin/activate
POLARS_START=$(date +%s)
python main.py 2>&1 | tee "$POLARS_LOG"
POLARS_EXIT_CODE=$?
POLARS_END=$(date +%s)
POLARS_ELAPSED=$((POLARS_END - POLARS_START))

if [ $POLARS_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Polars completed successfully${NC}"
else
    echo -e "${RED}✗ Polars failed with exit code $POLARS_EXIT_CODE${NC}"
fi

echo ""
echo "Polars finished at: $(date)"
echo "================================================================"
echo ""
sleep 2

# ============================================
# RUN SPARK METHOD
# ============================================
echo -e "${YELLOW}[2/2] Running SPARK method...${NC}"
echo "Starting at: $(date)"
echo ""

SPARK_START=$(date +%s)
python main_spark.py 2>&1 | tee "$SPARK_LOG"
SPARK_EXIT_CODE=$?
SPARK_END=$(date +%s)
SPARK_ELAPSED=$((SPARK_END - SPARK_START))

if [ $SPARK_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Spark completed successfully${NC}"
else
    echo -e "${RED}✗ Spark failed with exit code $SPARK_EXIT_CODE${NC}"
fi

echo ""
echo "Spark finished at: $(date)"
echo "================================================================"
echo ""
sleep 1

# ============================================
# EXTRACT TIMING INFORMATION
# ============================================
echo -e "${BLUE}Extracting timing information...${NC}"

# Extract from Polars log
POLARS_TIME=$(grep "POLARS TOTAL TIME" "$POLARS_LOG" | grep -oE '[0-9]+\.[0-9]+' | head -1)
POLARS_MINUTES=$(grep "POLARS TOTAL TIME" "$POLARS_LOG" | grep -oE '\([0-9]+\.[0-9]+ minutes\)' | grep -oE '[0-9]+\.[0-9]+')
POLARS_ROWS=$(grep "Total rows processed:" "$POLARS_LOG" | grep -oE '[0-9,]+' | tr -d ',')
POLARS_SAMPLED=$(grep "Total rows sampled:" "$POLARS_LOG" | grep -oE '[0-9,]+' | tr -d ',')

# Extract from Spark log
SPARK_TIME=$(grep "SPARK TOTAL TIME" "$SPARK_LOG" | grep -oE '[0-9]+\.[0-9]+' | head -1)
SPARK_MINUTES=$(grep "SPARK TOTAL TIME" "$SPARK_LOG" | grep -oE '\([0-9]+\.[0-9]+ minutes\)' | grep -oE '[0-9]+\.[0-9]+')
SPARK_ROWS=$(grep "Total rows:" "$SPARK_LOG" | grep -oE '[0-9,]+' | tr -d ',' | head -1)
SPARK_SAMPLED=$(grep "Sampled rows:" "$SPARK_LOG" | grep -oE '[0-9,]+' | tr -d ',' | head -1)

# ============================================
# GENERATE COMPARISON REPORT
# ============================================
echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}                    PERFORMANCE COMPARISON${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

{
    echo "PERFORMANCE COMPARISON REPORT"
    echo "Generated at: $(date)"
    echo "================================================================"
    echo ""
    echo "POLARS (Sequential Processing)"
    echo "------------------------------"
    echo "  Total Time:      ${POLARS_TIME:-N/A} seconds (${POLARS_MINUTES:-N/A} minutes)"
    echo "  Rows Processed:  ${POLARS_ROWS:-N/A}"
    echo "  Rows Sampled:    ${POLARS_SAMPLED:-N/A}"
    echo "  Exit Code:       $POLARS_EXIT_CODE"
    echo ""
    echo "SPARK (Distributed Processing)"
    echo "------------------------------"
    echo "  Total Time:      ${SPARK_TIME:-N/A} seconds (${SPARK_MINUTES:-N/A} minutes)"
    echo "  Rows Processed:  ${SPARK_ROWS:-N/A}"
    echo "  Rows Sampled:    ${SPARK_SAMPLED:-N/A}"
    echo "  Exit Code:       $SPARK_EXIT_CODE"
    echo ""
    echo "COMPARISON"
    echo "------------------------------"

    if [ -n "$POLARS_TIME" ] && [ -n "$SPARK_TIME" ]; then
        SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $POLARS_TIME / $SPARK_TIME}")
        TIME_SAVED=$(awk "BEGIN {printf \"%.2f\", $POLARS_TIME - $SPARK_TIME}")
        PERCENT_FASTER=$(awk "BEGIN {printf \"%.1f\", (($POLARS_TIME - $SPARK_TIME) / $POLARS_TIME) * 100}")

        echo "  Speedup:         ${SPEEDUP}x"
        echo "  Time Saved:      ${TIME_SAVED} seconds"
        echo "  Spark is:        ${PERCENT_FASTER}% faster"

        if (( $(echo "$SPARK_TIME < $POLARS_TIME" | bc -l) )); then
            echo "  Winner:          🏆 SPARK"
        else
            echo "  Winner:          🏆 POLARS"
        fi
    else
        echo "  Unable to calculate comparison (missing timing data)"
    fi

    echo ""
    echo "================================================================"
} | tee "$COMPARISON_LOG"

# Display the comparison on console
cat "$COMPARISON_LOG"

echo ""
echo -e "${GREEN}Full logs saved to:${NC}"
echo -e "  Polars:      ${POLARS_LOG}"
echo -e "  Spark:       ${SPARK_LOG}"
echo -e "  Comparison:  ${COMPARISON_LOG}"
echo ""

# Check output files
echo -e "${BLUE}Output Files:${NC}"
if [ -f "data/sampled_jobs.parquet" ]; then
    POLARS_SIZE=$(du -h "data/sampled_jobs.parquet" | cut -f1)
    echo -e "  ${GREEN}✓${NC} Polars output:  data/sampled_jobs.parquet (${POLARS_SIZE})"
else
    echo -e "  ${RED}✗${NC} Polars output:  data/sampled_jobs.parquet (not found)"
fi

if [ -f "data/sampled_jobs_spark.parquet" ]; then
    SPARK_SIZE=$(du -h "data/sampled_jobs_spark.parquet" | cut -f1)
    echo -e "  ${GREEN}✓${NC} Spark output:   data/sampled_jobs_spark.parquet (${SPARK_SIZE})"
else
    echo -e "  ${RED}✗${NC} Spark output:   data/sampled_jobs_spark.parquet (not found)"
fi

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}Comparison complete!${NC}"
echo -e "${BLUE}================================================================${NC}"
