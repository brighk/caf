# Why You Got 0% Accuracy (And How to Fix It)

## What Happened

Your experiment ran but got **0% accuracy** with all answers as "Unknown". This happened because:

### The Bug

There was a code bug in `run_counterbench_experiment.py`:
```python
# Line 172 - WRONG attribute name
'intermediate_scores': caf_result.intermediate_scores  # ❌ This doesn't exist!
```

The CAFOutput object has `iteration_logs`, not `intermediate_scores`. This caused:
- Every example to crash with error: `'CAFOutput' object has no attribute 'intermediate_scores'`
- CAF loop never ran (iterations: 0)
- All answers marked as "Unknown"
- 0% accuracy

### The Fix

I've fixed the code:
```python
# Now uses correct attribute
'iteration_logs': [log for log in caf_result.iteration_logs]  # ✅ Correct!
```

## What This Means

🔴 **First run (0% accuracy)**: Bug prevented CAF from running
🟢 **New run (in progress)**: Should work correctly now

The SPARQL errors you saw were fine - the real problem was this code bug.

## Next Steps

### Option 1: Wait for Current Test (Recommended)

A test with 3 examples is running now. Wait ~5 minutes, then check:
```bash
cat results/tiny_test_fixed/report.txt
```

If you see accuracy > 0%, the fix worked!

### Option 2: Run Full Test Again

```bash
cd /home/bright/projects/PhD/CAF
source venv/bin/activate

# Run with 10 examples
python -m experiments.run_counterbench_experiment \
    --input data/counterbench.json \
    --limit 10 \
    --use-llm \
    --llm-model tiny \
    --llm-4bit \
    --use-real-sparql \
    --sparql-endpoint http://localhost:3030/counterbench/query \
    --extract-kb \
    --output results/tiny_fixed

# Check results after ~10 minutes
cat results/tiny_fixed/report.txt
```

### Option 3: Simpler Test Without SPARQL First

To isolate the issue, test without SPARQL:
```bash
# LLM-only mode (no SPARQL complexity)
python -m experiments.run_counterbench_experiment \
    --input data/counterbench.json \
    --limit 5 \
    --use-llm \
    --llm-model tiny \
    --llm-4bit \
    --output results/llm_only_test

# Should get ~40-50% accuracy if working
cat results/llm_only_test/report.txt
```

## Understanding The Error

### What You Saw
```json
{
  "caf_answer": "Unknown",
  "caf_decision": "ERROR",
  "iterations": 0,
  "response_text": "Error: 'CAFOutput' object has no attribute 'intermediate_scores'"
}
```

### What This Meant
- CAF loop tried to run
- Hit error accessing wrong attribute
- Crashed before generating answer
- Marked as "Unknown"
- No iterations happened

### After Fix
```json
{
  "caf_answer": "no",  // ← Actual answer extracted
  "caf_decision": "ACCEPT",
  "iterations": 2-3,  // ← CAF loop ran
  "response_text": "Based on the causal chain..."
}
```

## Expected Results After Fix

### With LLM Only
```
Overall Accuracy: ~40-50%
Avg Iterations: 1 (no iteration)
```

### With Full CAF (LLM + SPARQL)
```
Overall Accuracy: ~60-65%
Avg Iterations: 2-3 (verification loop working)
```

## How to Verify Fix Worked

1. **Check iterations > 0**:
   ```bash
   jq '.avg_iterations' results/tiny_test_fixed/metrics.json
   # Should be > 0 (was 0.0 before)
   ```

2. **Check answers aren't all "Unknown"**:
   ```bash
   jq '.answer_distribution' results/tiny_test_fixed/metrics.json
   # Should show "Yes" and "No", not just "Unknown"
   ```

3. **Check no ERROR decisions**:
   ```bash
   jq '.[].caf_decision' results/tiny_test_fixed/results.json | sort | uniq -c
   # Should show ACCEPT/REJECT/UNCERTAIN, not ERROR
   ```

## Summary

✅ **Bug identified**: Wrong attribute name
✅ **Bug fixed**: Changed to `iteration_logs`
⏳ **Test running**: 3 examples to verify fix
🎯 **Expected**: ~60% accuracy with iterations > 0

Wait for the test to complete and check results. If accuracy is still 0%, we'll debug further. If it's > 0%, the fix worked and you can run full experiments!

## If Still 0% After Fix

Check these:

1. **Is TinyLlama generating responses?**
   ```bash
   jq '.[0].response_text' results/tiny_test_fixed/results.json
   # Should show actual text, not error
   ```

2. **Are there other errors?**
   ```bash
   grep -i error results/tiny_test_fixed/results.json
   ```

3. **Try simulation mode to isolate**:
   ```bash
   python -m experiments.run_counterbench_experiment \
       --input data/counterbench.json \
       --limit 5 \
       --output results/sim_baseline
   # Simulation should get ~40-50%
   ```

Let me know what the test results show!
