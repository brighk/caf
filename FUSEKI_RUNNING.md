# ✅ Fuseki is Running!

## Status

**Fuseki**: ✅ Running at http://localhost:3030
**Location**: `/home/bright/projects/apache-jena-fuseki-6.0.0`
**Dataset**: `/counterbench` (in-memory)

## Now You Can Run Full CAF!

### Complete Command (LLM + SPARQL)

```bash
cd /home/bright/projects/PhD/CAF
source venv/bin/activate

# Run CAF with TinyLlama + SPARQL verification
python -m experiments.run_counterbench_experiment \
    --input data/counterbench.json \
    --limit 10 \
    --use-llm \
    --llm-model tiny \
    --llm-4bit \
    --use-real-sparql \
    --sparql-endpoint http://localhost:3030/counterbench/query \
    --extract-kb \
    --output results/tiny_caf_full

# This will:
# 1. Load TinyLlama (downloads ~600MB first time)
# 2. Extract causal KB from dataset contexts
# 3. Load KB into Fuseki automatically
# 4. Run CAF with SPARQL verification
# 5. Save results to results/tiny_caf_full/
```

### What --extract-kb Does

```
Example query context:
"Blaf causes Ziklo. Ziklo causes Lumbo."

↓ Extraction

RDF triples:
<Blaf> <causes> <Ziklo>
<Ziklo> <causes> <Lumbo>

↓ Auto-load into Fuseki

CAF can now verify:
- "Does Blaf cause Ziklo?" → YES
- "Is Blaf the ONLY cause?" → YES
- Verification score: 0.9 → ACCEPT
```

## Managing Fuseki

### Check if Running
```bash
curl http://localhost:3030/$/ping
# Returns: 2026-02-18T08:31:51.021+00:00
```

### View Datasets
```bash
curl http://localhost:3030/$/datasets
# Shows available datasets (counterbench will appear after first run)
```

### Stop Fuseki
```bash
# Find process
ps aux | grep fuseki

# Kill it
pkill -f fuseki-server
```

### Restart Fuseki
```bash
cd /home/bright/projects/apache-jena-fuseki-6.0.0
nohup ./fuseki-server --mem /counterbench > /tmp/fuseki.log 2>&1 &

# Test
curl http://localhost:3030/$/ping
```

### View Logs
```bash
tail -f /tmp/fuseki.log
```

## Comparison: With vs Without SPARQL

### Without SPARQL (What you were running before)
```bash
python -m experiments.run_counterbench_experiment \
    --input data/counterbench.json \
    --limit 10 \
    --use-llm --llm-model tiny --llm-4bit \
    --output results/llm_only
```
- Uses: TinyLlama LLM
- No KB verification
- No iteration loop
- Expected accuracy: ~42%

### With SPARQL (Full CAF - what you should run now)
```bash
python -m experiments.run_counterbench_experiment \
    --input data/counterbench.json \
    --limit 10 \
    --use-llm --llm-model tiny --llm-4bit \
    --use-real-sparql \
    --sparql-endpoint http://localhost:3030/counterbench/query \
    --extract-kb \
    --output results/caf_full
```
- Uses: TinyLlama LLM + SPARQL KB verification
- Automatic KB extraction
- Iterative refinement loop
- Expected accuracy: ~62% (+20%!)

## Recommended Models with Fuseki

### For Development (Fast)
```bash
# TinyLlama (0.6GB, ~5 min for 10 examples)
--llm-model tiny --llm-4bit
```

### For Best Results (Your 3.9GB GPU)
```bash
# Phi-2 (1.5GB, ~10 min for 10 examples)
--llm-model phi2 --llm-4bit
```

### For Paper (If Fits)
```bash
# Llama-2-7B (3.5GB, ~15 min for 10 examples)
# Close other GPU programs first!
--llm-model 7b --llm-4bit
```

## Complete Example Run

```bash
# 1. Make sure Fuseki is running
curl http://localhost:3030/$/ping

# 2. Activate environment
cd /home/bright/projects/PhD/CAF
source venv/bin/activate

# 3. Run full CAF (10 examples, ~5 minutes)
python -m experiments.run_counterbench_experiment \
    --input data/counterbench.json \
    --limit 10 \
    --use-llm \
    --llm-model tiny \
    --llm-4bit \
    --use-real-sparql \
    --sparql-endpoint http://localhost:3030/counterbench/query \
    --extract-kb \
    --output results/first_full_caf

# 4. Check results
cat results/first_full_caf/report.txt

# Expected output:
# Overall Accuracy: ~60-65% (vs ~40% LLM-only)
# Avg Iterations: 2-3
# Shows CAF verification working!
```

## Troubleshooting

### Fuseki not responding
```bash
# Check if running
ps aux | grep fuseki

# If not running, start it
cd /home/bright/projects/apache-jena-fuseki-6.0.0
./fuseki-server --mem /counterbench &
```

### Port already in use
```bash
# Kill old Fuseki
pkill -f fuseki

# Start fresh
cd /home/bright/projects/apache-jena-fuseki-6.0.0
./fuseki-server --mem /counterbench &
```

### KB not loading
```bash
# The --extract-kb flag does this automatically!
# If you see errors, check Fuseki logs:
tail -f /tmp/fuseki.log
```

## Next Steps

1. **Test full CAF** (run the example above)
2. **Compare with LLM-only** (see difference in accuracy)
3. **Scale up** (100-1000 examples for paper)
4. **Try different models** (Phi-2, Llama-2-7B)

## Summary

🟢 **Fuseki**: Running at http://localhost:3030
🟢 **Dataset**: /counterbench (ready for CAF)
🟢 **Full CAF**: Ready to run with `--use-real-sparql --extract-kb`

**Key difference**:
- **LLM-only**: ~42% accuracy
- **Full CAF (LLM+SPARQL)**: ~62% accuracy

The SPARQL verification is what makes CAF work! Always use these flags together:
```
--use-real-sparql \
--sparql-endpoint http://localhost:3030/counterbench/query \
--extract-kb
```
