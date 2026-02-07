#!/bin/bash
# Backup Claude Code session and important files
# Run this before the host reboots!

BACKUP_DIR="/workspace/caf/session_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating backup in: $BACKUP_DIR"

# 1. Copy the conversation transcript
echo "Backing up conversation transcript..."
cp ~/.claude/projects/-workspace-caf/*.jsonl "$BACKUP_DIR/" 2>/dev/null || echo "No transcripts found"

# 2. Copy MEMORY.md (persistent knowledge)
echo "Backing up memory..."
cp ~/.claude/projects/-workspace-caf/memory/MEMORY.md "$BACKUP_DIR/" 2>/dev/null || echo "No memory file"

# 3. Create a human-readable summary
echo "Creating summary..."
cat > "$BACKUP_DIR/SESSION_SUMMARY.md" << 'EOF'
# CAF Experiment Session Summary

## Date
$(date)

## What We Built

1. **CAF Framework Implementation**
   - Simulated verification layer for experiments
   - Real LLM integration (Llama-2-7b-chat)
   - 4 baseline methods (Vanilla, CoT, RAG, RAG+CoT)

2. **Experimental Results**
   - 75 synthetic causal chains across 5 domains
   - CAF: 76.5% entailment accuracy
   - Vanilla: 62.0% (best baseline)
   - CoT: 52.4%, RAG: 53.8%, RAG+CoT: 52.7%

3. **Publication-Ready Paper**
   - LaTeX paper with algorithm, results, figures
   - 4 publication-quality figures (PDF + PNG)
   - Honest about simulation vs. real SPARQL

## Key Files

- **Paper:** /workspace/caf/paper/paper.tex
- **Figures:** /workspace/caf/paper/figures/*.pdf
- **Experiments:** /workspace/caf/experiments/
- **Results:** /workspace/caf/experiments/results/
- **Migration Guide:** /workspace/caf/REAL_SPARQL_MIGRATION.md

## How to Run

```bash
# Run experiment (simulation)
python -m experiments.run_experiment --use-llm --llm-4bit --num-chains 75

# Visualize results
python -m experiments.visualize_results

# Compile paper
cd paper && pdflatex paper.tex
```

## Next Steps

1. Review migration guide for real SPARQL integration (~12-24 hours)
2. Consider running on real datasets (FEVER, HotpotQA)
3. Submit to conference/journal

EOF

# 4. Copy all important code files
echo "Backing up code..."
cp -r /workspace/caf/experiments "$BACKUP_DIR/"
cp -r /workspace/caf/paper "$BACKUP_DIR/"
cp /workspace/caf/*.md "$BACKUP_DIR/" 2>/dev/null
cp /workspace/caf/*.sh "$BACKUP_DIR/" 2>/dev/null

# 5. Copy experimental results
echo "Backing up results..."
cp -r /workspace/caf/experiments/results "$BACKUP_DIR/" 2>/dev/null || echo "No results"

# 6. Create a tarball
echo "Creating tarball..."
cd /workspace/caf
tar -czf "${BACKUP_DIR}.tar.gz" session_backup_*

echo ""
echo "✓ Backup complete!"
echo ""
echo "Backup location: $BACKUP_DIR"
echo "Tarball: ${BACKUP_DIR}.tar.gz"
echo ""
echo "To copy to your local machine via scp:"
echo "  scp user@host:${BACKUP_DIR}.tar.gz ~/Downloads/"
echo ""
echo "Or sync to remote storage:"
echo "  rsync -avz ${BACKUP_DIR}.tar.gz user@backup-server:~/backups/"
echo ""
