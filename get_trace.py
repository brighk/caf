#!/usr/bin/env python3
import sys
import traceback
sys.path.insert(0, '.')

try:
    from experiments.run_counterbench_with_intervention import main
    sys.argv = ['test', '--input', 'data/counterbench.json', '--limit', '1', '--use-llm', '--llm-model', 'tiny', '--llm-4bit', '--use-intervention', '--output', 'results/trace_debug']
    main()
except Exception as e:
    print("\n" + "="*70)
    print("FULL STACK TRACE:")
    print("="*70)
    traceback.print_exc()
    print("="*70)
