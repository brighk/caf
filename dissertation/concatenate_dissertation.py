#!/usr/bin/env python3
"""
Script to concatenate all dissertation chapters into a single LaTeX file.

This script reads the main dissertation file structure and all chapter files,
then creates a single concatenated file (diss_temp.tex) that can be compiled
as a standalone document.

Usage:
    python concatenate_dissertation.py
"""

import os
from pathlib import Path

def read_file(filepath):
    """Read a file and return its contents."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return f"% File not found: {filepath}\n"
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return f"% Error reading file: {filepath}\n"

def extract_preamble(main_content):
    """Extract everything from the main file up to \\begin{document}"""
    lines = main_content.split('\n')
    preamble = []
    for line in lines:
        preamble.append(line)
        if r'\begin{document}' in line:
            break
    return '\n'.join(preamble)

def main():
    # Set up paths
    base_dir = Path(__file__).parent
    chapters_dir = base_dir / 'chapters'
    main_file = base_dir / 'dissertation_main.tex'
    output_file = base_dir / 'diss_temp.tex'

    print(f"Base directory: {base_dir}")
    print(f"Chapters directory: {chapters_dir}")
    print(f"Main file: {main_file}")
    print(f"Output file: {output_file}")
    print()

    # Read the main file
    print("Reading main file...")
    main_content = read_file(main_file)

    # Extract preamble (everything up to and including \begin{document})
    preamble = extract_preamble(main_content)

    # List of chapters in order
    chapter_files = [
        'frontmatter.tex',
        'chapter01_introduction.tex',
        'chapter02_background.tex',
        'chapter03_foundations.tex',
        'chapter04_caf_architecture.tex',
        'chapter05_causal_discovery.tex',
        'chapter06_eval_caf.tex',
        'chapter07_eval_causal.tex',
        'chapter08_deployment.tex',
        'chapter09_discussion.tex',
        'chapter10_conclusion.tex',
        'backmatter.tex'
    ]

    # Start building the concatenated document
    output_lines = []

    # Add preamble
    output_lines.append(preamble)
    output_lines.append('')
    output_lines.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    output_lines.append('%% FRONT MATTER')
    output_lines.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    output_lines.append('')
    output_lines.append(r'\frontmatter')
    output_lines.append('')

    # Add frontmatter
    print("Reading frontmatter...")
    frontmatter_path = chapters_dir / 'frontmatter.tex'
    frontmatter_content = read_file(frontmatter_path)
    output_lines.append(frontmatter_content)
    output_lines.append('')

    # Add ToC, LoF, LoT
    output_lines.append('% Table of Contents')
    output_lines.append(r'\tableofcontents')
    output_lines.append(r'\listoffigures')
    output_lines.append(r'\listoftables')
    output_lines.append('')

    # Add main matter
    output_lines.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    output_lines.append('%% MAIN MATTER')
    output_lines.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    output_lines.append('')
    output_lines.append(r'\mainmatter')
    output_lines.append('')

    # Add all chapters
    main_chapters = [
        'chapter01_introduction.tex',
        'chapter02_background.tex',
        'chapter03_foundations.tex',
        'chapter04_caf_architecture.tex',
        'chapter05_causal_discovery.tex',
        'chapter06_eval_caf.tex',
        'chapter07_eval_causal.tex',
        'chapter08_deployment.tex',
        'chapter09_discussion.tex',
        'chapter10_conclusion.tex'
    ]

    for chapter_file in main_chapters:
        chapter_path = chapters_dir / chapter_file
        if chapter_path.exists():
            print(f"Reading {chapter_file}...")
            chapter_content = read_file(chapter_path)
            output_lines.append(f'% ====== {chapter_file} ======')
            output_lines.append(chapter_content)
            output_lines.append('')
        else:
            print(f"Warning: Chapter not found: {chapter_file}")
            output_lines.append(f'% WARNING: Chapter file not found: {chapter_file}')
            output_lines.append('')

    # Add back matter
    output_lines.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    output_lines.append('%% BACK MATTER')
    output_lines.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    output_lines.append('')
    output_lines.append(r'\backmatter')
    output_lines.append('')

    # Add bibliography/backmatter
    backmatter_path = chapters_dir / 'backmatter.tex'
    if backmatter_path.exists():
        print("Reading backmatter...")
        backmatter_content = read_file(backmatter_path)
        output_lines.append(backmatter_content)
        output_lines.append('')
    else:
        print("Warning: backmatter.tex not found")
        output_lines.append('% WARNING: backmatter.tex not found')
        output_lines.append('')

    # Close document
    output_lines.append(r'\end{document}')

    # Write output file
    print()
    print(f"Writing concatenated file to {output_file}...")
    full_content = '\n'.join(output_lines)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_content)

    # Print statistics
    print()
    print("=" * 60)
    print("CONCATENATION COMPLETE!")
    print("=" * 60)
    print(f"Output file: {output_file}")
    print(f"Total lines: {len(output_lines):,}")
    print(f"Total characters: {len(full_content):,}")
    print(f"Estimated pages (assuming ~2000 chars/page): {len(full_content) // 2000}")
    print()
    print("You can now compile with:")
    print(f"  cd {base_dir}")
    print(f"  pdflatex diss_temp.tex")
    print(f"  bibtex diss_temp")
    print(f"  pdflatex diss_temp.tex")
    print(f"  pdflatex diss_temp.tex")
    print()

if __name__ == '__main__':
    main()
