#!/usr/bin/env python3
"""
Tool to compare before/after content through the pipeline processing stages.
"""

import spacy
from analyzer.utils.text_preprocessor import TextPreprocessor
from analyzer.content_separator.job_content_separator import JobContentSeparator
from extractor.job_data_accessor import JobDataAccessor
import pandas as pd
import textwrap

def create_side_by_side_comparison(original, processed, title="COMPARISON"):
    """Create a side-by-side comparison of original vs processed text."""
    
    print(f"\n{'='*20} {title} {'='*20}")
    
    # Wrap text to fit in columns
    width = 35
    original_lines = textwrap.wrap(original, width=width) if original else ["[Empty]"]
    processed_lines = textwrap.wrap(processed, width=width) if processed else ["[Empty]"]
    
    # Pad to same length
    max_lines = max(len(original_lines), len(processed_lines))
    original_lines.extend([""] * (max_lines - len(original_lines)))
    processed_lines.extend([""] * (max_lines - len(processed_lines)))
    
    print(f"{'ORIGINAL':<{width}} | {'PROCESSED':<{width}}")
    print("-" * width + " | " + "-" * width)
    
    for orig, proc in zip(original_lines, processed_lines):
        print(f"{orig:<{width}} | {proc:<{width}}")
    
    print("-" * (width * 2 + 3))
    print(f"Length: {len(original):<{width-8}} | Length: {len(processed)}")

def analyze_content_transformation(job_data):
    """Analyze how content is transformed through each pipeline stage."""
    
    print("🔄 CONTENT TRANSFORMATION ANALYSIS")
    print("="*80)
    
    # Setup
    nlp = spacy.load("en_core_web_sm")
    text_preprocessor = TextPreprocessor(nlp)
    job_content_separator = JobContentSeparator(nlp)
    
    # Get original description
    original_description = JobDataAccessor.get_effective_description(pd.Series(job_data))
    
    # Stage 1: Original vs Preprocessed
    preprocessed_text = text_preprocessor.preprocess_text(original_description)
    create_side_by_side_comparison(
        original_description, 
        preprocessed_text, 
        "STAGE 1: BASIC PREPROCESSING"
    )
    
    # Stage 2: Original vs Chunks
    chunks = text_preprocessor.chunk_text(original_description)
    chunks_text = "\n".join(f"{i+1}. {chunk}" for i, chunk in enumerate(chunks))
    create_side_by_side_comparison(
        original_description,
        chunks_text,
        "STAGE 2: TEXT CHUNKING"
    )
    
    # Stage 3: Original chunks vs Filtered chunks
    filtered_chunks = text_preprocessor.chunk_and_filter_text(original_description)
    filtered_chunks_text = "\n".join(f"{i+1}. {chunk}" for i, chunk in enumerate(filtered_chunks))
    
    print(f"\n{'='*20} STAGE 3: CONTENT FILTERING {'='*20}")
    print(f"📊 FILTERING RESULTS:")
    print(f"  Original chunks: {len(chunks)}")
    print(f"  Filtered chunks: {len(filtered_chunks)}")
    print(f"  Removed: {len(chunks) - len(filtered_chunks)} chunks")
    
    if len(chunks) > len(filtered_chunks):
        removed_chunks = [chunk for chunk in chunks if chunk not in filtered_chunks]
        print(f"\n🗑️  REMOVED CHUNKS:")
        for i, chunk in enumerate(removed_chunks, 1):
            print(f"  {i}. {chunk}")
    
    # Stage 4: Job Content Separation
    separation_result = job_content_separator.separate_job_posting(original_description)
    
    print(f"\n{'='*20} STAGE 4: JOB CONTENT SEPARATION {'='*20}")
    
    company_info = separation_result['company_info_text']
    job_content = separation_result['job_content_text']
    
    create_side_by_side_comparison(
        company_info,
        job_content,
        "COMPANY INFO vs JOB CONTENT"
    )
    
    # Stage 5: Final extraction
    pure_job_requirements = job_content_separator.extract_pure_job_requirements(original_description)
    
    create_side_by_side_comparison(
        original_description,
        pure_job_requirements['job_content_filtered'],
        "STAGE 5: ORIGINAL vs FINAL PURE JOB REQUIREMENTS"
    )
    
    # Summary metrics
    print(f"\n📊 TRANSFORMATION SUMMARY")
    print("-" * 40)
    
    original_len = len(original_description)
    preprocessed_len = len(preprocessed_text)
    company_len = len(company_info)
    job_len = len(job_content)
    final_len = len(pure_job_requirements['job_content_filtered'])
    
    print(f"Original description: {original_len:,} characters")
    print(f"After preprocessing: {preprocessed_len:,} characters ({preprocessed_len/original_len:.1%})")
    print(f"Company info extracted: {company_len:,} characters ({company_len/original_len:.1%})")
    print(f"Job content extracted: {job_len:,} characters ({job_len/original_len:.1%})")
    print(f"Final pure requirements: {final_len:,} characters ({final_len/original_len:.1%})")
    
    print(f"\n📈 EFFICIENCY METRICS:")
    print(f"  Signal-to-noise improvement: {final_len/original_len:.2f} -> 1.00 (pure signal)")
    print(f"  Compression ratio: {original_len/final_len:.2f}x")
    print(f"  Content focus: {final_len/(final_len + company_len):.1%} job requirements")

def main():
    """Run the content transformation analysis."""
    
    # Test with multiple job postings
    test_jobs = [
        {
            'id': 'test_001',
            'job_title': 'Senior Python Developer',
            'company_name': 'TechCorp Inc.',
            'description': """
TechCorp Inc. is a leading software development company founded in 2015. 
We specialize in creating innovative web applications and mobile solutions for enterprise clients.
Our team of 200+ engineers works across multiple time zones to deliver cutting-edge technology.

We are looking for a Senior Python Developer to join our dynamic team.
You will be responsible for developing and maintaining high-performance web applications.
The ideal candidate must have at least 5 years of experience in Python development.
You should be proficient in Django, Flask, and RESTful API development.
Experience with AWS cloud services is required.

We offer competitive salary, health benefits, and flexible working hours.
Join our innovative team and help us build the future of enterprise software!
            """.strip()
        },
        {
            'id': 'test_002',
            'job_title': 'Data Scientist',
            'company_name': 'AI Innovations Ltd.',
            'description': """
AI Innovations Ltd. is revolutionizing the field of artificial intelligence.
Founded by Stanford PhD graduates, we're building the next generation of ML platforms.
Our company has raised $50M in Series B funding and serves Fortune 500 clients.

Seeking a Data Scientist to join our research team.
You will develop machine learning models for predictive analytics.
PhD in Computer Science, Statistics, or related field required.
Must have experience with Python, TensorFlow, and big data technologies.
Knowledge of statistical modeling and hypothesis testing essential.

We provide top-tier compensation, equity, and unlimited PTO.
Our Palo Alto office features cutting-edge research facilities.
            """.strip()
        }
    ]
    
    for i, job in enumerate(test_jobs, 1):
        print(f"\n{'#'*80}")
        print(f"JOB ANALYSIS #{i}: {job['job_title']} at {job['company_name']}")
        print(f"{'#'*80}")
        
        analyze_content_transformation(job)
        
        if i < len(test_jobs):
            input("\nPress Enter to continue to next job analysis...")

if __name__ == "__main__":
    main()