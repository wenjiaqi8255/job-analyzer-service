#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer.content_separator.syntactic_pattern_classifier import SyntacticPatternClassifier
import spacy


def test_syntactic_classifier():
    """Test the SyntacticPatternClassifier with sample job posting data."""
    
    # Initialize classifier
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
        classifier = SyntacticPatternClassifier(nlp)
        print("✓ Classifier initialized successfully\n")
    except OSError:
        print("❌ Error: spaCy English model not found. Please install with:")
        print("python -m spacy download en_core_web_sm")
        return
    
    # Sample job posting texts
    sample_texts = [
        # Company introduction examples
        "We are a leading technology company founded in 2010, specializing in AI solutions.",
        "Our company has been providing innovative software solutions for over 15 years.",
        "The organization was established to revolutionize the fintech industry.",
        "We pride ourselves on being a customer-focused team with global reach.",
        
        # Job description examples  
        "You will be responsible for developing and maintaining web applications.",
        "The candidate must have at least 3 years of experience in Python programming.",
        "Develop high-quality software solutions using modern frameworks.",
        "You should have strong communication skills and ability to work in teams.",
        "Required: Bachelor's degree in Computer Science or related field.",
        "Responsibilities include code reviews, testing, and deployment.",
        
        # Mixed/ambiguous examples
        "This role offers excellent career growth opportunities.",
        "Join our dynamic team of experienced developers.",
        "The position requires knowledge of cloud technologies.",
    ]
    
    print("=== Individual Text Classification ===\n")
    
    for i, text in enumerate(sample_texts, 1):
        print(f"Text {i}: {text}")
        result = classifier.classify_text(text)
        
        print(f"  Classification: {result['type'].upper()}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Company Score: {result['company_score']:.2f}")
        print(f"  Job Score: {result['job_score']:.2f}")
        print(f"  Subjects: {result['syntax_analysis']['subjects']}")
        print(f"  Root Verbs: {result['syntax_analysis']['root_verbs']}")
        print(f"  Has Imperative: {result['syntax_analysis']['has_imperative']}")
        print(f"  Has Passive: {result['syntax_analysis']['has_passive']}")
        print("-" * 60)
    
    # Test batch classification
    print("\n=== Batch Classification Summary ===\n")
    
    summary = classifier.get_classification_summary(sample_texts)
    print(f"Total texts analyzed: {summary['total_texts']}")
    print(f"Company info: {summary['company_info_count']} ({summary['classification_ratio']['company_info']:.1%})")
    print(f"Job content: {summary['job_content_count']} ({summary['classification_ratio']['job_content']:.1%})")
    print(f"Neutral: {summary['neutral_count']} ({summary['classification_ratio']['neutral']:.1%})")
    print(f"Average company score: {summary['avg_company_score']:.2f}")
    print(f"Average job score: {summary['avg_job_score']:.2f}")
    
    # Test sentence-level classification
    print("\n=== Sentence-Level Classification ===\n")
    
    sample_paragraph = """
    We are a fast-growing startup that specializes in machine learning solutions. 
    Our team consists of experienced engineers and data scientists. 
    You will be responsible for building scalable ML pipelines. 
    The candidate must have experience with Python, TensorFlow, and cloud platforms. 
    We offer competitive salary and excellent benefits.
    """
    
    print("Sample paragraph:")
    print(sample_paragraph.strip())
    print("\nSentence-by-sentence analysis:")
    
    sentence_results = classifier.classify_sentences(sample_paragraph)
    for i, result in enumerate(sentence_results, 1):
        print(f"{i}. [{result['type'].upper()}] {result['sentence']}")
        print(f"   Confidence: {result['confidence']:.2f} | Company: {result['company_score']:.2f} | Job: {result['job_score']:.2f}")
    
    print("\n=== Test Complete ===")


def test_edge_cases():
    """Test edge cases and potential issues."""
    
    classifier = SyntacticPatternClassifier()
    
    print("\n=== Edge Case Testing ===\n")
    
    edge_cases = [
        "",  # Empty string
        "   ",  # Whitespace only
        "Hello.",  # Single word
        "We we we we we.",  # Repeated pronouns
        "You must you should you will you need.",  # Repeated requirements
        "The quick brown fox jumps over the lazy dog.",  # Neutral content
    ]
    
    for i, text in enumerate(edge_cases, 1):
        if text.strip():
            print(f"Edge case {i}: '{text}'")
        else:
            print(f"Edge case {i}: [Empty/Whitespace]")
            
        result = classifier.classify_text(text)
        print(f"  Result: {result['type']} (confidence: {result['confidence']:.2f})")
        print()


if __name__ == "__main__":
    test_syntactic_classifier()
    test_edge_cases()