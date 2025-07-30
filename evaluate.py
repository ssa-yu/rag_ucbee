import json
import string
import os
from typing import List, Dict
import logging
from dataclasses import dataclass
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QAPair:
    """Represents a question-answer pair for evaluation"""
    id: str
    question: str
    answer: str
    url: str

def normalize_answer(answer: str) -> str:
    """Normalize answer for evaluation (lowercase, remove punctuation, extra spaces)"""
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove punctuation
    answer = ''.join(char for char in answer if char not in string.punctuation)
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    return answer

def exact_match_score(predicted: str, reference: str) -> int:
    """Calculate exact match score (1 if exact match, 0 otherwise)"""
    pred_normalized = normalize_answer(predicted)
    ref_normalized = normalize_answer(reference)
    
    return 1 if pred_normalized == ref_normalized else 0

def load_qa_pairs(filepath: str) -> List[QAPair]:
    """Load QA pairs from JSONL file (one JSON object per line)"""
    qa_pairs = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            item = json.loads(line)
            
            # Generate ID if not present
            qa_id = item.get('id', f"{line_num:03d}")
            
            qa_pair = QAPair(
                id=qa_id,
                question=item['question'],
                answer=item['answer'],
                url=item['url']
            )
            qa_pairs.append(qa_pair)
    
    logger.info(f"Loaded {len(qa_pairs)} QA pairs from {filepath}")
    return qa_pairs

def evaluate_rag_system(rag_system, qa_pairs: List[QAPair], top_k: int = 3) -> Dict:
    """Evaluate RAG system on QA pairs using exact match"""
    logger.info(f"Starting evaluation of {len(qa_pairs)} QA pairs...")
    
    results = []
    correct_answers = 0
    retrieval_hits = 0
    total_questions = len(qa_pairs)
    
    for i, qa_pair in enumerate(qa_pairs):
        logger.info(f"Processing question {i+1}/{total_questions}: {qa_pair.id}")
        
        try:
            # Get answer from RAG system
            rag_result = rag_system.answer_question(qa_pair.question, top_k)
            predicted_answer = rag_result['answer']
            
            # Calculate exact match
            em_score = exact_match_score(predicted_answer, qa_pair.answer)
            correct_answers += em_score

            # Calculate retrieval hit rate (check if reference URL is in retrieved sources)
            retrieved_urls = [source.get('url', '') for source in rag_result['sources']]
            hit = 1 if qa_pair.url in retrieved_urls else 0
            retrieval_hits += hit
            
            # Store individual result
            result = {
                'id': qa_pair.id,
                'question': qa_pair.question,
                'reference_answer': qa_pair.answer,
                'predicted_answer': predicted_answer,
                'exact_match': em_score,
                'reference_url': qa_pair.url,
                'retrieved_urls': retrieved_urls,
                'retrieval_hit': hit,
                'sources_found': len(rag_result['sources'])
            }
            results.append(result)
            
            # Print progress
            em_status = "✓" if em_score == 1 else "✗"
            hit_status = "✓" if hit == 1 else "✗"
            logger.info(f"  EM: {em_status} | Hit: {hit_status} | Ref: '{qa_pair.answer}' | Pred: '{predicted_answer}'")
            
        except Exception as e:
            logger.error(f"Error processing question {qa_pair.id}: {str(e)}")
            result = {
                'id': qa_pair.id,
                'question': qa_pair.question,
                'reference_answer': qa_pair.answer,
                'predicted_answer': "",
                'exact_match': 0,
                'reference_url': qa_pair.url,
                'retrieved_urls': [],
                'retrieval_hit': 0,
                'sources_found': 0,
                'error': str(e)
            }
            results.append(result)
    
    # Calculate final metrics
    exact_match_accuracy = correct_answers / total_questions
    retrieval_hit_rate = retrieval_hits / total_questions
    
    evaluation_results = {
        'exact_match_accuracy': exact_match_accuracy,
        'retrieval_hit_rate': retrieval_hit_rate,
        'correct_answers': correct_answers,
        'retrieval_hits': retrieval_hits,
        'total_questions': total_questions,
        'individual_results': results
    }
    
    logger.info(f"Evaluation completed!")
    logger.info(f"Exact Match Accuracy: {exact_match_accuracy:.3f} ({correct_answers}/{total_questions})")
    logger.info(f"Retrieval Hit Rate: {retrieval_hit_rate:.3f} ({retrieval_hits}/{total_questions})")
    
    return evaluation_results

def print_results(results: Dict):
    """Print evaluation results in a clear format"""
    print("\n" + "="*80)
    print("RAG SYSTEM EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Exact Match Accuracy: {results['exact_match_accuracy']:.3f}")
    print(f"  Retrieval Hit Rate:   {results['retrieval_hit_rate']:.3f}")
    print(f"  Correct Answers: {results['correct_answers']}")
    print(f"  Retrieval Hits:  {results['retrieval_hits']}")
    print(f"  Total Questions: {results['total_questions']}")
    
    # Show correct answers
    correct_results = [r for r in results['individual_results'] if r['exact_match'] == 1]
    if correct_results:
        print(f"\nCORRECT ANSWERS ({len(correct_results)}):")
        for result in correct_results:
            hit_status = "✓" if result['retrieval_hit'] == 1 else "✗"
            print(f"  ✓ [{result['id']}] {result['question'][:60]}...")
            print(f"    Answer: {result['reference_answer']} | Hit: {hit_status}")
    
    # Show incorrect answers
    incorrect_results = [r for r in results['individual_results'] if r['exact_match'] == 0]
    if incorrect_results:
        print(f"\nINCORRECT ANSWERS ({len(incorrect_results)}):")
        for result in incorrect_results:
            hit_status = "✓" if result['retrieval_hit'] == 1 else "✗"
            print(f"  ✗ [{result['id']}] {result['question'][:60]}...")
            print(f"    Expected: '{result['reference_answer']}'")
            print(f"    Got:      '{result['predicted_answer']}'")
            print(f"    Hit: {hit_status} | URL: {result['reference_url']}")
    
    # Show errors if any
    error_results = [r for r in results['individual_results'] if 'error' in r]
    if error_results:
        print(f"\nERRORS ({len(error_results)}):")
        for result in error_results:
            print(f"  ⚠ [{result['id']}] {result['error']}")

def save_results(results: Dict, output_file: str):
    """Save results to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_file}")

def run_ablation_study(rag_system, qa_pairs: List[QAPair]):
    """Run ablation study with different top_k values"""
    print("\n" + "="*80)
    print("ABLATION STUDY - Different top_k values")
    print("="*80)
    
    top_k_values = [1, 3, 5, 7]
    ablation_results = []
    
    for top_k in top_k_values:
        print(f"\nTesting with top_k={top_k}...")
        results = evaluate_rag_system(rag_system, qa_pairs, top_k=top_k)
        accuracy = results['exact_match_accuracy']
        
        ablation_results.append({
            'top_k': top_k,
            'exact_match_accuracy': accuracy,
            'correct_answers': results['correct_answers']
        })
        
        print(f"  Exact Match Accuracy: {accuracy:.3f}")
    
    # Find best top_k
    best_result = max(ablation_results, key=lambda x: x['exact_match_accuracy'])
    print(f"\nBest performance: top_k={best_result['top_k']} with accuracy={best_result['exact_match_accuracy']:.3f}")
    
    return ablation_results

# Enhanced evaluation functions for GraphRAG
def evaluate_rag_system_enhanced(rag_system, qa_pairs: List[Dict], top_k: int = 3, use_graph: bool = True) -> Dict:
    """Enhanced evaluation that supports GraphRAG comparison"""
    logger.info(f"Evaluating RAG system with {len(qa_pairs)} QA pairs (Graph: {use_graph})")
    
    results = []
    
    for i, qa in enumerate(qa_pairs):
        question = qa.question
        expected_answer = qa.answer
        
        try:
            # Get answer from system
            if hasattr(rag_system, 'answer_question'):
                # Check if the system supports use_graph parameter
                import inspect
                sig = inspect.signature(rag_system.answer_question)
                if 'use_graph' in sig.parameters:
                    response = rag_system.answer_question(question, top_k=top_k, use_graph=use_graph)
                else:
                    response = rag_system.answer_question(question, top_k=top_k)
                generated_answer = response['answer']
                sources = response.get('sources', [])
            else:
                # Fallback for older systems
                response = rag_system.answer_question(question, top_k=top_k)
                generated_answer = response['answer']
                sources = response.get('sources', [])
            
            # Calculate metrics (implement your evaluation metrics here)
            relevance_score = calculate_relevance(question, generated_answer)
            faithfulness_score = calculate_faithfulness(generated_answer, sources)
            answer_similarity = calculate_answer_similarity(expected_answer, generated_answer)
            
            result = {
                'question': question,
                'expected_answer': expected_answer,
                'generated_answer': generated_answer,
                'relevance_score': relevance_score,
                'faithfulness_score': faithfulness_score,
                'answer_similarity': answer_similarity,
                'num_sources': len(sources),
                'sources': sources
            }
            
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(qa_pairs)} questions")
                
        except Exception as e:
            logger.error(f"Error processing question {i}: {e}")
            continue
    
    # Calculate aggregate metrics
    if results:
        avg_relevance = np.mean([r['relevance_score'] for r in results])
        avg_faithfulness = np.mean([r['faithfulness_score'] for r in results])
        avg_answer_similarity = np.mean([r['answer_similarity'] for r in results])
        avg_sources = np.mean([r['num_sources'] for r in results])
        
        aggregate_results = {
            'total_questions': len(qa_pairs),
            'successful_answers': len(results),
            'avg_relevance': avg_relevance,
            'avg_faithfulness': avg_faithfulness,
            'avg_answer_similarity': avg_answer_similarity,
            'avg_sources_used': avg_sources,
            'individual_results': results,
            'retrieval_precision': calculate_retrieval_precision(results)
        }
    else:
        aggregate_results = {
            'total_questions': len(qa_pairs),
            'successful_answers': 0,
            'error': 'No successful evaluations'
        }
    
    return aggregate_results

def calculate_relevance(question: str, answer: str) -> float:
    """Calculate relevance score between question and answer"""
    # Simple implementation - you can enhance this
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    
    if not question_words:
        return 0.0
    
    overlap = len(question_words.intersection(answer_words))
    return overlap / len(question_words)

def calculate_faithfulness(answer: str, sources: List[Dict]) -> float:
    """Calculate faithfulness score based on source content"""
    if not sources:
        return 0.0
    
    # Simple implementation - check if answer content appears in sources
    answer_words = set(answer.lower().split())
    source_words = set()
    
    for source in sources:
        content = source.get('content_snippet', '')
        source_words.update(content.lower().split())
    
    if not answer_words:
        return 0.0
    
    overlap = len(answer_words.intersection(source_words))
    return overlap / len(answer_words)

def calculate_answer_similarity(expected: str, generated: str) -> float:
    """Calculate similarity between expected and generated answers"""
    # Simple Jaccard similarity
    expected_words = set(expected.lower().split())
    generated_words = set(generated.lower().split())
    
    if not expected_words and not generated_words:
        return 1.0
    
    intersection = len(expected_words.intersection(generated_words))
    union = len(expected_words.union(generated_words))
    
    return intersection / union if union > 0 else 0.0

def calculate_retrieval_precision(results: List[Dict]) -> float:
    """Calculate retrieval precision based on source relevance"""
    if not results:
        return 0.0
    
    relevant_retrievals = 0
    total_retrievals = 0
    
    for result in results:
        sources = result.get('sources', [])
        total_retrievals += len(sources)
        
        # Count sources with high relevance scores as relevant
        for source in sources:
            if source.get('relevance_score', 0) > 0.5:
                relevant_retrievals += 1
    
    return relevant_retrievals / total_retrievals if total_retrievals > 0 else 0.0

def print_comparison_results(results: Dict):
    """Print comparison results between different RAG configurations"""
    print("\n" + "="*80)
    print("ENHANCED RAG EVALUATION RESULTS")
    print("="*80)
    
    if 'without_graph' in results:
        print("\n📊 WITHOUT GRAPHRAG:")
        print_single_results(results['without_graph'])
    
    if 'with_graph' in results:
        print("\n🔗 WITH GRAPHRAG:")
        print_single_results(results['with_graph'])
    
    # Print comparison if both exist
    if 'without_graph' in results and 'with_graph' in results:
        print("\n📈 IMPROVEMENT ANALYSIS:")
        no_graph = results['without_graph']
        with_graph = results['with_graph']
        
        metrics = [
            ('avg_relevance', 'Average Relevance'),
            ('avg_faithfulness', 'Average Faithfulness'),
            ('avg_answer_similarity', 'Answer Similarity'),
            ('retrieval_precision', 'Retrieval Precision')
        ]
        
        for metric_key, metric_name in metrics:
            if metric_key in no_graph and metric_key in with_graph:
                baseline = no_graph[metric_key]
                enhanced = with_graph[metric_key]
                improvement = ((enhanced - baseline) / baseline) * 100 if baseline > 0 else 0
                
                status = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                print(f"  {status} {metric_name}: {baseline:.3f} → {enhanced:.3f} ({improvement:+.1f}%)")

def print_single_results(results: Dict):
    """Print results for a single configuration"""
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"✅ Successful answers: {results['successful_answers']}/{results['total_questions']}")
    print(f"📊 Average Relevance: {results.get('avg_relevance', 0):.3f}")
    print(f"🎯 Average Faithfulness: {results.get('avg_faithfulness', 0):.3f}")
    print(f"📝 Answer Similarity: {results.get('avg_answer_similarity', 0):.3f}")
    print(f"🔍 Retrieval Precision: {results.get('retrieval_precision', 0):.3f}")
    print(f"📚 Average Sources Used: {results.get('avg_sources_used', 0):.1f}")
