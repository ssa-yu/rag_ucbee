import json
import os
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import faiss
from IPython import embed
import torch
import logging
from dataclasses import dataclass
import pickle
import evaluate

from models import ModelConfig, GeneratorFactory, MODEL_CONFIGS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a document chunk with metadata"""
    id: str
    content: str
    title: str
    url: str
    chunk_index: int

class DocumentChunker:
    """Handles text chunking for RAG system"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        # Split by sentences first for better boundaries
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence exceeds chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk.split()) > self.chunk_size:
                if current_chunk:  # Save current chunk if it exists
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap
                    words = current_chunk.split()
                    overlap_text = " ".join(words[-self.overlap:]) if len(words) > self.overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    # Single sentence is too long, split by words
                    words = sentence.split()
                    for i in range(0, len(words), self.chunk_size - self.overlap):
                        chunk_words = words[i:i + self.chunk_size]
                        chunks.append(" ".join(chunk_words))
                    current_chunk = ""
            else:
                current_chunk = potential_chunk
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_documents(self, crawled_data: List[Dict]) -> List[Document]:
        """Convert crawled data into document chunks"""
        documents = []
        doc_id = 0
        
        for page_data in crawled_data:
            content = page_data.get('text', '')  # Use 'text' field instead of 'content'
            title = self.extract_title_from_content(content)  # Extract title from content
            url = page_data.get('url', '')
            
            # Skip very short content
            if len(content.split()) < 20:
                continue
            
            chunks = self.chunk_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.split()) >= 10:  # Only keep substantial chunks
                    doc = Document(
                        id=f"doc_{doc_id}",
                        content=chunk,
                        title=title,
                        url=url,
                        chunk_index=chunk_idx
                    )
                    documents.append(doc)
                    doc_id += 1
        
        logger.info(f"Created {len(documents)} document chunks")
        return documents
    
    def extract_title_from_content(self, content: str) -> str:
        """Extract title from content since your JSON doesn't have separate title field"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('*') and len(line) < 100:
                # Remove markdown formatting
                title = line.replace('#', '').replace('**', '').strip()
                if title:
                    return title
        return "EECS Page"  # Default title

class EmbeddingRetriever:
    """Handles document embedding and retrieval"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.embeddings = None
    
    def build_index(self, documents: List[Document]):
        """Build FAISS index from documents"""
        self.documents = documents
        
        logger.info(f"Encoding {len(documents)} documents...")
        texts = [doc.content for doc in documents]
        
        # Encode in batches to manage memory
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encoder.encode(batch, show_progress_bar=True)
            embeddings.append(batch_embeddings)
        
        self.embeddings = np.vstack(embeddings)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        logger.info(f"Built FAISS index with {self.index.ntotal} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve top-k most relevant documents"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save_index(self, save_dir: str):
        """Save the index and related data"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "faiss_index.idx"))
        
        # Save documents and embeddings
        with open(os.path.join(save_dir, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        np.save(os.path.join(save_dir, "embeddings.npy"), self.embeddings)
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "num_documents": len(self.documents),
            "embedding_dimension": self.embeddings.shape[1]
        }
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    def load_index(self, save_dir: str):
        """Load the index and related data"""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(save_dir, "faiss_index.idx"))
        
        # Load documents and embeddings
        with open(os.path.join(save_dir, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        
        self.embeddings = np.load(os.path.join(save_dir, "embeddings.npy"))
        
        # Load metadata
        with open(os.path.join(save_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            self.model_name = metadata["model_name"]

class RAGSystem:
    """Complete RAG system combining retrieval and generation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.chunker = DocumentChunker(config.chunk_size, config.overlap)
        self.retriever = EmbeddingRetriever(config.retriever_model)
        self.generator = GeneratorFactory.create_generator(config.generator_type, config.generator_model)
        self.is_built = False
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    def build_system(self, crawled_data_path: str):
        """Build the complete RAG system from crawled data"""
        if self.is_built:
            logger.info("System already built - skipping build")
            return
        
        logger.info("Building RAG system...")
        
        # Load crawled data
        with open(crawled_data_path, 'r', encoding='utf-8') as f:
            file_content = f.read().strip()

        # Parse JSON objects (your file has multiple JSON objects, not a JSON array)
        crawled_data = []
        for line in file_content.split('\n'):
            if line.strip():
                try:
                    json_obj = json.loads(line.strip())
                    crawled_data.append(json_obj)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(crawled_data)} pages")
        
        # Create document chunks
        documents = self.chunker.process_documents(crawled_data)
        
        # Build retrieval index
        self.retriever.build_index(documents)
        
        self.is_built = True
        logger.info("RAG system built successfully!")
    
    def answer_question(self, question: str, top_k: int = 3) -> Dict:
        """Answer a question using the RAG system"""
        if not self.is_built:
            raise ValueError("System not built. Call build_system first.")
        
        k = top_k or self.config.top_k
        retrieved_docs = self.retriever.retrieve(question, k * 2)  # 多取幾筆，給 reranker 有選擇

        # 🟡 加入重排序步驟
        pairs = [(question, doc.content) for doc, _ in retrieved_docs]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = [(doc, score) for (doc, _), score in reranked[:k]]  # 取 top_k

        answer = self.generator.generate_answer(question, reranked_docs, self.config.max_length)

        sources = []
        for doc, score in reranked_docs:
            sources.append({
                'title': doc.title,
                'url': doc.url,
                'relevance_score': float(score),
                'content_snippet': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            })

        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'num_retrieved': len(reranked_docs),
            'config': {
                'retriever_model': self.config.retriever_model,
                'generator_model': self.config.generator_model,
                'generator_type': self.config.generator_type
            }
        }
    
    def evaluate_system(self, qa_file_path: str, top_k: int = 3, 
                       save_results_file: str = None, run_ablation: bool = False):
        """Evaluate the RAG system on QA pairs"""
        if not self.is_built:
            raise ValueError("System not built. Call build_system or load_system first.")
        
        # Load QA pairs
        if not os.path.exists(qa_file_path):
            logger.warning(f"QA file '{qa_file_path}' not found. Creating sample file...")
            qa_file_path = evaluate.create_sample_qa_file()
        
        qa_pairs = evaluate.load_qa_pairs(qa_file_path)
        
        # Run evaluation
        logger.info(f"Running evaluation with top_k={top_k}...")
        results = evaluate.evaluate_rag_system(self, qa_pairs, top_k=top_k)
        
        # Print results
        evaluate.print_results(results)
        
        # Save results if requested
        if save_results_file:
            evaluate.save_results(results, save_results_file)
        
        # # Run ablation study if requested
        # if run_ablation and len(qa_pairs) >= 5:
        #     ablation_results = evaluate.run_ablation_study(self, qa_pairs)
        #     return results, ablation_results
        
        return results
    
    def save_system(self, save_dir: str):
        """Save the complete system"""
        self.retriever.save_index(save_dir)
        
        # Save config
        config_dict = {
            'retriever_model': self.config.retriever_model,
            'generator_model': self.config.generator_model,
            'generator_type': self.config.generator_type,
            'chunk_size': self.config.chunk_size,
            'overlap': self.config.overlap,
            'top_k': self.config.top_k,
            'max_length': self.config.max_length
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"RAG system saved to {save_dir}")
    
    def load_system(self, save_dir: str):
        """Load the complete system"""
        if self.is_built:
            logger.info("System already built - skipping load")
            return
        
        self.retriever.load_index(save_dir)
        self.is_built = True
        logger.info(f"RAG system loaded from {save_dir}")

def main():
    """Example usage of the RAG system"""
    
    config = MODEL_CONFIGS["medium_balanced"]
    rag = RAGSystem(config)
    
    # Build system from crawled data
    crawled_data_path = "eecs_20250606_text_bs_rewritten.jsonl"
    
    if os.path.exists("rag_index/faiss_index.idx"):
        print("Loading existing RAG system...")
        rag.load_system("rag_index")
    else:
        print("Building new RAG system...")
        rag.build_system(crawled_data_path)
        rag.save_system("rag_index")
        
    print("\n" + "="*80)
    print("EVALUATION MODE")
    print("="*80)
    
    # Run evaluation on QA pairs
    if not os.path.exists("eval"):
        os.makedirs("eval")
    
    evaluation_results = rag.evaluate_system(
        qa_file_path="ucb_eecs_rag_eval_dataset.jsonl",  # Your QA pairs file
        top_k=3,
        save_results_file="eval/evaluation_results_rerank.json",
        run_ablation=True
    )
    
    print(f"\nEvaluation completed! Check evaluation_results.json for detailed results.")
    test_questions = [
        "Who is the EECS department chair?",
        "Where is the chair's office?", 
        "When was the chair appointed?"
    ]
    
    print("\n" + "="*80)
    print("RAG SYSTEM DEMO")
    print("="*80)
        
    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = rag.answer_question(question)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['sources'])} documents")
        print("-" * 40)

if __name__ == "__main__":
    main()