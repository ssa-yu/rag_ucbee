import json
import os
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import faiss
from IPython import embed
import torch
import logging
from dataclasses import dataclass
import pickle
import evaluate
import networkx as nx
from collections import defaultdict, Counter
import spacy
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from models import ModelConfig, GeneratorFactory, MODEL_CONFIGS
from rank_bm25 import BM25Okapi

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

@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    id: str
    name: str
    type: str
    mentions: List[str]
    documents: Set[str]
    embedding: Optional[np.ndarray] = None

@dataclass
class Relation:
    """Represents a relation between entities"""
    id: str
    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float
    documents: Set[str]

@dataclass
class Community:
    """Represents a community of entities"""
    id: str
    entities: Set[str]
    summary: str
    level: int
    documents: Set[str]

class EntityExtractor:
    """Extracts entities and relations from documents"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.encoder = SentenceTransformer(model_name)
        self.entity_types = {
            "PERSON": "person",
            "ORG": "organization", 
            "GPE": "location",
            "PRODUCT": "product",
            "EVENT": "event",
            "LAW": "law",
            "LANGUAGE": "language",
            "DATE": "date",
            "TIME": "time",
            "MONEY": "money",
            "QUANTITY": "quantity",
            "ORDINAL": "ordinal",
            "CARDINAL": "cardinal"
        }
    
    def extract_entities(self, documents: List[Document]) -> Dict[str, Entity]:
        """Extract entities from documents"""
        if not self.nlp:
            return {}
        
        logger.info("Extracting entities from documents...")
        entities = {}
        entity_counter = 0
        
        for doc in documents:
            # Process document with spaCy
            spacy_doc = self.nlp(doc.content)
            
            for ent in spacy_doc.ents:
                # Normalize entity name
                entity_name = ent.text.strip().lower()
                entity_type = self.entity_types.get(ent.label_, "other")
                
                # Skip very short entities or common words
                if len(entity_name) < 2 or entity_name in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']:
                    continue
                
                # Create entity ID
                entity_id = f"entity_{entity_name.replace(' ', '_')}"
                
                if entity_id not in entities:
                    entities[entity_id] = Entity(
                        id=entity_id,
                        name=entity_name,
                        type=entity_type,
                        mentions=[ent.text],
                        documents={doc.id}
                    )
                else:
                    entities[entity_id].mentions.append(ent.text)
                    entities[entity_id].documents.add(doc.id)
        
        # Generate embeddings for entities
        entity_names = [entity.name for entity in entities.values()]
        if entity_names:
            embeddings = self.encoder.encode(entity_names)
            for i, entity in enumerate(entities.values()):
                entity.embedding = embeddings[i]
        
        logger.info(f"Extracted {len(entities)} unique entities")
        return entities
    
    def extract_relations(self, documents: List[Document], entities: Dict[str, Entity]) -> List[Relation]:
        """Extract relations between entities"""
        if not self.nlp:
            return []
        
        logger.info("Extracting relations between entities...")
        relations = []
        relation_counter = 0
        
        # Simple co-occurrence based relation extraction
        for doc in documents:
            doc_entities = [ent_id for ent_id, entity in entities.items() 
                           if doc.id in entity.documents]
            
            # Create relations between co-occurring entities
            for i, ent1_id in enumerate(doc_entities):
                for ent2_id in doc_entities[i+1:]:
                    relation = Relation(
                        id=f"rel_{relation_counter}",
                        source_entity=ent1_id,
                        target_entity=ent2_id,
                        relation_type="co_occurs",
                        confidence=0.5,  # Simple co-occurrence confidence
                        documents={doc.id}
                    )
                    relations.append(relation)
                    relation_counter += 1
        
        logger.info(f"Extracted {len(relations)} relations")
        return relations

class CommunityDetector:
    """Detects communities in the knowledge graph"""
    
    def __init__(self, max_communities: int = 50):
        self.max_communities = max_communities
    
    def detect_communities(self, entities: Dict[str, Entity], relations: List[Relation]) -> List[Community]:
        """Detect communities using graph clustering"""
        logger.info("Detecting communities in knowledge graph...")
        
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add nodes (entities)
        for entity_id, entity in entities.items():
            G.add_node(entity_id, name=entity.name, type=entity.type)
        
        # Add edges (relations)
        for relation in relations:
            if relation.source_entity in entities and relation.target_entity in entities:
                G.add_edge(relation.source_entity, relation.target_entity, 
                          weight=relation.confidence)
        
        # Detect communities using Louvain algorithm
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G)
        except ImportError:
            logger.warning("python-louvain not installed. Using simple connected components.")
            components = list(nx.connected_components(G))
            partition = {}
            for i, component in enumerate(components):
                for node in component:
                    partition[node] = i
        
        # Group entities by community
        communities_dict = defaultdict(set)
        for entity_id, community_id in partition.items():
            communities_dict[community_id].add(entity_id)
        
        # Create Community objects
        communities = []
        for i, (community_id, entity_ids) in enumerate(communities_dict.items()):
            if i >= self.max_communities:
                break
                
            # Get all documents for this community
            community_docs = set()
            for entity_id in entity_ids:
                community_docs.update(entities[entity_id].documents)
            
            # Generate community summary
            entity_names = [entities[eid].name for eid in entity_ids]
            summary = f"Community of {len(entity_ids)} entities: {', '.join(entity_names[:5])}"
            if len(entity_names) > 5:
                summary += f" and {len(entity_names) - 5} others"
            
            community = Community(
                id=f"community_{i}",
                entities=entity_ids,
                summary=summary,
                level=0,  # Single level for now
                documents=community_docs
            )
            communities.append(community)
        
        logger.info(f"Detected {len(communities)} communities")
        return communities

class GraphRAGRetriever:
    """Graph-enhanced retriever that uses knowledge graph for retrieval"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.entities = {}
        self.relations = []
        self.communities = []
        self.entity_index = None
        self.community_index = None
        self.documents = []
        
    def build_graph(self, documents: List[Document]):
        """Build knowledge graph from documents"""
        self.documents = documents
        
        # Extract entities and relations
        extractor = EntityExtractor()
        self.entities = extractor.extract_entities(documents)
        self.relations = extractor.extract_relations(documents, self.entities)
        
        # Detect communities
        detector = CommunityDetector()
        self.communities = detector.detect_communities(self.entities, self.relations)
        
        # Build FAISS indices
        self._build_entity_index()
        self._build_community_index()
    
    def _build_entity_index(self):
        """Build FAISS index for entities"""
        if not self.entities:
            return
        
        embeddings = []
        entity_list = []
        
        for entity in self.entities.values():
            if entity.embedding is not None:
                embeddings.append(entity.embedding)
                entity_list.append(entity)
        
        if embeddings:
            embeddings_array = np.vstack(embeddings)
            dimension = embeddings_array.shape[1]
            
            self.entity_index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings_array)
            self.entity_index.add(embeddings_array)
            self.entity_list = entity_list
    
    def _build_community_index(self):
        """Build FAISS index for communities"""
        if not self.communities:
            return
        
        # Generate embeddings for community summaries
        summaries = [community.summary for community in self.communities]
        embeddings = self.encoder.encode(summaries)
        
        dimension = embeddings.shape[1]
        self.community_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.community_index.add(embeddings)
    
    def retrieve_entities(self, query: str, top_k: int = 5) -> List[Tuple[Entity, float]]:
        """Retrieve relevant entities"""
        if self.entity_index is None:
            return []
        
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.entity_index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.entity_list):
                results.append((self.entity_list[idx], float(score)))
        
        return results
    
    def retrieve_communities(self, query: str, top_k: int = 3) -> List[Tuple[Community, float]]:
        """Retrieve relevant communities"""
        if self.community_index is None:
            return []
        
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.community_index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.communities):
                results.append((self.communities[idx], float(score)))
        
        return results
    
    def retrieve_documents_from_graph(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve documents using graph-based approach"""
        # Get relevant entities and communities
        relevant_entities = self.retrieve_entities(query, top_k)
        relevant_communities = self.retrieve_communities(query, top_k//2)
        
        # Collect document IDs from entities and communities
        doc_scores = defaultdict(float)
        
        # Score documents based on entity relevance
        for entity, entity_score in relevant_entities:
            for doc_id in entity.documents:
                doc_scores[doc_id] += entity_score * 0.7
        
        # Score documents based on community relevance
        for community, community_score in relevant_communities:
            for doc_id in community.documents:
                doc_scores[doc_id] += community_score * 0.3
        
        # Convert to document objects and sort
        doc_dict = {doc.id: doc for doc in self.documents}
        results = []
        
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            if doc_id in doc_dict:
                results.append((doc_dict[doc_id], score))
        
        return results

class BM25Retriever:
    """Simple sparse keyword-based retriever using BM25"""

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.corpus = [doc.content.split() for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[idx], float(scores[idx])) for idx in ranked_indices]

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

class EnhancedRAGSystem:
    """Enhanced RAG system with GraphRAG capabilities"""
    
    def __init__(self, config: ModelConfig, use_graph: bool = True):
        self.config = config
        self.use_graph = use_graph
        self.chunker = DocumentChunker(config.chunk_size, config.overlap)
        self.embedding_retriever = EmbeddingRetriever(config.retriever_model)
        self.generator = GeneratorFactory.create_generator(config.generator_type, config.generator_model)
        self.is_built = False
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.bm25_retriever = None
        
        # GraphRAG components
        if self.use_graph:
            self.graph_retriever = GraphRAGRetriever(config.retriever_model)
    
    def build_system(self, crawled_data_path: str):
        """Build the complete RAG system from crawled data"""
        if self.is_built:
            logger.info("System already built - skipping build")
            return
        
        logger.info("Building Enhanced RAG system with GraphRAG...")
        
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
        
        # Build traditional retrieval indices
        self.embedding_retriever.build_index(documents)
        self.bm25_retriever = BM25Retriever(documents)
        
        # Build knowledge graph if enabled
        if self.use_graph:
            self.graph_retriever.build_graph(documents)

        self.is_built = True
        logger.info("Enhanced RAG system built successfully!")
    
    def answer_question(self, question: str, top_k: int = 3, use_graph: bool = None) -> Dict:
        """Answer a question using the enhanced RAG system"""
        if not self.is_built:
            raise ValueError("System not built. Call build_system first.")

        use_graph = use_graph if use_graph is not None else self.use_graph
        k = top_k or self.config.top_k

        # 🔹 Step 1: Retrieve from multiple sources
        dense_results = self.embedding_retriever.retrieve(question, k)
        sparse_results = self.bm25_retriever.retrieve(question, k)
        
        all_results = dense_results + sparse_results
        retrieval_methods = ["Dense", "Sparse"]
        
        # Add GraphRAG retrieval if enabled
        if use_graph and hasattr(self, 'graph_retriever'):
            try:
                graph_results = self.graph_retriever.retrieve_documents_from_graph(question, k)
                all_results.extend(graph_results)
                retrieval_methods.append("Graph")
                logger.info(f"GraphRAG retrieved {len(graph_results)} documents")
            except Exception as e:
                logger.warning(f"GraphRAG retrieval failed: {e}")

        # 🔹 Step 2: Merge and deduplicate by doc ID
        seen_ids = {}
        for doc, score in all_results:
            if doc.id not in seen_ids:
                seen_ids[doc.id] = (doc, score)
            else:
                # Keep the higher score
                if score > seen_ids[doc.id][1]:
                    seen_ids[doc.id] = (doc, score)
        
        merged_results = list(seen_ids.values())
        merged_results = sorted(merged_results, key=lambda x: x[1], reverse=True)[:k * 3]

        # 🔹 Step 3: Rerank top documents using cross-encoder
        pairs = [(question, doc.content) for doc, _ in merged_results]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(merged_results, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = [(doc, score) for (doc, _), score in reranked[:k]]

        # 🔹 Step 4: Generate enhanced context if using graph
        context_enhancement = ""
        if use_graph and hasattr(self, 'graph_retriever'):
            context_enhancement = self._generate_graph_context(question)

        # 🔹 Step 5: Generate answer with enhanced context
        if context_enhancement:
            # Modify the question to include graph context
            enhanced_question = f"{question}\n\nAdditional context: {context_enhancement}"
            answer = self.generator.generate_answer(enhanced_question, reranked_docs, self.config.max_length)
        else:
            answer = self.generator.generate_answer(question, reranked_docs, self.config.max_length)

        print("\n===== LLM Generated Answer =====")
        print(answer)
        print("================================\n")

        # 🔹 Step 6: Prepare sources metadata
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
            'retrieval_methods': retrieval_methods,
            'graph_enhanced': use_graph,
            'context_enhancement': context_enhancement if context_enhancement else None,
            'config': {
                'retriever_model': self.config.retriever_model,
                'generator_model': self.config.generator_model,
                'generator_type': self.config.generator_type,
                'use_graph': use_graph
            }
        }
    
    def _generate_graph_context(self, question: str) -> str:
        """Generate additional context from knowledge graph"""
        if not hasattr(self, 'graph_retriever'):
            return ""
        
        try:
            # Get relevant entities and communities
            entities = self.graph_retriever.retrieve_entities(question, 3)
            communities = self.graph_retriever.retrieve_communities(question, 2)
            
            context_parts = []
            
            if entities:
                entity_info = []
                for entity, score in entities:
                    entity_info.append(f"{entity.name} ({entity.type})")
                context_parts.append(f"Key entities: {', '.join(entity_info)}")
            
            if communities:
                community_info = []
                for community, score in communities:
                    community_info.append(community.summary)
                context_parts.append(f"Related topics: {'; '.join(community_info)}")
            
            return " | ".join(context_parts) if context_parts else ""
            
        except Exception as e:
            logger.warning(f"Failed to generate graph context: {e}")
            return ""
    
    def get_graph_statistics(self) -> Dict:
        """Get statistics about the knowledge graph"""
        if not self.use_graph or not hasattr(self, 'graph_retriever'):
            return {"error": "GraphRAG not enabled"}
        
        stats = {
            "num_entities": len(self.graph_retriever.entities),
            "num_relations": len(self.graph_retriever.relations),
            "num_communities": len(self.graph_retriever.communities),
            "entity_types": {}
        }
        
        # Count entity types
        for entity in self.graph_retriever.entities.values():
            entity_type = entity.type
            stats["entity_types"][entity_type] = stats["entity_types"].get(entity_type, 0) + 1
        
        return stats
    
    def evaluate_system(self, qa_file_path: str, top_k: int = 3, 
                       save_results_file: str = None, run_ablation: bool = False,
                       compare_graph: bool = True):
        """Evaluate the enhanced RAG system"""
        if not self.is_built:
            raise ValueError("System not built. Call build_system or load_system first.")
        
        # Load QA pairs
        if not os.path.exists(qa_file_path):
            logger.warning(f"QA file '{qa_file_path}' not found. Creating sample file...")
            qa_file_path = evaluate.create_sample_qa_file()
        
        qa_pairs = evaluate.load_qa_pairs(qa_file_path)
        
        results = {}
        
        # Evaluate without graph
        logger.info("Evaluating without GraphRAG...")
        results_no_graph = evaluate.evaluate_rag_system_enhanced(
            self, qa_pairs, top_k=top_k, use_graph=False
        )
        results['without_graph'] = results_no_graph
        
        # Evaluate with graph if enabled
        if self.use_graph and compare_graph:
            logger.info("Evaluating with GraphRAG...")
            results_with_graph = evaluate.evaluate_rag_system_enhanced(
                self, qa_pairs, top_k=top_k, use_graph=True
            )
            results['with_graph'] = results_with_graph
            
            # Compare results
            self._compare_results(results_no_graph, results_with_graph)
        
        # Print and save results
        if compare_graph and self.use_graph:
            evaluate.print_comparison_results(results)
        else:
            evaluate.print_single_results(results_no_graph)
        
        # Save results if requested
        if save_results_file:
            with open(save_results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def _compare_results(self, results_no_graph: Dict, results_with_graph: Dict):
        """Compare results between traditional RAG and GraphRAG"""
        print("\n" + "="*60)
        print("GRAPH RAG COMPARISON")
        print("="*60)
        
        metrics = ['avg_relevance', 'avg_faithfulness', 'avg_answer_similarity', 'retrieval_precision']
        
        for metric in metrics:
            if metric in results_no_graph and metric in results_with_graph:
                no_graph_val = results_no_graph[metric]
                with_graph_val = results_with_graph[metric]
                improvement = ((with_graph_val - no_graph_val) / no_graph_val) * 100 if no_graph_val > 0 else 0
                
                print(f"{metric.replace('_', ' ').title()}:")
                print(f"  Without Graph: {no_graph_val:.3f}")
                print(f"  With Graph:    {with_graph_val:.3f}")
                print(f"  Improvement:   {improvement:+.1f}%")
                print()
    
    def save_system(self, save_dir: str):
        """Save the complete enhanced system"""
        self.embedding_retriever.save_index(save_dir)
        
        # Save GraphRAG components if enabled
        if self.use_graph and hasattr(self, 'graph_retriever'):
            graph_dir = os.path.join(save_dir, "graph")
            os.makedirs(graph_dir, exist_ok=True)
            
            # Save entities
            with open(os.path.join(graph_dir, "entities.pkl"), "wb") as f:
                pickle.dump(self.graph_retriever.entities, f)
            
            # Save relations
            with open(os.path.join(graph_dir, "relations.pkl"), "wb") as f:
                pickle.dump(self.graph_retriever.relations, f)
            
            # Save communities
            with open(os.path.join(graph_dir, "communities.pkl"), "wb") as f:
                pickle.dump(self.graph_retriever.communities, f)
            
            # Save graph indices if they exist
            if self.graph_retriever.entity_index is not None:
                faiss.write_index(self.graph_retriever.entity_index, 
                                os.path.join(graph_dir, "entity_index.idx"))
            
            if self.graph_retriever.community_index is not None:
                faiss.write_index(self.graph_retriever.community_index, 
                                os.path.join(graph_dir, "community_index.idx"))
        
        # Save config
        config_dict = {
            'retriever_model': self.config.retriever_model,
            'generator_model': self.config.generator_model,
            'generator_type': self.config.generator_type,
            'chunk_size': self.config.chunk_size,
            'overlap': self.config.overlap,
            'top_k': self.config.top_k,
            'max_length': self.config.max_length,
            'use_graph': self.use_graph
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Enhanced RAG system saved to {save_dir}")
    
    def load_system(self, save_dir: str):
        """Load the complete enhanced system"""
        if self.is_built:
            logger.info("System already built - skipping load")
            return
        
        self.embedding_retriever.load_index(save_dir)
        self.bm25_retriever = BM25Retriever(self.embedding_retriever.documents)
        
        # Load GraphRAG components if they exist
        graph_dir = os.path.join(save_dir, "graph")
        if self.use_graph and os.path.exists(graph_dir):
            try:
                # Load entities
                with open(os.path.join(graph_dir, "entities.pkl"), "rb") as f:
                    entities = pickle.load(f)
                
                # Load relations
                with open(os.path.join(graph_dir, "relations.pkl"), "rb") as f:
                    relations = pickle.load(f)
                
                # Load communities
                with open(os.path.join(graph_dir, "communities.pkl"), "rb") as f:
                    communities = pickle.load(f)
                
                # Initialize graph retriever and set data
                self.graph_retriever.entities = entities
                self.graph_retriever.relations = relations
                self.graph_retriever.communities = communities
                self.graph_retriever.documents = self.embedding_retriever.documents
                
                # Load indices if they exist
                entity_index_path = os.path.join(graph_dir, "entity_index.idx")
                if os.path.exists(entity_index_path):
                    self.graph_retriever.entity_index = faiss.read_index(entity_index_path)
                    # Rebuild entity list
                    self.graph_retriever.entity_list = [entity for entity in entities.values() 
                                                      if entity.embedding is not None]
                
                community_index_path = os.path.join(graph_dir, "community_index.idx")
                if os.path.exists(community_index_path):
                    self.graph_retriever.community_index = faiss.read_index(community_index_path)
                
                logger.info("GraphRAG components loaded successfully")
                
            except Exception as e:
                logger.warning(f"Failed to load GraphRAG components: {e}")
                self.use_graph = False

        self.is_built = True
        logger.info(f"Enhanced RAG system loaded from {save_dir}")


def main():
    """Example usage of the Enhanced RAG system with GraphRAG"""
    
    config = MODEL_CONFIGS["medium_balanced"]
    
    # Initialize with GraphRAG enabled
    rag = EnhancedRAGSystem(config, use_graph=True)
    
    # Build system from crawled data
    crawled_data_path = "eecs_20250606_text_bs_rewritten.jsonl"
    
    if os.path.exists("enhanced_rag_index/faiss_index.idx"):
        print("Loading existing Enhanced RAG system...")
        rag.load_system("enhanced_rag_index")
    else:
        print("Building new Enhanced RAG system with GraphRAG...")
        rag.build_system(crawled_data_path)
        rag.save_system("enhanced_rag_index")
    
    # Print graph statistics
    if rag.use_graph:
        stats = rag.get_graph_statistics()
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("="*60)
        print(f"📊 Entities: {stats.get('num_entities', 0)}")
        print(f"🔗 Relations: {stats.get('num_relations', 0)}")
        print(f"👥 Communities: {stats.get('num_communities', 0)}")
        print(f"🏷️ Entity Types: {stats.get('entity_types', {})}")
        
    print("\n" + "="*80)
    print("EVALUATION MODE")
    print("="*80)
    
    # Run evaluation comparing traditional RAG vs GraphRAG
    if not os.path.exists("eval"):
        os.makedirs("eval")
    
    evaluation_results = rag.evaluate_system(
        qa_file_path="ucb_eecs_rag_eval_dataset.jsonl",
        top_k=3,
        save_results_file="eval/enhanced_evaluation_results.json",
        compare_graph=True
    )
    
    print(f"\nEvaluation completed! Check enhanced_evaluation_results.json for detailed results.")
    
    # Demo questions
    test_questions = [
        "Who is the EECS department chair?",
        "What research areas does EECS focus on?", 
        "How are different faculty members connected through their research?",
        "What are the main research communities in EECS?"
    ]
    
    print("\n" + "="*80)
    print("ENHANCED RAG SYSTEM DEMO")
    print("="*80)
        
    for question in test_questions:
        print(f"\nQuestion: {question}")
        
        # Answer with GraphRAG
        result = rag.answer_question(question, use_graph=True)
        print(f"Answer (with Graph): {result['answer']}")
        if result.get('context_enhancement'):
            print(f"Graph Context: {result['context_enhancement']}")
        print(f"Sources: {len(result['sources'])} documents")
        print(f"Methods: {', '.join(result['retrieval_methods'])}")
        print("-" * 60)

if __name__ == "__main__":
    main()