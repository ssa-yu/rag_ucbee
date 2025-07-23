import torch
import logging
from typing import List, Tuple
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model components"""
    retriever_model: str = "all-MiniLM-L6-v2"
    generator_model: str = "google/flan-t5-base"
    generator_type: str = "seq2seq"  # "seq2seq" or "causal"
    chunk_size: int = 400
    overlap: int = 50
    top_k: int = 3
    max_length: int = 150

class BaseGenerator(ABC):
    """Abstract base class for generators"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def generate_answer(self, question: str, retrieved_docs: List[Tuple[any, float]], max_length: int = 150) -> str:
        pass

class Seq2SeqGenerator(BaseGenerator):
    """Seq2Seq generator (T5, BART, etc.)"""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        super().__init__(model_name)
        
        logger.info(f"Loading seq2seq generator: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
        except:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.device = "cpu"
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
    
    def generate_answer(self, question: str, retrieved_docs: List[Tuple[any, float]], max_length: int = 150) -> str:
        """Generate answer using seq2seq model"""
        context_parts = []
        for doc, score in retrieved_docs:
            context_parts.append(f"Document: {doc.content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Context information:
        {context}

        Question: {question}

        Based on the context above, provide a concise and accurate answer:
        Answer:"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2000)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

class CausalLMGenerator(BaseGenerator):
    """Causal LM generator (GPT-style models)"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        super().__init__(model_name)
        
        logger.info(f"Loading causal LM generator: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
        except:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.device = "cpu"
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
    
    def generate_answer(self, question: str, retrieved_docs: List[Tuple[any, float]], max_length: int = 150) -> str:
        """Generate answer using causal LM"""
        context_parts = []
        for doc, score in retrieved_docs:
            context_parts.append(f"Document: {doc.content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Context information:
        {context}

        Question: {question}

        Answer:"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2000)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        # Extract only the generated part
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

class GeneratorFactory:
    """Factory for creating generators"""
    
    @staticmethod
    def create_generator(generator_type: str, model_name: str) -> BaseGenerator:
        if generator_type == "seq2seq":
            return Seq2SeqGenerator(model_name)
        elif generator_type == "causal":
            return CausalLMGenerator(model_name)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")

# Predefined model configurations
MODEL_CONFIGS = {
    "small_fast": ModelConfig(
        retriever_model="all-MiniLM-L6-v2",
        generator_model="google/flan-t5-small",
        generator_type="seq2seq",
        chunk_size=500,
        top_k=3
    ),
    "medium_balanced": ModelConfig(
        retriever_model="all-MiniLM-L6-v2", 
        generator_model="google/flan-t5-base",
        generator_type="seq2seq",
        chunk_size=500,
        top_k=3
    ),
    "large_quality": ModelConfig(
        retriever_model="all-MiniLM-L6-v2",
        generator_model="google/flan-t5-large", 
        generator_type="seq2seq",
        chunk_size=500,
        top_k=5
    ),
    # Failed
    "causal_small": ModelConfig(
        retriever_model="all-MiniLM-L6-v2",
        generator_model="microsoft/DialoGPT-small",
        generator_type="causal",
        chunk_size=500,
        top_k=5
    )
}