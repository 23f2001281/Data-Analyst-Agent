"""
Vector Embeddings Pipeline for Scraper Agent

This script creates vector embeddings for scraped web content and stores them
in Pinecone vector database for efficient similarity search.
"""

import os
import json
import pandas as pd
import google.generativeai as genai
import asyncio
from pinecone import Pinecone
import re
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any
from collections import Counter
from semantic_chunker import semantic_chunker

load_dotenv()

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "data-analyst-agent-embedder"

def clean_text(text: str) -> str:
    """Clean and normalize text for better matching."""
    if not text:
        return ""
    
    # Remove HTML tags but preserve important formatting
    text = re.sub(r'<div[^>]*>', ' ', text)
    text = re.sub(r'</div>', ' ', text)
    text = re.sub(r'<span[^>]*>', ' ', text)
    text = re.sub(r'</span>', ' ', text)
    text = re.sub(r'<a[^>]*>', ' ', text)
    text = re.sub(r'</a>', ' ', text)
    text = re.sub(r'<p>', ' ', text)
    text = re.sub(r'</p>', ' ', text)
    text = re.sub(r'<br>', ' ', text)
    text = re.sub(r'<strong>', ' ', text)
    text = re.sub(r'</strong>', ' ', text)
    text = re.sub(r'<em>', ' ', text)
    text = re.sub(r'</em>', ' ', text)
    # Normalize whitespace but preserve line breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,!?@-]', ' ', text)
    return text.strip()

def _extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    """Extracts the most common non-stop-words from text, including capitalized words and phrases."""
    stop_words = set([
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "with", "is", "was", "it", "that", "as", "by",
        "are", "were", "this", "from", "at", "about", "s", "t", "i", "me", "my", "myself", "we", "our", "ours",
        "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
        "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
        "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
        "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    ])
    
    keywords = []
    
    # 1. Extract consecutive capitalized words (e.g., "National Parivar Mediclaim Plus Policy")
    capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    for phrase in capitalized_phrases:
        if len(phrase.split()) >= 2:  # Only multi-word phrases
            keywords.append(phrase.lower())
    
    # 2. Extract single capitalized words (e.g., "Policy", "Premium", "Grace")
    single_capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    for word in single_capitalized:
        if word.lower() not in stop_words:
            keywords.append(word.lower())
    
    # 3. Extract regular words (existing logic)
    words = re.findall(r'\b\w{3,}\b', text.lower())
    words = [word for word in words if word not in stop_words]
    keywords.extend(words)
    
    if not keywords:
        return []
    
    # Count frequencies and return most common
    keyword_counts = Counter(keywords)
    return [keyword for keyword, freq in keyword_counts.most_common(num_keywords)]

async def generate_table_summary(table_metadata: Dict[str, Any]) -> str:
    """
    Generate a natural language summary of a table using Gemini.
    """
    try:
        # Load the table
        df = pd.read_csv(table_metadata["table_path"])
        
        # Create a preview of the table in markdown format
        table_preview = df.head(5).to_markdown(index=False)
        
        # Create a rich prompt for table summarization
        prompt = f"""You are a data analyst. Below is a table extracted from a webpage. Your task is to write a concise, dense, natural-language summary of this table. The summary should describe what the table is about, what its columns represent, and mention any key patterns or important data points you can observe from the preview.

This summary will be used for semantic search, so it should be rich with keywords and concepts that someone might search for.

Table Source: {table_metadata.get('source_url', 'Unknown')}
Number of Rows: {table_metadata['num_rows']}
Number of Columns: {table_metadata['num_columns']}
Columns: {', '.join(table_metadata['columns'])}

Table Preview (first 5 rows):
```markdown
{table_preview}
```

Write a comprehensive summary in 2-3 sentences that captures the essence and content of this table:"""

        # Initialize Gemini if not already done
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not genai.api_key:
            genai.configure(api_key=GEMINI_API_KEY)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = await model.generate_content_async(prompt)
        
        summary = response.text.strip()
        print(f"📝 Generated table summary: {summary[:100]}...")
        
        return summary
        
    except Exception as e:
        print(f"⚠️ Error generating table summary: {str(e)}")
        # Fallback to a basic summary
        columns_str = ', '.join(table_metadata['columns'][:5])  # First 5 columns
        return f"A data table with {table_metadata['num_rows']} rows and {table_metadata['num_columns']} columns. Columns include: {columns_str}. Source: {table_metadata.get('source_url', 'Unknown source')}."

def create_structured_chunks(text: str, max_chunk_size: int = 3000) -> List[Dict[str, Any]]:
    """
    Create structured chunks using advanced semantic chunking.
    """
    if not text or not text.strip():
        return []
    
    # Clean the text first
    cleaned_text = clean_text(text)
    
    # Use advanced semantic chunking with overlap
    chunks = advanced_semantic_chunker(cleaned_text, max_chunk_size, overlap=200)
    
    structured_chunks = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
                continue
            
        # Extract keywords for this chunk
        keywords = _extract_keywords(chunk, num_keywords=5)
        
        structured_chunk = {
            "content": chunk,
            "chunk_id": f"chunk_{i}_{hash(chunk) % 10000}",
            "keywords": keywords,
            "length": len(chunk),
            "metadata": {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunking_method": "advanced_semantic"
            }
        }
        
        structured_chunks.append(structured_chunk)
    
    print(f"📦 Created {len(structured_chunks)} structured chunks using advanced semantic chunking")
    return structured_chunks

def semantic_chunker(content: str, max_chars: int = 3000) -> list:
    """
    Advanced semantic chunking with multiple strategies:
    1. Paragraph-based chunking (primary)
    2. Semantic boundary detection
    3. Context preservation
    4. Overlap for continuity
    """
    if len(content) <= max_chars:
        return [content]
    
    chunks = []
    current_chunk = ""
    
    # Split by paragraphs first
    paragraphs = content.split('\n\n')
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed limit
        if len(current_chunk) + len(paragraph) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        
    # Add the last chunk
    if current_chunk:
            chunks.append(current_chunk.strip())
    
    # If any chunk is still too long, split by sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            # Split by sentences with better regex
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            current_sentence_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if len(current_sentence_chunk) + len(sentence) > max_chars and current_sentence_chunk:
                    final_chunks.append(current_sentence_chunk.strip())
                    current_sentence_chunk = sentence
                else:
                    if current_sentence_chunk:
                        current_sentence_chunk += ". " + sentence
                    else:
                        current_sentence_chunk = sentence
            
            if current_sentence_chunk:
                final_chunks.append(current_sentence_chunk.strip())
    
    return final_chunks

def advanced_semantic_chunker(content: str, max_chars: int = 3000, overlap: int = 200) -> list:
    """
    Advanced semantic chunking with overlap and context preservation.
    This is the new improved chunking strategy.
    """
    if len(content) <= max_chars:
        return [content]
    
    chunks = []
    
    # Strategy 1: Try paragraph-based chunking first
    paragraphs = content.split('\n\n')
    current_chunk = ""
    
    for i, paragraph in enumerate(paragraphs):
        # Check if adding this paragraph would exceed limit
        if len(current_chunk) + len(paragraph) > max_chars and current_chunk:
            # Add current chunk
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous
            if overlap > 0 and chunks:
                # Get last few sentences from previous chunk for overlap
                last_chunk = chunks[-1]
                overlap_text = _extract_overlap_text(last_chunk, overlap)
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Strategy 2: Handle oversized chunks with sentence splitting
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            # Split by sentences with semantic boundaries
            sentence_chunks = _split_by_semantic_boundaries(chunk, max_chars, overlap)
            final_chunks.extend(sentence_chunks)
    
    return final_chunks

def _extract_overlap_text(text: str, overlap_chars: int) -> str:
    """Extract the last portion of text for overlap purposes."""
    if len(text) <= overlap_chars:
        return text
    
    # Try to break at sentence boundary
    sentences = re.split(r'(?<=[.!?])\s+', text)
    overlap_text = ""
    
    for sentence in reversed(sentences):
        if len(overlap_text + sentence) <= overlap_chars:
            overlap_text = sentence + " " + overlap_text
        else:
            break
    
    return overlap_text.strip()

def _split_by_semantic_boundaries(text: str, max_chars: int, overlap: int = 0) -> list:
    """Split text by semantic boundaries (sentences, clauses, etc.)."""
    chunks = []
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def create_pinecone_index():
    """
    Create or get the Pinecone index and clear all existing data.
    Note: The index must be created manually through the Pinecone console with integrated inference enabled.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    try:
        index = pc.Index(INDEX_NAME)
        
        # Clear all existing data from the index by checking for namespace first
        try:
            stats = index.describe_index_stats()
            if "scraped-content" in stats.namespaces:
                index.delete(namespace="scraped-content", delete_all=True)

        except Exception as e:
            print(f"⚠️  Could not clear data: {e}")
        
        # Test if integrated inference is working
        try:
            test_record = {"id": "test", "text": "test"}
            index.upsert_records(namespace="test", records=[test_record])
            # Clean up the test namespace
            index.delete(namespace="test", delete_all=True)
            return index
        except Exception as e:
            if "integrated inference is not configured" in str(e).lower():
                print("ERROR: Index exists but doesn't have integrated inference configured.")
                print("Please create the index manually through the Pinecone console:")
                print("1. Go to https://app.pinecone.io/")
                print("2. Create a new index named 'llama-text-embed-v2-index'")
                print("3. Set dimension to 1024")
                print("4. Set metric to 'cosine'")
                print("5. Enable 'Integrated inference' and select 'llama-text-embed-v2' model")
                print("6. Choose serverless with AWS us-east-1")
                raise Exception("Index needs to be created manually with integrated inference")
            else:
                raise e
        
    except Exception as e:
        if "not found" in str(e).lower() or "404" in str(e):
            print("ERROR: Index 'llama-text-embed-v2-index' not found.")
            print("Please create the index manually through the Pinecone console:")
            print("1. Go to https://app.pinecone.io/")
            print("2. Create a new index named 'llama-text-embed-v2-index'")
            print("3. Set dimension to 1024")
            print("4. Set metric to 'cosine'")
            print("5. Enable 'Integrated inference' and select 'llama-text-embed-v2' model")
            print("6. Choose serverless with AWS us-east-1")
            raise Exception("Index needs to be created manually with integrated inference")
        else:
            raise e

def process_scraped_content(scraped_data: Dict[str, Any], user_prompt: str) -> List[Dict[str, Any]]:
    """
    Enhanced version that processes both text content and tables from scraped data.
    """
    all_chunks = []
    
    # Process text content (existing logic)
    if scraped_data.get("scraped_content"):
        for page in scraped_data["scraped_content"]:
            content = page.get("content", "")
            if content:
                # Create structured chunks for text content
                text_chunks = create_structured_chunks(content)
                
                # Add metadata to each chunk
                for chunk in text_chunks:
                    chunk["metadata"] = {
                        "content_type": "text",
                        "source_url": page.get("url", ""),
                        "page_title": page.get("title", ""),
                        "keywords": chunk.get("keywords", [])
                    }
                
                all_chunks.extend(text_chunks)
    
    # Process tables (new logic)
    if scraped_data.get("tables"):
        print(f"📊 Processing {len(scraped_data['tables'])} tables for embedding...")
        
        for table_metadata in scraped_data["tables"]:
            try:
                # Generate table summary synchronously for now
                try:
                    columns_list = [str(col) for col in table_metadata['columns'][:5]]  # Convert to strings
                    columns_str = ', '.join(columns_list)
                except Exception as e:
                    columns_str = "Unknown columns"
                
                summary = f"A data table with {table_metadata['num_rows']} rows and {table_metadata['num_columns']} columns. Columns include: {columns_str}. This table contains data about {table_metadata.get('source_url', 'unknown source')}."
                
                # Load the actual table data for metadata (just for basic info, not storage)
                df = pd.read_csv(table_metadata["table_path"])
                
                # Create table chunk for embedding
                table_chunk = {
                    "content": summary,  # This gets embedded
                    "metadata": {
                        "content_type": "table",
                        "summary": summary,
                        "table_id": table_metadata["table_id"],
                        "table_path": table_metadata["table_path"],
                        "source_url": table_metadata.get("source_url", ""),
                        "num_rows": table_metadata["num_rows"],
                        "num_columns": table_metadata["num_columns"],
                        "columns": [str(col) for col in table_metadata["columns"]],  # Convert to strings
                        "keywords": ", ".join([str(col) for col in table_metadata["columns"][:10]])  # Convert to strings
                    }
                }
                
                all_chunks.append(table_chunk)
                print(f"   ✅ Table {table_metadata['table_id']}: {table_metadata['num_rows']}x{table_metadata['num_columns']}")
                
            except Exception as e:
                print(f"   ❌ Error processing table {table_metadata.get('table_id', 'unknown')}: {str(e)}")
                continue
    
    print(f"📦 Total chunks created: {len(all_chunks)} ({len([c for c in all_chunks if c.get('metadata', {}).get('content_type') != 'table'])} text, {len([c for c in all_chunks if c.get('metadata', {}).get('content_type') == 'table'])} table)")
    
    return all_chunks

def create_vector_embeddings(scraped_data: Dict[str, Any], user_prompt: str):
    """
    Create vector embeddings for scraped content and upload to Pinecone.
    """
    try:
        # Process the scraped content to create chunks
        records = process_scraped_content(scraped_data, user_prompt)
        
        if not records:
            print("No records to upload to Pinecone")
            return
        
        # Initialize Pinecone
        INDEX_NAME = "data-analyst-agent-embedder"
        create_pinecone_index()  # Ensure index exists
        
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        
        # Clean up old data before uploading new data
        try:
            # Delete the namespace to remove old data
            index.delete(namespace="scraped-content")
            print("🧹 Cleaned up old Pinecone data")
        except Exception as e:
            print(f"⚠️ Could not clean up old data: {str(e)}")
            
        # Upload records to Pinecone
        upload_to_pinecone(records, index)
        
    except Exception as e:
        print(f"Error creating vector embeddings: {str(e)}")
        raise e

def upload_to_pinecone(records: List[Dict[str, Any]], index):
    """
    Upload records to Pinecone with proper formatting for both text and table records.
    """
    try:
        formatted_records = []
        
        for record in records:
            # Handle both old and new record formats
            if "content" in record:
                # New format from process_scraped_content
                content = record["content"]
                metadata = record.get("metadata", {})
                
                # Create unique ID
                if metadata.get("content_type") == "table":
                    record_id = f"table_{metadata.get('table_id', 'unknown')}"
                else:
                    # Text content
                    import hashlib
                    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
                    record_id = f"text_{content_hash}"
                
                # Flatten metadata for Pinecone (no nested objects)
                flat_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        flat_metadata[key] = value
                    elif isinstance(value, list):
                        # Convert lists to comma-separated strings
                        flat_metadata[key] = ", ".join([str(v) for v in value])
                    else:
                        # Convert other types to strings
                        flat_metadata[key] = str(value)
                
                formatted_record = {
                    "id": record_id,
                    "text": content,  # This gets embedded by Pinecone
                    **flat_metadata  # Spread flattened metadata at top level
                }
            else:
                # Old format - direct from process_scraped_content (legacy)
                formatted_record = record
            
            formatted_records.append(formatted_record)
        
        # Upload in batches
        batch_size = 96  # Reduced from 100 to fit Pinecone limits
        for i in range(0, len(formatted_records), batch_size):
            batch = formatted_records[i:i + batch_size]
            
            try:
                result = index.upsert_records(
                    records=batch,
                    namespace="scraped-content"
                )
                print(f"   Uploaded batch {i//batch_size + 1}: {len(batch)} records")
            except Exception as batch_error:
                print(f"   ❌ Error uploading batch {i//batch_size + 1}: {str(batch_error)}")
                continue
        
        print(f"Upload summary: {len(formatted_records)} total records processed")
            
    except Exception as e:
        print(f"Error uploading to Pinecone: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        "scraped_pages": 2,
        "total_words": 5000,
        "url": "https://example.com",
        "prompt": "What is artificial intelligence?",
        "content_summary": "--- Page: Example Page (https://example.com) ---\nThis is sample content about AI and machine learning...",
        "pages": [
            {
                "title": "Example Page",
                "url": "https://example.com",
                "word_count": 2500
            }
        ]
    }
    
    create_vector_embeddings(sample_data, "What is artificial intelligence?") 