"""
Enhanced PDF and Multi-format Document Parser
Functionality: Uses pdfplumber for better table extraction, supports multiple formats, maintains existing API.
"""
import os
import re
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
from vector_embeddings import create_vector_embeddings
from search_and_analyze import search_and_analyze
from pinecone import Pinecone
import asyncio
from dotenv import load_dotenv
import json
import shutil
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime

# Document processing imports
import pdfplumber
import docx
import email
from email import policy
import mammoth

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata"""
    content: str
    page_number: int
    chunk_id: str
    document_name: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

class DocumentProcessor:
    """Handles extraction and processing of different document types"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.doc', '.eml', '.msg']
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a document and extract content based on file type"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return self.extract_from_docx(file_path)
        elif file_ext in ['.eml', '.msg']:
            return self.extract_from_email(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return []
    
    def extract_from_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Extract text and tables from PDF using pdfplumber with better table detection"""
        chunks = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                doc_name = os.path.basename(file_path)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text with layout preservation
                    text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        layout=True,
                        x_density=6,
                        y_density=13
                    )
                    
                    if text:
                        # Extract tables if present using pdfplumber's built-in table detection
                        tables = page.extract_tables()
                        table_content = ""
                        
                        if tables:
                            for i, table in enumerate(tables):
                                if table and len(table) > 0:
                                    # Clean the table data
                                    cleaned_table = []
                                    for row in table:
                                        if row and any(cell and str(cell).strip() for cell in row):
                                            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                                            cleaned_table.append(cleaned_row)
                                    
                                    if len(cleaned_table) >= 2 and len(cleaned_table[0]) >= 2:
                                        table_content += f"\n\nTable {i+1}:\n"
                                        for row in cleaned_table:
                                            table_content += "  ".join(row) + "\n"
                        
                        full_content = text + table_content
                        
                        chunk = DocumentChunk(
                            content=full_content,
                            page_number=page_num,
                            chunk_id=f"{doc_name}_page_{page_num}",
                            document_name=doc_name,
                            metadata={'extraction_method': 'pdfplumber', 'char_count': len(full_content)}
                        )
                        chunks.append(chunk)
                        
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
        
        return chunks
    
    def extract_from_docx(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from DOCX files"""
        chunks = []
        
        try:
            # Using mammoth for better formatting preservation
            with open(file_path, "rb") as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                text = result.value
                
                if text:
                    doc_name = os.path.basename(file_path)
                    chunk = DocumentChunk(
                        content=text,
                        page_number=1,
                        chunk_id=f"{doc_name}_full",
                        document_name=doc_name,
                        metadata={'extraction_method': 'mammoth', 'char_count': len(text)}
                    )
                    chunks.append(chunk)
                    
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {str(e)}")
            
        return chunks
    
    def extract_from_email(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from email files"""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                msg = email.message_from_file(f, policy=policy.default)
                
            # Extract email metadata
            subject = msg.get('Subject', '')
            sender = msg.get('From', '')
            date = msg.get('Date', '')
            
            # Extract email body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_content()
            else:
                body = msg.get_content()
            
            if body or subject:
                full_content = f"Subject: {subject}\nFrom: {sender}\nDate: {date}\n\n{body}"
                
                doc_name = os.path.basename(file_path)
                chunk = DocumentChunk(
                    content=full_content,
                    page_number=1,
                    chunk_id=f"{doc_name}_email",
                    document_name=doc_name,
                    metadata={
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'type': 'email'
                    }
                )
                chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Error processing email {file_path}: {str(e)}")
            
        return chunks

def extract_tables_with_pdfplumber(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract tables from PDF using pdfplumber with better detection than Tabula.
    Returns tables in the same format as the existing system for compatibility.
    """
    tables = []
    
    try:
        print(f"🔍 Extracting tables from PDF using pdfplumber...")
        
        with pdfplumber.open(pdf_path) as pdf:
            table_index = 0
            
            for page_num, page in enumerate(pdf.pages):
                # Extract tables from this page
                page_tables = page.extract_tables()
                
                for table in page_tables:
                    if table and len(table) > 0:
                        # Clean the table data
                        cleaned_table = []
                        for row in table:
                            if row and any(cell and str(cell).strip() for cell in row):
                                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                                cleaned_table.append(cleaned_row)
                        
                        # Check if table is meaningful (similar to web scraper logic)
                        if len(cleaned_table) >= 2 and len(cleaned_table[0]) >= 2:
                            # Create DataFrame
                            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
                            df = df.dropna(how='all').dropna(axis=1, how='all')
                            
                            if len(df) >= 2 and len(df.columns) >= 2:
                                # Generate deterministic table ID
                                table_id = f"pdf_table_{hash(pdf_path)}_{table_index}.csv"
                                table_path = f"temp_files/{table_id}"
                                
                                # Save as CSV
                                df.to_csv(table_path, index=False)
                                
                                # Find context (simplified version)
                                caption = f"Table {table_index + 1} from PDF page {page_num + 1}"
                                associated_header = f"PDF Table {table_index + 1}"
                                
                                # Create metadata matching existing structure exactly
                                table_metadata = {
                                    "table_id": table_id,
                                    "table_path": table_path,
                                    "num_rows": len(df),
                                    "num_columns": len(df.columns),
                                    "columns": [str(col) for col in df.columns],
                                    "source_url": f"local_file:{pdf_path}",
                                    "caption": caption,
                                    "associated_header": associated_header,
                                    "summary": f"A data table with {len(df)} rows and {len(df.columns)} columns. Columns include: {', '.join([str(col) for col in df.columns])}"
                                }
                                
                                # Add context to summary
                                context_parts = []
                                if associated_header:
                                    context_parts.append(f"Header: {associated_header}")
                                if caption:
                                    context_parts.append(f"Caption: {caption}")
                                
                                if context_parts:
                                    table_metadata['summary'] = f"{' | '.join(context_parts)}. {table_metadata['summary']}"
                                
                                tables.append(table_metadata)
                                
                                # Print info
                                header_preview = associated_header[:30] if associated_header else "No header"
                                print(f"   ✅ Table {table_id}: {len(df)}x{len(df.columns)} - {header_preview}")
                                
                                table_index += 1
        
        print(f"📊 Successfully extracted {len(tables)} tables from PDF using pdfplumber")
        return tables
        
    except Exception as e:
        print(f"❌ pdfplumber extraction error: {e}")
        return []

def parse_document_enhanced_pdfplumber(pdf_path: str, save_parsed_text: bool = False) -> dict:
    """
    Parse PDF using pdfplumber with better text and table extraction.
    Maintains compatibility with existing API.
    """
    try:
        doc_name = os.path.basename(pdf_path)
        
        print(f"🚀 Starting pdfplumber parsing for {doc_name}...")
        
        # Use DocumentProcessor for extraction
        processor = DocumentProcessor()
        chunks = processor.extract_from_pdf(pdf_path)
        
        if not chunks:
            return {"error": "No content could be extracted from the PDF"}
        
        # Combine all chunks into single content (maintaining existing API)
        all_content = ""
        ordered_content = []
        
        for chunk in chunks:
            all_content += chunk.content + "\n\n"
            ordered_content.append({
                'content': chunk.content,
                'type': 'text',
                'page': chunk.page_number,
                'source': 'pdfplumber'
            })
        
        # Clean the final text
        final_text = clean_text(all_content)
        
        result = {
            'document_name': doc_name,
            'content': final_text,
            'ordered_content': ordered_content,
            'total_pages': len(chunks),
            'parsing_method': 'pdfplumber',
            'processing_time': 0,  # Will be calculated if needed
            'metadata': {
                'total_elements': len(ordered_content),
                'text_elements': len(ordered_content),
                'table_elements': 0,  # Tables are now integrated into text
                'pages_processed': len(chunks),
                'characters_extracted': len(final_text)
            }
        }
        
        print(f"✅ pdfplumber parsing complete: {len(final_text)} characters")
        return result
        
    except Exception as e:
        print(f"❌ pdfplumber parsing error: {e}")
        return {"error": str(e)}

def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""
    
    # Remove excessive whitespace while preserving line breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple line breaks to double
    
    return text.strip()

def cleanup_temp_files():
    """
    Clean up all temporary files before starting a new PDF analysis task.
    """
    try:
        if os.path.exists("temp_files"):
            # Remove only CSV and table files, keep uploaded PDFs
            for filename in os.listdir("temp_files"):
                if filename.endswith('.csv') or filename.startswith('table_') or filename.startswith('pdf_table_'):
                    file_path = os.path.join("temp_files", filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"⚠️ Error deleting {file_path}: {e}")
            print(f"🧹 Cleaned up old table files from temp_files")
        else:
            # Create the directory if it doesn't exist
            os.makedirs("temp_files", exist_ok=True)
            print(f"📁 Created temp_files directory")
    except Exception as e:
        print(f"❌ Error during cleanup: {str(e)}")

async def run_pdf_analysis_task(pdf_path: str, user_prompt: str) -> dict:
    """
    Complete PDF analysis pipeline: parse PDF, extract tables, create embeddings, search and analyze.
    Now uses pdfplumber for better table extraction.
    """
    try:
        # Clean up temp files before starting (except the PDF being analyzed)
        cleanup_temp_files()
        
        print(f"📄 Starting PDF analysis for: {pdf_path}")
        print(f"🔍 User prompt: {user_prompt}")

        # 1. Parse the PDF to get the text content using pdfplumber
        parsed_data = parse_document_enhanced_pdfplumber(pdf_path)
        content = parsed_data.get("content")

        if not content:
            return {"error": "Failed to extract text from the PDF."}

        # 2. Extract tables using pdfplumber (better than Tabula)
        extracted_tables = extract_tables_with_pdfplumber(pdf_path)
        print(f"📊 Extracted {len(extracted_tables)} tables from PDF")

        # 3. Structure the data for the analysis pipeline (same as before)
        analysis_result = {
            "scraped_pages": 1,
            "total_words": len(content.split()),
            "url": f"local_file:{pdf_path}",
            "prompt": user_prompt,
            "content_summary": content[:3000],
            "scraped_content": [{"title": parsed_data.get('document_name', 'PDF Content'), "content": content}],
            "pages": [{
                "title": parsed_data.get('document_name', 'PDF Content'),
                "url": f"local_file:{pdf_path}",
                "word_count": len(content.split())
            }],
            "tables": extracted_tables  # Add extracted tables to analysis result
        }

        # 4. Run the vector embedding and search/analysis pipeline (unchanged)
        create_vector_embeddings(analysis_result, user_prompt)
        await asyncio.sleep(5)  # Allow time for indexing

        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        INDEX_NAME = "data-analyst-agent-embedder"
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)

        search_analysis_result = await search_and_analyze(
            user_prompt, index, scraped_content=analysis_result["scraped_content"]
        )

        analysis_result["search_analysis"] = search_analysis_result
        return analysis_result

    except Exception as e:
        print(f"An error occurred during the PDF analysis task: {e}")
        return {"error": str(e)}

# Legacy function names for backward compatibility
def parse_document_enhanced_pymupdf(pdf_path: str, save_parsed_text: bool = False) -> dict:
    """Legacy function name - now uses pdfplumber"""
    return parse_document_enhanced_pdfplumber(pdf_path, save_parsed_text)

def parse_document_paddleocr(pdf_path: str, save_parsed_text: bool = False, use_gpu: bool = False) -> dict:
    """Legacy function name - now uses pdfplumber"""
    return parse_document_enhanced_pdfplumber(pdf_path, save_parsed_text)

def extract_tables_with_tabula(pdf_path: str) -> List[Dict[str, Any]]:
    """Legacy function name - now uses pdfplumber"""
    return extract_tables_with_pdfplumber(pdf_path)

# Additional functions for multi-format support
def process_multi_format_document(file_path: str) -> List[DocumentChunk]:
    """Process any supported document format"""
    processor = DocumentProcessor()
    return processor.process_document(file_path)

def get_supported_formats() -> List[str]:
    """Get list of supported file formats"""
    processor = DocumentProcessor()
    return processor.supported_formats

if __name__ == '__main__':
    # This is an example of how to run this script directly for testing
    test_pdf_path = "BAJHLIP23020V012223.pdf" 
    if os.path.exists(test_pdf_path):
        test_prompt = "What does the policy cover under in-patient hospitalization in India?"
        analysis_output = asyncio.run(run_pdf_analysis_task(test_pdf_path, test_prompt))
        print(json.dumps(analysis_output, indent=2))
    else:
        print(f"Test file not found: {test_pdf_path}") 