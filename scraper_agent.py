import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright
from dotenv import load_dotenv
import time
import re
from vector_embeddings import create_vector_embeddings
from search_and_analyze import search_and_analyze
from pinecone import Pinecone
import asyncio
import uuid
from typing import Dict, Any, List, Tuple
import pandas as pd
from io import StringIO
import shutil
import hashlib

load_dotenv()

def scrape_with_beautifulsoup(url):
    """
    Scrapes a website using requests and BeautifulSoup.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"Error making request to {url}: {e}")
        return None

def find_associated_header(table_element, soup):
    """
    Find the header element that comes immediately before the table.
    Returns the header text or empty string if not found.
    """
    try:
        # Look for the most recent heading (h1, h2, h3, h4, h5, h6) before this table
        current = table_element.previous_sibling
        
        # Go backwards through siblings to find the most recent heading
        while current:
            if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                header_text = current.get_text().strip()
                if header_text:
                    return header_text
            current = current.previous_sibling
        
        # If no heading found in siblings, look in parent's previous siblings
        parent = table_element.parent
        if parent:
            current = parent.previous_sibling
            while current:
                if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    header_text = current.get_text().strip()
                    if header_text:
                        return header_text
                current = current.previous_sibling
        
        return ""
    except Exception as e:
        print(f"⚠️ Error finding associated header: {str(e)}")
        return ""

def extract_tables_from_html(html_content: str, url: str) -> tuple[list[str], list[dict]]:
    """
    Extract tables from HTML content and save them as CSV files.
    Returns table_ids and table_metadata.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table', class_='wikitable')
    
    table_ids = []
    table_metadata = []
    
    for i, table in enumerate(tables):
        try:
            # Create deterministic table ID based on URL and table index
            table_id_base = f"{url}_{i}"
            table_id_hash = hashlib.md5(table_id_base.encode()).hexdigest()[:8]
            table_id = f"table_{table_id_hash}_{i}.csv"
            
            # Extract table caption for context
            caption = table.find('caption')
            caption_text = caption.get_text().strip() if caption else ""
            
            # Find associated header
            associated_header = find_associated_header(table, soup)
            
            # Convert table to DataFrame
            df = table_to_dataframe(table)
            
            # Check if table is relevant
            if not is_relevant_table(df):
                print(f"⏭️ Skipping irrelevant table {i+1}")
                continue
            
            # Save table as CSV
            table_path = f"temp_files/{table_id}"
            df.to_csv(table_path, index=False)
            
            # Create metadata with caption and associated header
            metadata = {
                'table_id': table_id,
                'table_path': table_path,
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'columns': [str(col) for col in df.columns],
                'caption': caption_text,
                'associated_header': associated_header,
                'summary': f"A data table with {len(df)} rows and {len(df.columns)} columns. Columns include: {', '.join([str(col) for col in df.columns])}"
            }
            
            # Add caption and header to summary if available
            context_parts = []
            if associated_header:
                context_parts.append(f"Header: {associated_header}")
            if caption_text:
                context_parts.append(f"Caption: {caption_text}")
            
            if context_parts:
                metadata['summary'] = f"{' | '.join(context_parts)}. {metadata['summary']}"
            
            table_ids.append(table_id)
            table_metadata.append(metadata)
            
            header_preview = associated_header[:30] if associated_header else "No header"
            print(f"   ✅ Table {table_id}: {len(df)}x{len(df.columns)} - {header_preview}")
            
        except Exception as e:
            print(f"   ❌ Error processing table {i+1}: {str(e)}")
            continue
    
    return table_ids, table_metadata

def table_to_dataframe(table):
    """
    Convert a BeautifulSoup table to a pandas DataFrame with proper handling.
    """
    try:
        # Get table caption for context
        caption = table.find('caption')
        caption_text = caption.get_text().strip() if caption else ""
        
        # Find all rows
        rows = table.find_all('tr')
        if not rows:
            return None
            
        # Extract headers from first row
        header_row = rows[0]
        headers = []
        for cell in header_row.find_all(['th', 'td']):
            # Clean the header text
            header_text = cell.get_text().strip()
            # Remove any citation markers like [12], [update], etc.
            header_text = re.sub(r'\[\d+\]|\[update\]|\[Inf\]', '', header_text).strip()
            headers.append(header_text)
        
        # If no headers found, create generic ones
        if not headers or all(not h for h in headers):
            max_cols = max(len(row.find_all(['td', 'th'])) for row in rows)
            headers = [f"Column_{i+1}" for i in range(max_cols)]
        
        # Extract data rows
        data_rows = []
        for row in rows[1:]:  # Skip header row
            cells = row.find_all(['td', 'th'])
            if cells:
                row_data = []
                for cell in cells:
                    # Clean cell text
                    cell_text = cell.get_text().strip()
                    # Remove citation markers
                    cell_text = re.sub(r'\[\d+\]|\[update\]|\[Inf\]', '', cell_text).strip()
                    row_data.append(cell_text)
                
                data_rows.append(row_data)
        
        # Create DataFrame
        if data_rows:
            df = pd.DataFrame(data_rows, columns=headers)
            return df
        else:
            return None
            
    except Exception as e:
        return None

def is_relevant_table(df: pd.DataFrame) -> bool:
    """
    Determine if a table is relevant for analysis.
    Filters out navigation tables, small reference tables, and empty tables.
    """
    try:
        # Skip None or empty DataFrames
        if df is None or len(df) == 0:
            return False
            
        # Skip very small tables (likely navigation or reference)
        if len(df) < 5:
            return False
            
        # Skip tables with mostly numeric indices in first column
        if len(df.columns) > 0:
            first_col = df.iloc[:, 0]
            if first_col.dtype in ['int64', 'float64']:
                # Check if first column is just sequential numbers (0,1,2,3...)
                if len(first_col.unique()) == len(first_col) and first_col.min() == 0:
                    return False
                    
        # Skip tables with too few columns (likely navigation)
        if len(df.columns) < 3:
            return False
            
        # Skip tables that are mostly empty
        non_null_ratio = df.notna().sum().sum() / (len(df) * len(df.columns))
        if non_null_ratio < 0.3:  # Less than 30% of cells have data
            return False
            
        # Look for data-rich indicators
        has_text_content = False
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column has meaningful text content
                sample_values = df[col].dropna().astype(str).str.len()
                if sample_values.mean() > 5:  # Average text length > 5 characters
                    has_text_content = True
                    break
                    
        if not has_text_content:
            return False
            
        return True
        
    except Exception as e:
        print(f"⚠️ Error checking table relevance: {str(e)}")
        return False

def remove_tables_from_html(html_content: str) -> str:
    """
    Remove table elements from HTML to prevent them from interfering with text extraction.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove all table elements
        for table in soup.find_all('table'):
            table.decompose()
            
        return str(soup)
    except Exception as e:
        print(f"⚠️ Error removing tables from HTML: {str(e)}")
        return html_content

async def scrape_with_playwright(url: str) -> Dict[str, Any]:
    """
    Enhanced scraper that extracts both text content and tables separately.
    """
    try:
        print(f"🔍 Starting Playwright scraping for: {url}")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            print(f"🌐 Navigating to: {url}")
            await page.goto(url, wait_until='networkidle', timeout=30000)
            await asyncio.sleep(3)  # Increased wait time for JS-heavy sites
            
            # Get the full HTML content
            full_html = await page.content()
            print(f"📄 HTML content length: {len(full_html)} characters")
            
            # Extract tables first
            table_ids, table_metadata = extract_tables_from_html(full_html, url)
            print(f"📊 Found {len(table_ids)} tables")
            
            # Remove tables from HTML for clean text extraction
            clean_html = remove_tables_from_html(full_html)
            
            # Create a new page with clean HTML for text extraction
            clean_page = await browser.new_page()
            await clean_page.set_content(clean_html)
            
            # Extract clean text content with better debugging
            content = await clean_page.evaluate('''() => {
                // Remove script and style elements
                const scripts = document.querySelectorAll('script, style, nav, footer, header, aside');
                scripts.forEach(el => el.remove());
                
                // Get main content areas
                const contentSelectors = [
                    'main', '[role="main"]', '.content', '#content', 
                    '.main-content', '.article-content', '.post-content',
                    'article', '.entry-content'
                ];
                
                let mainContent = '';
                for (const selector of contentSelectors) {
                    const element = document.querySelector(selector);
                    if (element) {
                        mainContent = element.innerText || element.textContent || '';
                        console.log('Found content with selector:', selector, 'Length:', mainContent.length);
                        break;
                    }
                }
                
                // Fallback to body if no main content found
                if (!mainContent.trim()) {
                    mainContent = document.body.innerText || document.body.textContent || '';
                    console.log('Using body content, length:', mainContent.length);
                }
                
                return mainContent.trim();
            }''')
            
            title = await clean_page.evaluate('() => document.title')
            print(f"📝 Title: {title}")
            print(f"📄 Content length: {len(content)} characters")
            print(f"📊 Word count: {len(content.split())} words")
            
            # Debug: Show first 500 characters of content
            if content:
                print(f"🔍 Content preview: {content[:500]}...")
            else:
                print("⚠️ No content extracted!")
            
            await clean_page.close()
            await browser.close()
            
            return {
                "url": url,
                "title": title or "No title",
                "content": content,
                "word_count": len(content.split()) if content else 0,
                "tables": table_metadata,
                "table_ids": table_ids,
                "extraction_method": "playwright_with_tables"
            }
            
    except Exception as e:
        print(f"❌ Playwright scraping failed for {url}: {str(e)}")
        return {"url": url, "title": "Error", "content": "", "word_count": 0, "tables": [], "table_ids": [], "error": str(e)}

def is_content_dynamic(soup):
    """
    Checks if the content of a page is likely dynamically loaded.
    This is a heuristic and might need to be adjusted for specific sites.
    """
    if soup is None:
        return False
        
    # Case 1: Low number of tags in the body
    if len(soup.body.find_all(recursive=False)) < 5:
        return True

    # Case 2: Check for common JavaScript framework placeholders
    if soup.find(id='root') or soup.find(id='app') or soup.find("div", {"data-reactroot" : re.compile(r'.*')}):
        return True

    # Case 3: Check for script tags that might indicate a large JS application
    script_tags = soup.find_all('script')
    for script in script_tags:
        if script.get('src') and ('bundle.js' in script.get('src') or 'app.js' in script.get('src')):
            return True
            
    return False

def is_valid_url_for_crawling(url, base_netloc):
    """
    Checks if a URL is valid for crawling (not a fragment, same domain, etc.)
    """
    parsed = urlparse(url)
    
    # Skip URL fragments (anchor links)
    if parsed.fragment:
        return False
    
    # Skip non-HTTP/HTTPS URLs
    if parsed.scheme not in ['http', 'https']:
        return False
    
    # Skip external domains
    if parsed.netloc != base_netloc:
        return False
    
    # Skip common non-content URLs and Wikipedia navigation pages
    skip_patterns = [
        '/login', '/signup', '/register', '/logout',
        '/admin', '/dashboard', '/profile',
        '/api/', '/ajax/', '/json/',
        '/search', '/sitemap', '/robots.txt',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx',
        '.jpg', '.jpeg', '.png', '.gif', '.svg',
        '.css', '.js', '.xml', '.rss'
    ]
    
    for pattern in skip_patterns:
        if pattern in url.lower():
            return False
    
    return True

async def crawl_site(start_url, max_pages=5):
    """
    Crawls a website starting from a given URL using the old-fashioned approach.
    """
    # Only scrape the main page - no crawling to other pages
    scraped_content = []
    
    try:
        # Always use Playwright to handle modern websites
        soup = await scrape_with_playwright(start_url)
        
        if soup is None:
            return scraped_content
        
        # Extract title
        title = soup.title.string if soup.title else "No title found"
        
        # Generalized content extraction - find the main content area
        main_content = None
        
        # Try common content selectors
        content_selectors = [
            'main', 'article', '[role="main"]', '.main-content', '.content', '.post-content',
            '.entry-content', '.article-content', '.story-content', '.page-content',
            '#content', '#main', '#article', '#post-content', '#entry-content'
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no main content found, try to identify the largest text block
        if not main_content:
            # Find all divs with substantial text content
            text_blocks = []
            for div in soup.find_all('div'):
                text = div.get_text().strip()
                if len(text) > 200:  # Only consider blocks with substantial content
                    text_blocks.append((div, len(text)))
            
            # Sort by text length and take the largest
            if text_blocks:
                text_blocks.sort(key=lambda x: x[1], reverse=True)
                main_content = text_blocks[0][0]
        
        # Extract text from main content or fallback to body
        if main_content:
            # Remove navigation and non-content elements from main content
            for element in main_content.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'menu']):
                element.decompose()
            
            # Remove UI elements and tables (since tables don't scrape properly)
            for element in main_content.find_all(['button', 'input', 'select', 'option', 'form', 'label', 'table', 'thead', 'tbody', 'tr', 'th', 'td']):
                element.decompose()
            
            # Remove social media and sharing elements
            for element in main_content.find_all(['share', 'social', 'twitter', 'facebook', 'linkedin']):
                element.decompose()
            
            # Remove advertisement elements
            for element in main_content.find_all(['ad', 'advertisement', 'sponsor', 'promo']):
                element.decompose()
            
            # Remove Wikipedia-specific non-content elements
            for element in main_content.find_all(['sup', 'sub', 'cite', 'ref', 'references', 'reflist', 'notelist']):
                element.decompose()
            
            # Remove elements with specific classes that are typically references
            for element in main_content.find_all(class_=re.compile(r'ref|reference|citation|note|footnote|bibliography')):
                element.decompose()
            
            # Remove elements with specific IDs that are typically references
            for element in main_content.find_all(id=re.compile(r'ref|reference|citation|note|footnote|bibliography')):
                element.decompose()
            
            # Get text content with better formatting, preserving structure
            text_content = main_content.get_text(separator='\n', strip=True)
            
            # Clean up excessive whitespace but preserve paragraph breaks
            text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
            text_content = re.sub(r' +', ' ', text_content)
            text_content = text_content.strip()
        else:
            # Fallback to general extraction
            # Extract text content (remove scripts, styles, etc.)
            for script in soup(["script", "style", 'nav', 'footer', 'header', 'aside']):
                script.decompose()
            
            # Remove common UI elements and tables
            for element in soup.find_all(['button', 'input', 'select', 'option', 'form', 'label', 'table', 'thead', 'tbody', 'tr', 'th', 'td']):
                element.decompose()
            
            # Remove navigation elements
            for element in soup.find_all(['nav', 'menu', 'breadcrumb']):
                element.decompose()
            
            # Remove social media and sharing elements
            for element in soup.find_all(['share', 'social', 'twitter', 'facebook', 'linkedin']):
                element.decompose()
            
            # Remove advertisement elements
            for element in soup.find_all(['ad', 'advertisement', 'sponsor', 'promo']):
                element.decompose()
            
            # Remove Wikipedia-specific non-content elements
            for element in soup.find_all(['sup', 'sub', 'cite', 'ref', 'references', 'reflist', 'notelist']):
                element.decompose()
            
            # Remove elements with specific classes that are typically references
            for element in soup.find_all(class_=re.compile(r'ref|reference|citation|note|footnote|bibliography')):
                element.decompose()
            
            # Remove elements with specific IDs that are typically references
            for element in soup.find_all(id=re.compile(r'ref|reference|citation|note|footnote|bibliography')):
                element.decompose()
            
            # Remove common UI text patterns
            for element in soup.find_all(string=True):
                if element.parent:
                    text = element.strip()
                    # Remove common UI text patterns
                    ui_patterns = [
                        'skip to', 'skip navigation', 'menu', 'search', 'subscribe', 'log in', 'sign up',
                        'advertisement', 'ad', 'sponsored', 'promoted', 'share', 'print', 'email',
                        'facebook', 'twitter', 'linkedin', 'instagram', 'youtube',
                        'cookie', 'privacy', 'terms', 'contact', 'about', 'help',
                        'breaking news', 'live', 'updated', 'min read', 'min ago',
                        'subscribe for', 'log in', 'today\'s paper', 'crosswords', 'games',
                        'sections', 'top stories', 'most popular', 'trending'
                    ]
                    
                    if any(pattern in text.lower() for pattern in ui_patterns):
                        element.extract()
            
            # Get text content with better formatting, preserving structure
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Clean up excessive whitespace but preserve paragraph breaks
            text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
            text_content = re.sub(r' +', ' ', text_content)
            text_content = text_content.strip()
        
        # Clean up the text - preserve line structure
        lines = []
        for line in text_content.splitlines():
            line = line.strip()
            if line:
                # Clean up excessive whitespace within the line
                line = re.sub(r' +', ' ', line)
                lines.append(line)
        # Join lines with proper line breaks to preserve structure
        text_content = '\n'.join(lines)
        
        # Additional cleaning to remove remaining UI artifacts
        # Remove timestamps and time-related text
        text_content = re.sub(r'\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.|AM|PM)', '', text_content)
        text_content = re.sub(r'\d+\s*min\s*(?:read|ago)', '', text_content)
        text_content = re.sub(r'\d+\s*min\s*read', '', text_content)
        text_content = re.sub(r'LIVE.*?ET', '', text_content)
        text_content = re.sub(r'Updated.*?ET', '', text_content)
        # Remove common navigation text
        text_content = re.sub(r'Skip to.*?content', '', text_content, flags=re.IGNORECASE)
        text_content = re.sub(r'Search.*?Navigation', '', text_content, flags=re.IGNORECASE)
        text_content = re.sub(r'Subscribe for.*?week', '', text_content, flags=re.IGNORECASE)
        text_content = re.sub(r'Log in', '', text_content, flags=re.IGNORECASE)
        # Remove Wikipedia-specific reference patterns
        text_content = re.sub(r'\[\d+\]', '', text_content)  # Remove [1], [2], etc.
        text_content = re.sub(r'\(\d{4}\)', '', text_content)  # Remove (2023), etc.
        text_content = re.sub(r'doi:\s*[^\s]+', '', text_content)  # Remove DOI references
        text_content = re.sub(r'ISBN\s*[^\s]+', '', text_content)  # Remove ISBN references
        text_content = re.sub(r'PMID\s*\d+', '', text_content)  # Remove PMID references
        text_content = re.sub(r'S2CID\s*\d+', '', text_content)  # Remove S2CID references
        # Remove excessive whitespace but preserve line breaks
        text_content = re.sub(r'[ \t]+', ' ', text_content)  # Replace multiple spaces/tabs with single space
        text_content = text_content.strip()
        
        # Store the scraped content
        page_data = {
            "url": start_url,
            "title": title.strip(),
            "content": text_content,  # Don't limit content length
            "word_count": len(text_content.split())
        }
        
        scraped_content.append(page_data)
        
    except Exception as e:
        print(f"    ❌ Error scraping {start_url}: {str(e)}")
    
    return scraped_content

def cleanup_temp_files():
    """
    Clean up all temporary files before starting a new scraping task.
    """
    try:
        if os.path.exists("temp_files"):
            # Remove all files in temp_files directory
            for filename in os.listdir("temp_files"):
                file_path = os.path.join("temp_files", filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"⚠️ Error deleting {file_path}: {e}")
            print(f"🧹 Cleaned up temp_files directory")
        else:
            # Create the directory if it doesn't exist
            os.makedirs("temp_files", exist_ok=True)
            print(f"📁 Created temp_files directory")
    except Exception as e:
        print(f"❌ Error during cleanup: {str(e)}")

async def run_scraping_task(url: str, user_prompt: str) -> dict:
    """
    Executes a web scraping task for a given URL and prompt with enhanced table extraction.
    """
    try:
        # Clean up temp files before starting
        cleanup_temp_files()
        
        # Use the enhanced Playwright scraper that handles tables
        scraped_data = await scrape_with_playwright(url)
        
        if "error" in scraped_data:
            return {"error": scraped_data["error"]}
        
        if not scraped_data.get("content"):
            return {"error": "No content could be scraped from the website"}
        
        # Format scraped content for the analysis pipeline
        scraped_content = [{
            "title": scraped_data["title"],
            "content": scraped_data["content"],
            "url": scraped_data["url"],
            "word_count": scraped_data["word_count"]
        }]
        
        analysis_result = {
            "scraped_pages": 1,
            "total_words": scraped_data["word_count"],
            "url": url,
            "prompt": user_prompt,
            "content_summary": scraped_data["content"][:3000],
            "scraped_content": scraped_content,
            "tables": scraped_data.get("tables", []),
            "table_ids": scraped_data.get("table_ids", []),
            "pages": [{
                "title": scraped_data["title"],
                "url": scraped_data["url"],
                "word_count": scraped_data["word_count"]
            }]
        }

        try:
            # Create vector embeddings for both text and tables
            create_vector_embeddings(analysis_result, user_prompt)
            
            # Add a longer delay to allow Pinecone to index the new data
            await asyncio.sleep(5)
            
            # Initialize Pinecone
            PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
            INDEX_NAME = "data-analyst-agent-embedder"
            
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(INDEX_NAME)
            
            # Pass the actual scraped content to the search function
            search_analysis_result = await search_and_analyze(user_prompt, index, scraped_content=scraped_content)
            
            # Add the analysis to the result
            analysis_result["search_analysis"] = search_analysis_result
            
            # Print summary of extraction
            print(f"\n📊 Extraction Summary:")
            print(f"   Text content: {scraped_data['word_count']} words")
            print(f"   Tables found: {len(scraped_data.get('tables', []))}")
            if scraped_data.get("tables"):
                for i, table in enumerate(scraped_data["tables"]):
                    print(f"     Table {i+1}: {table['num_rows']} rows, {table['num_columns']} columns")
            
        except Exception as e:
            print(f"⚠️  Vector embeddings or analysis failed: {str(e)}")
    
        # Add a small delay before exiting to allow background tasks to clean up.
        await asyncio.sleep(1)
    
        return analysis_result

    except Exception as e:
        print(f"An error occurred during the scraping task: {e}")
        return {"error": str(e)}

async def run_scraping_only_task(url: str, user_prompt: str) -> dict:
    """
    Executes a web scraping task for a given URL, creates vector embeddings,
    and returns a summary of the tables found without performing analysis.
    """
    try:
        cleanup_temp_files()
        
        scraped_data = await scrape_with_playwright(url)
        
        if "error" in scraped_data:
            return {"error": scraped_data["error"]}
        
        if not scraped_data.get("content"):
            return {"error": "No content could be scraped from the website"}
        
        # Format scraped content for vector embeddings
        analysis_result = {
            "scraped_pages": 1,
            "total_words": scraped_data["word_count"],
            "url": url,
            "prompt": user_prompt,
            "scraped_content": [{
                "title": scraped_data["title"],
                "content": scraped_data["content"],
                "url": scraped_data["url"],
            }],
            "tables": scraped_data.get("tables", []),
        }

        try:
            create_vector_embeddings(analysis_result, user_prompt)
            await asyncio.sleep(5) # Allow time for indexing
            
            table_ids = [table['table_id'] for table in scraped_data.get("tables", [])]
            return {
                "status": "Scraping complete",
                "message": f"Successfully scraped and processed the website. Found {len(table_ids)} tables.",
                "available_tables": table_ids
            }

        except Exception as e:
            return {"error": f"Vector embeddings failed: {str(e)}"}
    
    except Exception as e:
        return {"error": f"An error occurred during the scraping task: {str(e)}"}

if __name__ == '__main__':
    test_url = "https://tds.s-anand.net/#/prompt-engineering"
    test_prompt = "What is a good prompt to use for LLMs in this course"

    # To run the async function from a sync context
    analysis_result = asyncio.run(run_scraping_task(test_url, test_prompt))
    # No longer need to print the result here as the function does it
