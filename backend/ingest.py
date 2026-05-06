import os
import re
import numpy as np
import faiss
import pickle
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def clean_text(text):
    # fix broken words with spaces (e.g. "eq uality" → "equality")
    
    # fix multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # fix newlines
    text = re.sub(r'\n+', '\n', text)
    
    return text

def load_pdf(filepath):
    # with(open, %r, "constitutin.pdf"):
    reader = PdfReader("constitution.pdf")
    # pages = reader.pages

    text = ""
    for page in reader.pages:
        # text = text + page.extract_text()
        #read all
        extracted = page.extract_text()

        if extracted:
            text += extracted

    text = clean_text(text)

    return text 


## SIMPLE CHUNKING BASED ON NUMBER OF CHARACTERS
# def chunk_text(text, chunk_size =500, overlap = 50):
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        # step 1: get words from start to start+chunk_size
        # step 2: join those words back into a string
        # step 3: append to chunks list
        # step 4: move start forward by (chunk_size - overlap)

        chunk_words = words[start: start + chunk_size]

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

## RECURSIVE CHUNKING DOESN'T WORK AS EXPECTED

# def chunk_text(text):
    # split at article boundaries like "18." or "19."
    # pattern: number + period + space at start of section
    pattern = r'(?=\d+\.\s+[A-Z])'
    
    chunks = re.split(pattern, text)
    
    # clean and filter empty chunks
    chunks = [c.strip() for c in chunks if len(c.strip()) > 100 and is_meaningful_chunk(c)]
    
    return chunks


def chunk_text(text, chunk_size=500, overlap=50):
    # split at natural boundaries first
    paragraphs = text.split("\n")
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # if adding paragraph keeps us under limit → add it
        if len(current_chunk.split()) + len(paragraph.split()) < chunk_size:
            current_chunk += " " + paragraph
        else:
            # save current chunk if not empty
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # start new chunk with overlap from previous
            if current_chunk.strip():
                previous_words = current_chunk.split()[-overlap:]
                current_chunk = " ".join(previous_words) + " " + paragraph
            else:
                current_chunk = paragraph
    
    # don't forget the last chunk!
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

### CHUNKING BY ARTICLE

def is_meaningful_chunk(chunk, min_length=200):
    """
    Filter out TOC entries and headings-only chunks
    """
    # too short = probably just a title
    if len(chunk.strip()) < min_length:
        return False
    
    # has no sentence structure = probably TOC
    if '.' not in chunk and ':' not in chunk:
        return False
    
    # mostly numbers = probably TOC page numbers
    words = chunk.split()
    if len(words) < 20:
        return False
    
    return True

def chunk_text_with_metadata(text, source="constitution.pdf"):
    pattern = r'(?=\d+\.\s+[A-Z])'
    raw_chunks = re.split(pattern, text)
    
    chunks = []
    metadata = []
    
    for chunk in raw_chunks:
        chunk = chunk.strip()
        
        # skip if too short (TOC entries)
        if len(chunk) < 200:
            continue
            
        # skip if no real content
        if len(chunk.split()) < 20:
            continue
        
        # extract title from first line
        lines = chunk.split('\n')
        title = lines[0].strip()[:100]  # first line = title
        content = '\n'.join(lines[1:]).strip()  # rest = content
        
        # skip if content is empty (heading only!)
        if len(content) < 100:
            continue
        
        chunks.append(chunk)
        metadata.append({
            "title": title,
            "source": source,
            "length": len(chunk),
            "word_count": len(chunk.split())
        })
    
    return chunks, metadata

def get_embeddings(chunks):
    embeddings = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        vector = response.data[0].embedding
        embeddings.append(vector)

        # progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  Progress: {i+1}/{total} chunks embedded...")

    return embeddings

def build_faiss_index(embeddings):
    # step 1: convert embeddings list to numpy array
    embeddings = np.array(embeddings, dtype=np.float32)

    # step 2: get dimension size
    dimension = embeddings.shape[1]

    # step 3: create faiss index
    index = faiss.normalize_L2(embeddings)

    # step 4: add vectors to index
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # step 5: save index to disk
    faiss.write_index(index, "constitution.index")

    return index

if __name__ == "__main__":
    print("Loading PDF...")
    text = load_pdf("constitution.pdf")
    print(f"Total characters extracted:{len(text)}")

    # test chunk_text
    print("Chunking text...")
    # chunks = chunk_text(text)
    chunks, metadata = chunk_text_with_metadata(text)
    print(f"Total chunks created: {len(chunks)}")

    print("Saving chunks and metadata...")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks,f)

    # save metadata separately!
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("Generating embeddings... (this may take 1-2 minutes)")
    embeddings = get_embeddings(chunks)
    print(f"Embeddings generated: {len(embeddings)}")

    print("Building FAISS index...")
    build_faiss_index(embeddings)

    print("Done! Files saved: constitution.index, chunks.pkl")