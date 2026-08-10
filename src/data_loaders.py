"""Multiple data loading strategies for the Agentic RAG system.

You don't need the original Wikipedia file! Pick any option below.
"""
import os
import json
import gzip
from io import BytesIO
from typing import List
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


# ===========================================================================
# OPTION A: Wikipedia API (No file needed - fetches live)
# ===========================================================================
def load_from_wikipedia_api(
    topics: List[str] = None,
    sentences: int = 10
) -> List[Document]:
    """
    Fetch Wikipedia articles directly via the Wikipedia API.
    No file download needed!

    Args:
        topics: List of article titles to fetch. Default: ["India", "Delhi", "Mumbai"]
        sentences: Number of sentences to fetch per article.

    Returns:
        List of LangChain Document objects.

    Install dependency:
        pip install wikipedia
    """
    try:
        import wikipedia
    except ImportError:
        raise ImportError("Install wikipedia API: pip install wikipedia")

    if topics is None:
        topics = [
            "India", "New Delhi", "Mumbai", "Bangalore", "Chennai",
            "Indian independence movement", "Taj Mahal", "Himalayas",
            "Ganges", "Bollywood", "Indian cuisine", "Yoga",
            "Cricket in India", "Indian Railways", "Infosys"
        ]

    docs = []
    for topic in topics:
        try:
            page = wikipedia.page(topic, auto_suggest=True)
            content = page.content[:5000]  # First ~5000 chars
            docs.append(Document(
                page_content=content,
                metadata={"title": page.title, "article_id": topic}
            ))
            print(f"   Fetched: {page.title}")
        except Exception as e:
            print(f"   Skipped '{topic}': {e}")
            continue

    print(f"   Total fetched: {len(docs)} articles")
    return docs


# ===========================================================================
# OPTION B: Any text files in a folder
# ===========================================================================
def load_from_text_folder(folder_path: str) -> List[Document]:
    """
    Load all .txt files from a folder as documents.
    Perfect if you have your own documents!

    Args:
        folder_path: Path to folder containing .txt files.

    Returns:
        List of LangChain Document objects.
    """
    docs = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append(Document(
                page_content=content,
                metadata={"title": filename.replace(".txt", ""), "article_id": filename}
            ))
            print(f"   Loaded: {filename}")

    print(f"   Total loaded: {len(docs)} files")
    return docs


# ===========================================================================
# OPTION C: CSV file (e.g., news articles, product reviews)
# ===========================================================================
def load_from_csv(
    csv_path: str,
    text_column: str = "text",
    title_column: str = "title"
) -> List[Document]:
    """
    Load documents from a CSV file.
    Great for custom datasets like news, reviews, or support tickets.

    Args:
        csv_path: Path to CSV file.
        text_column: Column name containing the document text.
        title_column: Column name for document title/metadata.

    Returns:
        List of LangChain Document objects.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Install pandas: pip install pandas")

    df = pd.read_csv(csv_path)
    docs = []
    for idx, row in df.iterrows():
        docs.append(Document(
            page_content=str(row[text_column]),
            metadata={
                "title": str(row.get(title_column, f"doc_{idx}")),
                "article_id": idx
            }
        ))

    print(f"   Loaded {len(docs)} rows from {csv_path}")
    return docs


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import PyPDF2
    except ImportError as exc:
        raise ImportError("Install PyPDF2 for PDF uploads: pip install PyPDF2") from exc

    from io import BytesIO
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text.append(page_text)
    return "\n\n".join(text).strip()


def load_from_uploaded_files(uploaded_files) -> List[Document]:
    """Load documents from uploaded Streamlit files."""
    if not uploaded_files:
        return []

    docs: List[Document] = []
    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        extension = name.lower().split(".")[-1]
        file_bytes = uploaded_file.getvalue()

        if not file_bytes:
            print(f"   Skipped empty upload: {name}")
            continue

        try:
            if extension == "txt":
                docs.append(Document(
                    page_content=file_bytes.decode("utf-8", errors="ignore"),
                    metadata={"title": name.removesuffix(".txt"), "article_id": name},
                ))
            elif extension == "pdf":
                text = _extract_text_from_pdf(file_bytes)
                if not text:
                    print(f"   Skipped PDF with no extractable text: {name}")
                    continue
                docs.append(Document(
                    page_content=text,
                    metadata={"title": name.removesuffix(".pdf"), "article_id": name},
                ))
            elif extension == "csv":
                import pandas as pd
                df = pd.read_csv(BytesIO(file_bytes))
                columns = list(df.columns)
                if not columns:
                    print(f"   Skipped CSV with no columns: {name}")
                    continue
                text_column = next((column for column in columns if column.lower() in ("text", "content", "body", "description")), columns[0])
                title_column = next((column for column in columns if column.lower() in ("title", "name", "headline")), None)
                for idx, row in df.iterrows():
                    docs.append(Document(
                        page_content=str(row[text_column]),
                        metadata={"title": str(row[title_column]) if title_column else f"{name}:{idx}", "article_id": f"{name}-{idx}"},
                    ))
            elif extension in ("jsonl", "json"):
                text = file_bytes.decode("utf-8", errors="ignore")
                records = [json.loads(line) for line in text.splitlines() if line.strip()] if extension == "jsonl" else json.loads(text)
                if not isinstance(records, list):
                    records = [records]
                for idx, record in enumerate(records):
                    data = record if isinstance(record, dict) else {"text": str(record)}
                    docs.append(Document(
                        page_content=str(data.get("text", data.get("content", json.dumps(data)))),
                        metadata={"title": str(data.get("title", data.get("name", name))), "article_id": data.get("id", f"{name}-{idx}")},
                    ))
            else:
                print(f"   Skipped unsupported upload: {name}")
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            print(f"   Skipped unreadable upload '{name}': {exc}")

    docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]

    print(f"   Loaded {len(docs)} uploaded documents")
    return docs


# ===========================================================================
# OPTION D: JSON/JSONL file
# ===========================================================================
def load_from_jsonl(filepath: str, text_key: str = "text", title_key: str = "title") -> List[Document]:
    """
    Load documents from a JSONL file.
    Each line should be a JSON object.

    Args:
        filepath: Path to .jsonl or .jsonl.gz file.
        text_key: Key for document text in JSON.
        title_key: Key for document title in JSON.

    Returns:
        List of LangChain Document objects.
    """
    docs = []
    opener = gzip.open if filepath.endswith(".gz") else open
    mode = "rt" if filepath.endswith(".gz") else "r"
    encoding = "utf8" if filepath.endswith(".gz") else "utf-8"

    with opener(filepath, mode, encoding=encoding) as f:
        for line in f:
            data = json.loads(line.strip())
            docs.append(Document(
                page_content=data.get(text_key, ""),
                metadata={
                    "title": data.get(title_key, "Unknown"),
                    "article_id": data.get("id", len(docs))
                }
            ))

    print(f"   Loaded {len(docs)} documents from {filepath}")
    return docs


# ===========================================================================
# OPTION E: Synthetic demo data (works instantly, no dependencies)
# ===========================================================================
def load_synthetic_data() -> List[Document]:
    """
    Generate synthetic Wikipedia-style documents for instant testing.
    No downloads, no API keys, no files needed!

    Returns:
        List of LangChain Document objects.
    """
    articles = [
        {
            "title": "India",
            "content": """India, officially the Republic of India, is a country in South Asia. 
It is the seventh-largest country by area and the most populous country in the world. 
India has a rich history dating back thousands of years. The Indus Valley Civilization, 
one of the world's oldest urban civilizations, flourished here around 2500 BCE. 
India gained independence from British rule on August 15, 1947. 
The country is known for its diverse culture, languages, religions, and traditions. 
Hindi and English are the official languages at the national level. 
India has a parliamentary system of government and is a federal republic. 
The economy is one of the fastest-growing major economies in the world."""
        },
        {
            "title": "New Delhi",
            "content": """New Delhi is the capital of India and a union territory of Delhi. 
It was officially inaugurated as the capital in 1931, replacing Calcutta. 
New Delhi is situated on the west bank of the Yamuna River. 
It serves as the seat of all three branches of the Government of India. 
The city was designed by British architects Edwin Lutyens and Herbert Baker. 
New Delhi has a population of about 9.4 million people. 
It is known for landmarks such as India Gate, Rashtrapati Bhavan, and the Parliament House. 
The city is a major cultural, political, and commercial center of India."""
        },
        {
            "title": "Mumbai",
            "content": """Mumbai, previously known as Bombay, is the capital city of Maharashtra state. 
It is located on the west coast of India and has a natural harbor. 
Mumbai is the financial, commercial, and entertainment capital of India. 
It is home to the Bombay Stock Exchange and the Reserve Bank of India. 
The city is also the center of the Hindi film industry, known as Bollywood. 
Mumbai has a population of over 20 million, making it one of the most populous cities in the world. 
The city was originally a group of seven islands that were merged through land reclamation. 
Mumbai is known for landmarks such as the Gateway of India and Marine Drive."""
        },
        {
            "title": "Taj Mahal",
            "content": """The Taj Mahal is an ivory-white marble mausoleum on the right bank of the Yamuna River in Agra, India. 
It was commissioned in 1632 by Mughal Emperor Shah Jahan to house the tomb of his favorite wife, Mumtaz Mahal. 
The Taj Mahal is considered the finest example of Mughal architecture, combining Indian, Persian, and Islamic styles. 
It was designated as a UNESCO World Heritage Site in 1983. 
The construction involved about 20,000 artisans and took approximately 22 years to complete. 
The Taj Mahal attracts millions of visitors each year and is regarded as a symbol of love."""
        },
        {
            "title": "Indian Railways",
            "content": """Indian Railways is the national railway system of India, operated by the Ministry of Railways. 
It is one of the largest railway networks in the world, with a route length of over 67,000 kilometers. 
Indian Railways carries more than 8 billion passengers annually and employs over 1.2 million people. 
The first train in India ran from Mumbai to Thane on April 16, 1853. 
The railway network connects almost every part of the country, from the Himalayas to the southern tip. 
Indian Railways operates various classes of trains, including the high-speed Vande Bharat Express."""
        },
        {
            "title": "Bollywood",
            "content": """Bollywood is the Hindi-language film industry based in Mumbai, India. 
It is the largest film industry in the world in terms of the number of films produced annually. 
The term Bollywood is a combination of Bombay and Hollywood. 
Bollywood films are known for their musical numbers, dramatic storylines, and colorful costumes. 
The industry has a global audience, with films screened in over 90 countries. 
Some of the most famous Bollywood actors include Shah Rukh Khan, Amitabh Bachchan, and Aishwarya Rai. 
Bollywood has significantly influenced Indian culture and fashion."""
        },
        {
            "title": "Yoga",
            "content": """Yoga is a group of physical, mental, and spiritual practices that originated in ancient India. 
The word yoga comes from the Sanskrit root yuj, meaning to join or unite. 
Yoga is mentioned in the Rigveda and was systematized by Patanjali in the Yoga Sutras around 400 CE. 
There are various schools of yoga, including Hatha Yoga, Ashtanga Yoga, and Kundalini Yoga. 
Yoga has gained worldwide popularity as a form of exercise and stress relief. 
The United Nations declared June 21 as International Day of Yoga in 2014. 
Modern yoga focuses on physical postures called asanas and breathing techniques called pranayama."""
        },
        {
            "title": "Cricket in India",
            "content": """Cricket is the most popular sport in India and is often considered a religion in the country. 
The Board of Control for Cricket in India (BCCI) is the governing body for cricket in India. 
India won the Cricket World Cup in 1983 under Kapil Dev and again in 2011 under MS Dhoni. 
The Indian Premier League (IPL), launched in 2008, is one of the most-watched cricket leagues globally. 
Sachin Tendulkar, known as the God of Cricket, holds numerous batting records. 
The Indian cricket team has produced legendary players like Virat Kohli, Rahul Dravid, and Anil Kumble. 
Cricket matches in India attract massive crowds and television audiences."""
        },
        {
            "title": "Himalayas",
            "content": """The Himalayas are a mountain range in Asia, separating the plains of the Indian subcontinent from the Tibetan Plateau. 
They include the highest peaks in the world, including Mount Everest and K2. 
The Himalayas stretch across five countries: India, Nepal, Bhutan, China, and Pakistan. 
The range is about 2,400 kilometers long and forms an arc. 
The Himalayas are the source of major rivers such as the Ganges, Indus, and Brahmaputra. 
The region is home to diverse flora and fauna, including snow leopards and red pandas. 
The Himalayas have significant cultural and religious importance in Hinduism and Buddhism."""
        },
        {
            "title": "Indian Cuisine",
            "content": """Indian cuisine consists of a variety of regional and traditional cuisines native to India. 
Given the range of diversity in soil type, climate, and occupations, these cuisines vary significantly. 
Indian food is known for its extensive use of spices and herbs. 
Staple foods include rice, wheat, and lentils. 
Popular dishes include biryani, butter chicken, dosa, and samosas. 
Indian cuisine has influenced other cuisines worldwide, especially those in Southeast Asia and the Caribbean. 
Vegetarianism is widely practiced in India due to religious and cultural beliefs."""
        },
    ]

    docs = [
        Document(page_content=article["content"], metadata={"title": article["title"], "article_id": i})
        for i, article in enumerate(articles)
    ]

    print(f"   Generated {len(docs)} synthetic articles")
    print("   These are realistic Wikipedia-style docs for testing")
    return docs


# ===========================================================================
# UNIVERSAL: Chunk any loaded documents
# ===========================================================================
def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping, word-boundary-aware chunks.

    This local implementation avoids importing the heavyweight optional text
    splitter package during Streamlit startup while preserving the existing
    chunk-size and overlap configuration.
    """
    chunks: List[Document] = []
    for document in docs:
        text = document.page_content.strip()
        start = 0
        chunk_number = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            content = text[start:end].strip()
            if content:
                metadata = dict(document.metadata)
                metadata["chunk"] = chunk_number
                chunks.append(Document(page_content=content, metadata=metadata))
                chunk_number += 1
            if end >= len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks
