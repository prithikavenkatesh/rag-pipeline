
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import fitz  # PyMuPDF
import json

# Step 1: Extract paragraphs from PDF
def extract_paragraphs(pdf_path):
    doc = fitz.open(pdf_path)
    paragraphs = []
    for page in doc:
        text = page.get_text()
        paragraphs.extend([p.strip() for p in text.split('\n\n') if p.strip()])
    return paragraphs

# Step 2: Generate embeddings
def generate_embeddings(paragraphs, model):
    return model.encode(paragraphs)

# Step 3: Store in Milvus
def store_in_milvus(paragraphs, embeddings):
    connections.connect(alias="default", host="10.10.70.57", port="19530")
    collection_name = "pdf_chunks"

    # Drop existing collection if needed
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0])),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096)
    ]
    schema = CollectionSchema(fields, description="PDF paragraph embeddings")
    collection = Collection(name=collection_name, schema=schema)

    # Insert data
    data_to_insert = [embeddings.tolist(), paragraphs]
    collection.insert(data_to_insert)
    collection.flush()

    print(f"Collection count: {collection.num_entities}")

    # Create index
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
    )
    print("Index created.")

    # Load collection
    collection.load()
    print("Collection loaded into memory.")

# Step 4: Search using a query
def search_query(query_text, model):
    query_embedding = model.encode(query_text).tolist()

    connections.connect(alias="default", host="10.10.70.57", port="19530")
    collection = Collection("pdf_chunks")
    collection.load()

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=5,
        output_fields=["text"]
    )

    print(f"\nTop 5 matches for query: \"{query_text}\"")
    for result in results[0]:
        print(f"Score: {result.distance:.4f}, Text: {result.entity.get('text')[:100]}...")

# Step 5: Run everything
if __name__ == "__main__":
    MAX_LENGTH = 1000
    pdf_path =  r"C:\Users\prithikavenkatesh\RAGModel\rag-pipeline\assets\sample_pdfs\ST_SGB_2019_7-EN.pdf"

    model = SentenceTransformer('all-MiniLM-L6-v2')

    raw_paragraphs = extract_paragraphs(pdf_path)

    # Truncate long paragraphs
    paragraphs = []
    for i, p in enumerate(raw_paragraphs):
        if len(p) > MAX_LENGTH:
            print(f"Truncating paragraph {i} from {len(p)} to {MAX_LENGTH} characters.")
            p = p[:MAX_LENGTH]
        paragraphs.append(p)

    embeddings = generate_embeddings(paragraphs, model)
    store_in_milvus(paragraphs, embeddings)

    # Test with a real query
    search_query("What measures does the UN Secretariat take to reduce greenhouse gas emissions?", model)
