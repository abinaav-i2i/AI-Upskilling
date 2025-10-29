CREATING A GENERIC QA BASED RAG APPLICATION USING I2I POLICY DATA

**There are 4 PDF documents overall containing the following Policy Data:**
- IT Policy (3 pages): Document contains the policy related data regarding the IT assets of the organization regarding how to fetch the assets from the admin team, cost of damage/broken assets, stolen asset, recovery amount, laptop security best practices and etc.,
- Rewards and Recognition policy (3 pages): Document contains rewards and recognition policy informations details about docker point, long service awards and Quarterly awards and FAQ's on the policy
- Leave Policy (3 pages): Document contains leave policy informations related to Casual leaves, Privileage leave, sick leave, Maternity leave, paternity leave.

- Staff Loan Policy (3 pages): Containing informations on staff loan policy loan dispursement policy, process of availing staff loan, eligibility criteria

**Setting up the environment and dependencies**
- PDF Parsing : PDF Parsing (with metadata like page number and file name) using Document Loader for Text
- Chunking Strategy: Text Chunking using LangChain's Recursive Strategy 
- Embedding: Embedding using Sentence Transformers
- Vector Store: Vector Store using FAISS
- QA Retrieval: QA Retrieval using Claude 3.5 Sonnet (via Anthropic API)
- Meta Data: Source metadata displayed in answers

Step 1: Create a conda environment
Since, we are using langchain, python 3.10 version is best.
- conda create --name rag_env python=3.10
- conda activate rag_env
- pip install -r requirements.txt
If you are running the application for the first time, the below code is used so that pdf is parsed, chunked and then vetor store using faiss is created or If you want to override the existing embedding and generate new embedding for the RAG
- python main1.py --mode build

If you have already parsed and stored the embeddings just run to ask questions regarding the policy of our organization.
- python main1.py --mode query 

Type exit in the console to exit out of the applcation.


