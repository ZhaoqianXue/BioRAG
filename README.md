# BioRAG

## Proposal

### Title: Precision-Focused Retrieval for Medical RAG: Optimized AVL Trees with Semantic Mapping and Chunk-Aware Balancing

### Abstract
Retrieval-Augmented Generation (RAG) systems in domains like medicine demand high retrieval precision to ensure patient safety and diagnostic accuracy. While approximate nearest neighbor (ANN) search methods like FAISS prioritize speed, they inherently risk missing critical information. This proposal introduces an optimized AVL tree variant designed for exact retrieval in medical RAG. Our approach integrates UMAP and Z-Order mappings for semantic embedding indexing, chunk-aware balancing to maintain efficiency with document data, semantic similarity-driven organization for relevance, and efficient pruning to manage search space. Leveraging BioBERT embeddings and implemented in Python, our method demonstrably enhances precision on medical benchmarks compared to standard AVL trees and offers a compelling alternative to FAISS + ANN for precision-critical RAG applications. This research aligns with EMNLP 2025's focus on Retrieval-Augmented Language Models and LLM Efficiency, particularly in high-stakes domains.

### 1. Introduction
Retrieval-Augmented Generation (RAG) systems are crucial for enhancing language models in knowledge-intensive tasks. However, in precision-sensitive fields like medicine, the trade-off between retrieval speed and accuracy becomes paramount. While methods like FAISS with approximate nearest neighbor (ANN) search, sparse retrieval (BM25), and learned sparse retrieval (SPLADE) excel in speed and scalability, their inherent approximation can lead to unacceptable consequences. Missing a single critical document—a rare case study, a crucial drug interaction—can be detrimental in medical contexts.
AVL trees, known for their logarithmic search efficiency and exact retrieval capabilities, present a compelling alternative. When combined with effective dimensionality reduction and mapping techniques like UMAP + Z-Order, AVL trees can efficiently index and retrieve high-dimensional semantic embeddings. However, standard AVL tree implementations are not directly optimized for RAG workloads, potentially suffering from inefficiencies with specific data insertion patterns and lacking semantic awareness to prioritize relevant information. This project addresses these limitations by developing an optimized AVL tree specifically tailored for exact retrieval in medical RAG. By employing BioBERT for generating semantic embeddings, we aim to bridge the gap between AVL trees and the stringent precision requirements of medical RAG, offering a lightweight, interpretable, and highly precise alternative to approximate methods like FAISS + ANN in high-stakes applications.

### 2. Problem Statement
FAISS and ANN-based methods accelerate retrieval by approximating nearest neighbor searches in high-dimensional embedding spaces. Techniques like Inverted File System (IVF) in FAISS partition the search space, significantly boosting speed but inherently introducing approximation errors. In medical RAG, where comprehensive literature retrieval is essential for tasks like diagnosis support, treatment planning, and literature reviews, these approximations pose a significant risk. The potential to miss relevant documents due to ANN's approximation (reduced recall) or retrieve semantically distant but numerically close documents (reduced precision) is unacceptable when patient safety and diagnostic accuracy are at stake.
While standard AVL trees guarantee exact retrieval with O(log n) complexity, they are not inherently optimized for the characteristics of RAG systems and semantic embeddings. Firstly, their balancing mechanisms, based solely on tree height, can become inefficient when handling large, sequentially ingested document sets common in RAG pipelines. Secondly, standard AVL trees lack semantic awareness; they treat keys as mere numerical values without leveraging the underlying semantic relationships between document embeddings to enhance retrieval relevance. Therefore, directly applying standard AVL trees to RAG may not fully realize their potential for precision-focused retrieval. This project directly addresses these inefficiencies, focusing on enhancing AVL trees for scenarios where exactness and interpretability are prioritized over raw speed, particularly in medical RAG applications dealing with moderate-sized, high-value datasets.

### 3. Proposed Solution
Our optimized AVL tree variant for RAG systems centers around three key innovations, designed to maintain exact retrieval while enhancing efficiency and semantic relevance:
* Chunk-aware Balancing: Standard AVL tree rotations are solely based on node height differences. We propose modifying the balancing criteria to incorporate document chunk sizes and their semantic distribution within the tree. This chunk-aware balancing mechanism will dynamically adjust rotations to maintain tree balance even when inserting variable-sized document chunks or semantically clustered data, preventing performance degradation and ensuring consistent O(log n) exact retrieval time. This contrasts with ANN methods that sacrifice retrieval guarantees for speed gains.
* Semantic Similarity-Based Organization with UMAP + Z-Order Mapping: To effectively index high-dimensional semantic embeddings within an AVL tree, we employ a two-stage dimensionality reduction and mapping strategy. First, BioBERT embeddings of document chunks are generated to capture semantic meaning. These high-dimensional embeddings are then reduced to a lower-dimensional space (e.g., 2 or 3 dimensions) using UMAP (Uniform Manifold Approximation and Projection). UMAP is chosen for its ability to preserve the global and local semantic structure of the embedding space. Subsequently, these low-dimensional UMAP embeddings are mapped to a single scalar value using Z-Order curves (also known as Morton curves). Z-Order mapping effectively linearizes multi-dimensional data while preserving spatial locality to a significant extent. These scalar Z-Order values serve as keys in our AVL tree, enabling exact and ordered retrieval based on semantic proximity. Furthermore, each AVL tree node will store aggregate embeddings (e.g., centroid embedding) of its descendant chunks. This semantic aggregation at each node facilitates efficient pruning and range query optimization, clustering semantically similar chunks within tree sub-branches.
* Efficient Pruning Mechanisms: To optimize search efficiency without compromising exactness, we introduce dynamic pruning. During tree traversal for a query, we compare the query embedding with the aggregate embeddings stored at each node. If the semantic dissimilarity between the query embedding and a node's aggregate embedding exceeds a predefined relevance threshold, we can confidently prune the entire subtree rooted at that node, as it is unlikely to contain relevant documents. This semantic pruning significantly narrows down the search space for most queries, enhancing retrieval speed while still guaranteeing exact retrieval within the un-pruned branches. This dynamic pruning contrasts with the static, pre-computed approximations inherent in FAISS, allowing for more flexible and context-aware search space reduction.
#### Core Argument:
* FAISS + ANN Limitations in Medical RAG: ANN methods like FAISS, while fast, inherently trade precision for speed. Techniques like IVF create partitions in the embedding space, searching only within a subset of these partitions. This approximation, although efficient, risks missing relevant documents that fall outside the searched partitions. In medical RAG, this can be critical; for instance, overlooking a rare but vital study on a drug's side effects due to ANN approximation could have severe consequences. Hyperparameter tuning in ANN methods to balance speed and precision adds further complexity and uncertainty to retrieval accuracy.
* AVL Tree Advantages for Precision and Traceability: Optimized AVL trees guarantee exact retrieval by systematically searching through all mapped embedding values within a defined semantic range. The UMAP + Z-Order mapping is crucial here, as it maintains sufficient semantic locality after dimensionality reduction, enabling meaningful range queries. This exactness is paramount in medical RAG, where traceability and justification of information are crucial. For example, in supporting a diagnosis, an AVL tree-based system can reliably retrieve all relevant literature, providing a complete and auditable set of sources.
* Medical RAG Context Justification: Medical datasets, such as curated collections of PubMed articles, often exhibit moderate sizes but extremely high value. In such scenarios, the O(log n) retrieval time of optimized AVL trees becomes practically feasible and acceptable, especially when compared to the potential risks associated with the imprecise nature of ANN methods. In high-stakes medical decision support, precision, reliability, and interpretability are often prioritized over raw throughput. Our proposed solution directly addresses these needs, offering a robust, exact, and interpretable retrieval mechanism.
Our optimized AVL tree approach aims to offer a compelling alternative to FAISS + ANN in precision-critical RAG contexts. It provides the crucial benefit of exact retrieval with enhanced efficiency and semantic awareness, while maintaining interpretability and traceability, contrasting sharply with the inherent approximation and complexity of ANN-based methods and the pre-computation overhead of index-based solutions.

### 4. Implementation
The optimized AVL tree variant will be implemented in Python, building upon a standard AVL tree framework and incorporating the following key modules:
* UMAP + Z-Order Mapping Module: This module will handle the two-stage embedding transformation. It will utilize:
    * BioBERT: To generate high-dimensional semantic embeddings for document chunks. Libraries like transformers will be used for BioBERT model integration.
    * UMAP: To reduce the dimensionality of BioBERT embeddings. Libraries like umap-learn will be employed for UMAP implementation. The target dimensionality (e.g., 2 or 3) will be experimentally determined.
    * Z-Order Curve: To map the low-dimensional UMAP embeddings to one-dimensional scalar keys for AVL tree indexing. A custom or readily available Z-Order implementation in Python will be used.
* Chunk-aware Balancing Module: This module will modify the standard AVL tree rotation logic. It will:
    * Extend the AVL tree node structure to store chunk size and potentially aggregated semantic information.
    * Modify the rotation functions to consider chunk size and semantic distribution during rebalancing decisions, aiming to minimize imbalance caused by variable chunk sizes and semantic clusters.
* Semantic Pruning Module: This module will implement the dynamic pruning mechanism during tree traversal. It will:
    * Calculate and store aggregate embeddings (e.g., centroid) at each AVL tree node, representing the semantic center of its subtree.
    * Implement a pruning function that compares the query embedding to a node's aggregate embedding using a semantic similarity metric (e.g., cosine similarity).
    * Define a relevance threshold; if the similarity falls below this threshold, the subtree is pruned from the search. The threshold will be empirically tuned.
* Testing and Validation Suite: A comprehensive suite of unit tests will be developed to ensure the correctness and efficiency of each module and the integrated system. This suite will include tests for:
    * Correct AVL tree operations (insertion, deletion, search) with chunk-aware balancing.
    * Accuracy and efficiency of UMAP + Z-Order mapping.
    * Effectiveness of semantic pruning in reducing search space without compromising exactness.
    * End-to-end retrieval performance on diverse datasets, including medical document collections.

### 5. Evaluation
We will rigorously evaluate our optimized AVL tree approach through the following methods:
* Benchmarks on Medical and Standard RAG Datasets:
    * Medical Datasets: We will use subsets of PubMed, focusing on specific medical domains (e.g., cardiology, oncology) to simulate realistic medical RAG scenarios.
    * Standard RAG Datasets: We will also evaluate on standard RAG benchmarks like MS MARCO to assess generalizability and compare performance in broader contexts.
    * Comparative Analysis: Retrieval performance will be compared against:
        * Standard AVL Trees: To isolate the benefits of our proposed optimizations.
        * FAISS + ANN (HNSW and IVF variants): To benchmark against state-of-the-art approximate retrieval methods, particularly focusing on precision trade-offs.
* Complexity Analysis:
    * Time Complexity: Empirically measure and analyze retrieval times for our optimized AVL tree, standard AVL trees, and FAISS + ANN across varying dataset sizes. We will verify the O(log n) exact retrieval time of our approach and contrast it with the near-constant time approximations of ANN methods.
    * Space Complexity: Analyze memory usage for each method, considering index size and runtime memory footprint, to understand practical resource trade-offs.
* Ablation Studies: To isolate the contribution of each proposed innovation, we will conduct ablation studies by systematically removing each optimization:
    * Without Chunk-aware Balancing: Evaluate performance with standard AVL balancing to assess the impact of our modified balancing strategy.
    * Without UMAP + Z-Order Mapping: Test with direct indexing of high-dimensional BioBERT embeddings in standard AVL trees (if feasible) or alternative dimensionality reduction/mapping techniques to evaluate the effectiveness of our chosen mapping approach.
    * Without Semantic Pruning: Measure performance without pruning to quantify the efficiency gains from our dynamic pruning mechanism.
* Performance Metrics: We will employ the following metrics to comprehensively evaluate retrieval performance:
    * Precision@k: Measures the proportion of relevant documents among the top-k retrieved results, crucial for precision-focused medical RAG.
    * Recall@k: Measures the proportion of relevant documents retrieved within the top-k results out of all relevant documents in the dataset.
    * Retrieval Time (Latency): Measures the time taken to perform a retrieval query.
    * Memory Usage: Measures the memory footprint of the index and retrieval process.
This multi-faceted evaluation will rigorously validate the enhancements of our optimized AVL tree over standard AVL trees and comprehensively compare its performance with FAISS + ANN, particularly highlighting its advantages in precision-driven medical RAG applications.

### 6. Expected Outcomes
We anticipate achieving the following key outcomes:
* Demonstrably Superior Precision in Medical RAG: We expect our optimized AVL tree to significantly outperform FAISS + ANN in precision and recall metrics on medical RAG benchmarks. Specifically, we aim for a measurable improvement (e.g., >10% increase in Precision@10 and Recall@10) compared to FAISS + ANN when evaluated on medical datasets, ensuring minimal loss of critical medical literature during retrieval.
* Competitive Retrieval Efficiency for Moderate-Scale Datasets: We expect to achieve retrieval times significantly faster than standard AVL trees and approach the efficiency of mainstream methods like FAISS + ANN for moderate-sized medical datasets. While not aiming to surpass ANN methods in raw speed, we aim to minimize the performance gap while maintaining exact retrieval.
* Enhanced Relevance and Traceability for Medical Applications: The exact retrieval and semantic organization of our approach will provide inherently higher relevance and traceability compared to approximate methods. This will make our optimized AVL tree a particularly compelling choice for high-stakes medical RAG deployments where interpretability and auditability of retrieval results are critical for building trust and ensuring responsible AI application in healthcare.
This project aims to deliver a technically sound and practically valuable solution for precision-critical RAG applications, particularly in the medical domain, advancing the state-of-the-art in retrieval-augmented language models and contributing to more reliable and trustworthy AI systems in high-stakes scenarios.




## Modules for Implementation

#### 1. Embedding Generation Module
- **Purpose**: Generate semantic embeddings for medical document chunks using BioBERT to capture their meaning.
- **Key Tasks**:
  - Integrate the BioBERT model using the `transformers` library from Hugging Face.
  - Preprocess medical documents (e.g., PubMed articles) by splitting them into chunks suitable for embedding (e.g., fixed-size segments or semantically coherent units).
  - Generate high-dimensional embeddings (e.g., 768-dimensional vectors) for each document chunk.
- **Implementation Tips**:
  - Use a preprocessing function to clean text (remove noise, normalize terms) and split documents.
  - Leverage BioBERT’s pre-trained weights fine-tuned for medical text to ensure domain-specific semantic accuracy.
  - Save embeddings to disk (e.g., in NumPy arrays) for reuse across experiments.

#### 2. Dimensionality Reduction and Mapping Module
- **Purpose**: Transform high-dimensional BioBERT embeddings into scalar keys for AVL tree indexing while preserving semantic relationships.
- **Key Tasks**:
  - Implement UMAP (Uniform Manifold Approximation and Projection) using `umap-learn` to reduce embeddings to a lower-dimensional space (e.g., 2D or 3D).
  - Apply Z-Order curve mapping to convert the reduced embeddings into one-dimensional scalar values, ensuring spatial locality is maintained.
  - Validate that the mapping preserves semantic proximity (e.g., nearby points in the original space remain close in the scalar space).
- **Implementation Tips**:
  - Experiment with UMAP parameters (e.g., `n_components`, `n_neighbors`, `min_dist`) to balance dimensionality reduction quality and runtime.
  - Write a custom Z-Order function or adapt an existing library (e.g., `morton`) to handle your reduced dimensions.
  - Test the module standalone by visualizing reduced embeddings and their scalar mappings.

#### 3. Optimized AVL Tree Implementation
- **Purpose**: Build the core data structure with enhancements tailored for RAG, including chunk-aware balancing.
- **Key Tasks**:
  - Extend a standard AVL tree implementation in Python to include additional node attributes (e.g., chunk size, aggregate embeddings like centroids).
  - Modify rotation functions (left, right, left-right, right-left) to account for chunk size and semantic distribution during balancing, ensuring O(log n) performance.
  - Use the scalar keys from the mapping module as the ordering criterion for nodes.
- **Implementation Tips**:
  - Start with a basic AVL tree class, then incrementally add chunk-aware features.
  - Store chunk metadata (e.g., original text or embedding references) in nodes for retrieval.
  - Test balancing behavior with synthetic datasets of varying chunk sizes and insertion patterns.

#### 4. Semantic Pruning Mechanism
- **Purpose**: Improve search efficiency by pruning irrelevant subtrees based on semantic similarity.
- **Key Tasks**:
  - Compute and store aggregate embeddings (e.g., centroids of subtree embeddings) in each AVL tree node during insertion or updates.
  - Develop a pruning function that compares a query embedding to a node’s aggregate embedding using a similarity metric (e.g., cosine similarity).
  - Set a relevance threshold (tunable parameter) to decide when to prune a subtree—subtrees with similarity below the threshold are skipped.
- **Implementation Tips**:
  - Use `numpy` or `scipy` for efficient cosine similarity calculations.
  - Implement pruning within the tree traversal logic (see Query Processing module below).
  - Tune the threshold empirically using a validation set to balance efficiency and recall.

#### 5. Query Processing and Retrieval Module
- **Purpose**: Handle incoming queries by retrieving relevant document chunks efficiently.
- **Key Tasks**:
  - Generate a BioBERT embedding for the query input, mirroring the document embedding process.
  - Map the query embedding to a scalar key using the same UMAP + Z-Order pipeline as the documents.
  - Traverse the AVL tree using the query’s scalar key, applying semantic pruning to skip irrelevant subtrees.
  - Retrieve and rank document chunks based on their similarity to the query embedding.
- **Implementation Tips**:
  - Implement a recursive traversal function that integrates pruning logic.
  - Return a ranked list of chunks (e.g., top-k results) with similarity scores for evaluation.
  - Optimize traversal to minimize redundant similarity computations.

#### 6. Evaluation and Testing Suite
- **Purpose**: Validate the system’s correctness, efficiency, and effectiveness for your EMNLP paper.
- **Key Tasks**:
  - Write unit tests for each module (e.g., embedding generation, AVL balancing, pruning accuracy) using `pytest`.
  - Create integration tests to verify the end-to-end retrieval pipeline.
  - Develop benchmarks to measure precision@k, recall@k, retrieval time, and memory usage on datasets like PubMed subsets and MS MARCO.
  - Conduct ablation studies by disabling features (chunk-aware balancing, pruning, UMAP + Z-Order) to quantify their impact.
- **Implementation Tips**:
  - Use a testing framework to automate evaluation across multiple dataset sizes and query types.
  - Log results (e.g., with `pandas` or `csv`) for easy analysis and visualization.
  - Compare against baselines (standard AVL trees, FAISS + ANN) using the same metrics.

#### 7. Data Preparation and Management
- **Purpose**: Prepare and manage datasets for training, testing, and evaluation.
- **Key Tasks**:
  - Collect medical datasets (e.g., PubMed subsets for cardiology or oncology) and standard RAG datasets (e.g., MS MARCO).
  - Preprocess data into a consistent format (e.g., tokenized text ready for BioBERT).
  - Split data into training (for UMAP fitting) and testing sets, ensuring no overlap.
- **Implementation Tips**:
  - Automate data downloading and preprocessing with scripts (e.g., using `requests` or `pubmed_parser`).
  - Store preprocessed data and embeddings in a structured directory to streamline experiments.
  - Document dataset statistics (e.g., number of documents, average chunk size) for your paper.

#### 8. Documentation and Reporting
- **Purpose**: Document the codebase and experiments for reproducibility and your EMNLP submission.
- **Key Tasks**:
  - Write inline comments and docstrings for each module to explain functionality and usage.
  - Maintain an experiment log (e.g., in a Jupyter notebook or Markdown file) tracking hyperparameters, results, and Ablation study outcomes.
  - Generate visualizations (e.g., precision-recall curves, retrieval time plots) using `matplotlib` or `seaborn` for your paper.
- **Implementation Tips**:
  - Use a consistent docstring format (e.g., Google or NumPy style) for clarity.
  - Automate result aggregation and plotting to save time during write-up.
  - Include a `README.md` with setup instructions and module descriptions.
