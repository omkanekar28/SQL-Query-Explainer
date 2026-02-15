# SQL-Query-Explainer
A small instruction-tuned language model fine-tuned to translate SQL queries into concise, human-readable explanations. Emphasizes dataset curation, LoRA fine-tuning, and evaluation of generation quality on structured queries.

# Flow

## 1. Dataset Creation
- **Data Collection**: Gather a diverse set of SQL queries (SELECT, INSERT, UPDATE, DELETE) from various sources (e.g., open-source databases, SQL tutorials).
- **Annotation**: Annotate each SQL query with a clear, concise explanation of its functionality using LLMs.
- **Generalization**: Ensure the dataset includes a wide range of query types and complexities to promote generalization (Query type, length, etc).

## 2. Model Fine-tuning
- **Base Model**: Start with a pre-trained language model (e.g., GPT-3, BERT).
- **LoRA Fine-tuning**: Apply Low-Rank Adaptation (LoRA) to fine-tune the model on the annotated dataset, allowing it to learn the mapping from SQL queries to explanations efficiently while keeping the base model's parameters mostly unchanged.
- **Evaluation during Training**: Monitor the model's performance on a validation set to prevent overfitting and ensure that it learns to generate accurate explanations.
- **Quantization**: Implement quantization techniques to reduce the model size and improve inference speed without significantly compromising performance.

## 3. Evaluation
- **Metrics**: Use BLEU, ROUGE, and LLM as a judge to assess the quality of generated explanations.
- **Human Evaluation**: Conduct human evaluations to assess the clarity and accuracy of the explanations, ensuring they are understandable to users with varying levels of SQL knowledge.

## 4. Deployment
- **Streamlit App**: Develop a Streamlit application that allows users to input SQL queries and receive concise explanations in real-time.