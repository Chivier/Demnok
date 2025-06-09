import json

def load_docs_from_jsonl(jsonl_path):
    """
    Load documents from JSONL file into a dictionary.
    
    Args:
        jsonl_path: Path to JSONL file
    
    Returns:
        Dictionary mapping chunk_id to content
    """
    docs = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            docs[data['chunk_id']] = data['text']
    return docs

def load_questions_from_jsonl(jsonl_path):
    """
    Load questions from JSONL file into a list.
    
    Args:
        jsonl_path: Path to JSONL file
    
    Returns:
        List of questions
    """
    questions = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            questions.append(data['question'])
    return questions

# Class-based approach for better organization
class RAGPromptTemplate:
    def __init__(self, jsonl_path=None, docs_dict=None):
        """Initialize with JSONL file path"""
        if not jsonl_path and not docs_dict:
            raise ValueError("Either jsonl_path or docs_dict must be provided")

        self.jsonl_path = jsonl_path

        if docs_dict is None:
            self.docs_dict = self.load_docs_from_jsonl(jsonl_path)
        else:
            self.docs_dict = docs_dict
        
        self.base_template = """
With provided related documents:
<documents_section>
{docs_section}
</documents_section>

Answer the question:
<question_section>
{question}
</question_section>"""

        self.reordered_template = """
With provided related documents:
<documents_section>
{docs_section}
</documents_section>

Answer the question:
<question_section>
{question}
</question_section>

The importance ranking of documents for this query is:
<importance_ranking_section>
{importance_ranking}
</importance_ranking_section>

Please prioritize information from higher-ranked documents when there are conflicts or overlapping information."""
    
    def load_docs_from_jsonl(self, jsonl_path):
        """Load documents from JSONL file"""
        docs = {}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                docs[data['chunk_id']] = data['text']
        return docs
    
    def format_docs(self, reordered_doc_ids):
        """Format documents section using doc IDs"""
        docs_section = ""
        for doc_id in reordered_doc_ids:
            if doc_id in self.docs_dict:
                content = self.docs_dict[doc_id]
                docs_section += f"[Doc {doc_id}] {content}\n\n"
            else:
                print(f"Warning: Doc {doc_id} not found")
        return docs_section.strip()
    
    def format_importance(self, original_doc_order):
        """Format importance ranking"""
        return " > ".join([f"[Doc {doc_id}]" for doc_id in original_doc_order])
    
    def create_prompt(self, reordered_doc_ids, original_doc_order, question):
        """Create the complete RAG prompt"""
        return self.reordered_template.format(
            docs_section=self.format_docs(reordered_doc_ids),
            question=question,
            importance_ranking=self.format_importance(original_doc_order)
        )
    
    def create_prompts(self, reordered_doc_ids_list, original_doc_orders, questions):
        """Create multiple prompts for a list of document IDs and orders"""
        prompts = []
        for reordered_doc_ids, original_doc_order, question in zip(reordered_doc_ids_list, original_doc_orders, questions):
            prompts.append(self.create_prompt(reordered_doc_ids, original_doc_order, question))
        return prompts
    
    def create_original_order_prompt(self, original_doc_order, question):
        """Create a prompt using the original document order"""
        return self.base_template.format(
            docs_section=self.format_docs(original_doc_order),
            question=question
        )
    
    def create_original_order_prompts(self, original_doc_orders, questions):
        """Create multiple prompts using the original document orders"""
        prompts = []
        for original_doc_order, question in zip(original_doc_orders, questions):
            prompts.append(self.create_original_order_prompt(original_doc_order, question))
        return prompts