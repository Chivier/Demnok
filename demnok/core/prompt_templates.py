SIMPLE_RAG_PROMPT = '''
Answer the question based on the given document.
Only give me the answer and do not output any other words.
The following are given documents:
<documents> 
{}.
</documents>

Address the following question: {}.
'''

DOCUMENT_PROMPT = '''
{}
'''

SIMPLE_COT_TEMPLATE = '''
Q: {}
A: Let's think step by step.

**IMPORTANT**
- Structure each thought in paragraphs.
'''

INNER_REVISE_PROMPT = '''
With the provided information:
{},

and your previous revised answers: 
{},

modify your following answer: 
{}.

**IMPORTANT**
- Just output the revised answer directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
- You need to check whether the answer is correct.
- If you find some errors in the answer, revise the answer to make it better.
- If you find some necessary details are ignored, add it to make the answer more plausible according to the related text.
- If you find the answer is right and do not need to add more details, just output the original answer directly.
'''


FIRST_RAT_PROMPT = '''
With the provided information as context: 
{}.

Address the following question in a step-by-step manner: {}.

**IMPORTANT**
- Structure each thought in paragraphs, ensuring each is clearly separated by "##". 
- Refrain from using digits or numbered lists. Limit the response to a maximum of {} paragraphs.
'''


SUMMARIZE_PROMPT = '''
Given the following related information:
{},

and your revised answer chain:
{},

Summarize and structure your answer to the following question: {}.

**IMPORTANT**
- DO NOT add additional explanations or introducement in the answer unless you are asked to.
- You need to check whether the answer is correct.
- Start with "In summary,"
- If you find some errors in the answer, revise the answer to make it better.
- If you find some necessary details are ignored, add it to make the answer more plausible according to the related text.
- If you find the answer is right and do not need to add more details, just output the original answer directly.
'''