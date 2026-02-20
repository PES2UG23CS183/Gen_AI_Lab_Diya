# Unit 2 Assignment: Building a Mixture of Experts (MoE) Router
**Topic:** Advanced Architecture using Groq API  
**Tools:** Python, Groq API, Dotenv

## Overview
This folder contains a .ipynb file which builds a "Smart Customer Support Router" using a Mixture of Experts (MoE) architecture. It also contains a pdf file with the output screenshots of the MOE Router. It builds a router that takes a user query, decides which expert is best suited for it, and then forwards the query to that specific expert configuration. It classfies the given prompt into:
- A Technical Expert for bug reports.
- A Billing Expert for refund requests.
- A General Expert for for casual chat.

Additionally, it also incorporates a "Tool Use" expert which routes the user prompt to a function that (mock) fetches data, instead of just an LLM response.
  
