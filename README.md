🤖 Text Morph: A Secure, Multi-Task NLP Platform
Text Morph is a secure, scalable, and manageable platform designed to enhance content efficiency and accessibility through automated summarization and advanced paraphrasing. It features a robust architecture and a comprehensive Admin Dashboard with Role-Based Access Control (RBAC) to ensure data security and platform governance.

✨ Key Features

Abstractive Summarization: Condenses long-form text using fine-tuned BART and PEGASUS Transformer models to produce coherent and relevant summaries.
Advanced Paraphrasing: Rewrites text while preserving semantic similarity, utilizing PEGASUS and FLAN-T5 for diverse, high-quality output.
Readability Analysis: Every output includes the Flesch Reading Ease Score, allowing users to quantify the comprehension level and measure the impact of content transformation.
Role-Based Access Control (RBAC): Ensures secure access with distinct roles (user/admin) validated via the JWT payload before granting access to specific endpoints.
JWT Authentication: Uses JSON Web Tokens for secure, stateless sessions, optimizing performance by eliminating repeated credential verification.
Admin Dashboard: Provides real-time insights, user management, and Content Curation (the ability to view, correct, or delete generated content) for quality control and auditing.

🛠️ Technology Stack (Architectural Stack)
The platform is built on a high-performance, asynchronous foundation:

Component	Technology	Role
Frontend/UI	Streamlit	
Provides an intuitive, responsive Single-Page Application (SPA) user interface.

Backend/API	FastAPI	
Serves as the high-performance, asynchronous API gateway, managing requests and authentication securely.

Model Layer	Hugging Face Transformers	
Environment for managing, fine-tuning, and deploying specialized NLP models (BART, PEGASUS, FLAN-T5).

Database	MySQL	
Ensures persistent, secure storage for user credentials, generated content history, and analytics data.

🚀 Getting Started
To set up the Text Morph environment, you will need to:

Clone this repository.
Set up the MySQL database for persistent storage of user data and content history.
Configure environment variables for the FastAPI backend, including database connection strings and JWT secrets.
Download the necessary fine-tuned Transformer models (BART, PEGASUS, FLAN-T5) and configure the Model Layer for inference.
Run the FastAPI service (Backend) to handle all API requests.
Launch the Streamlit application (Frontend) to access the interactive user interface.

📊 Evaluation & Metrics
The NLP core was subjected to rigorous assessment:

Performance: Evaluated using ROUGE scores for summarization and paraphrasing quality.

Accessibility: Measured using the Flesch Reading Ease Score to quantify the comprehension level of generated text.

🔒 Security & Governance
Security is enforced at multiple layers:

Authentication: Utilizes JWT for secure, stateless sessions.

Authorization (RBAC): Backend logic validates the user's role (user/admin) from the JWT before granting access to sensitive APIs, such as the administrative endpoints.

Credential Storage: Ensures secure credential storage using industry-standard hashing techniques (e.g., bcrypt).
