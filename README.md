# Virtual AI Tutor
This is the prototype of a project through which we can laverage the generative AI to serve the student a virtual teacher where they can ask anything from their course content or text book and it will answer as per your class, age and subject. The aim of this project is to give a platform to students where they can quickly resolve their queries.

## Chat Interface
An HTML file is there which will server as an AI chatbot where student an ask question and AI will respond from the textbook or course content. It will only entertain question relevant to their class and subject.

![image](https://github.com/user-attachments/assets/7a967920-2e0d-4290-851e-2661c5c826ca)

In the backend an api is called to get response from an LLM powered by Gemini. API is built on FastAPI. This is actually a RAG app which is history aware by session. We can use any frontend technology and integrate with the api to make a powerful tutor chatbot.

Currently the app usses class 11 and 12 NCERT computer science books for the context.
