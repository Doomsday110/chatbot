AI Chatbot

Tessa is an AI-powered chatbot built using Python, Tkinter for the GUI, and the LangChain library with an Ollama model (LLaMA3). It can engage in conversations, solve mathematical expressions, detect emotions, and provide voice responses.


Features:
- Conversational AI** – Uses the LLaMA3 model for generating responses.
- Small Talk & Emotional Responses** – Handles basic greetings and emotional queries.
- Math Solver** – Evaluates mathematical expressions using SymPy.
- Voice Output** – Uses `pyttsx3` for text-to-speech.
- User-Friendly GUI** – Built with Tkinter for an interactive chat experience.


Installation:
1. Clone the Repository:
   git clone https://github.com/Doomsday110/chatbot.git
2. Install Dependencies:
   pip install -r requirements.txt
  (Ensure you have Python installed. If not, download it from https://python.org)
3. Run the Chatbot:
   python main.py


Dependencies:
tkinter (Built-in with Python)
pyttsx3
langchain
langchain_ollama
sympy
re
threading

Install them manually if needed using:
pip install pyttsx3 langchain langchain_ollama sympy


Usage:
- Type a message in the input box and press "Send".  
- Click the voice button to toggle sound ON/OFF.  
- Supports small talk, emotional responses, and math computations. 
- Chatbot name: Tessa.


Future Improvements:
- Adding memory for context-based responses.
- Enhancing the GUI for a better user experience.
- Integrating more AI models for improved conversation.


License:
This project is open-source. Feel free to modify and enhance it!


Created by Doomsday110 
