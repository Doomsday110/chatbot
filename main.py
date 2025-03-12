import threading
import time
import tkinter as tk
from tkinter import scrolledtext
import pyttsx3  # For voice output
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import re  # For emotion detection
import sympy as sp  # For mathematical computations

# Chatbot Configuration
template = """
The bot's name is Tessa. Answer the user's query accurately.

Conversation history: {context}

User's question: {question}

Answer:
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Small talk responses
SMALL_TALK_RESPONSES = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hey there! What's on your mind?",
    "how are you": "I'm just a chatbot, but I'm feeling great! How about you?",
    "what's up": "Not much, just here to chat with you!",
    "who are you": "I'm Tessa, your AI assistant!",
    "thank you": "You're very welcome! 😊",
    "bye": "Goodbye! Have a great day!",
    "where are you": "I am a locally built chatbot and I am present inside your computer."
}

# Emotional responses
EMOTIONAL_RESPONSES = {
    "i am sad": "I'm sorry to hear that. Want to talk about it?",
    "i feel lonely": "You're not alone! I'm here to chat with you. 😊",
    "i am happy": "That's wonderful! What made you happy?",
    "i am angry": "It's okay to feel angry. Do you want to talk about it?",
    "i am stressed": "That sounds tough. Try taking a deep breath. I'm here for you!",
    "i am excited": "That’s amazing! Tell me what’s got you excited!"
}

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tessa - AI Chatbot")
        self.root.resizable(False, False)  # Disable maximize button

        # Chat Display
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, height=20, width=50)
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # User Input
        self.user_input = tk.Entry(root, width=40)
        self.user_input.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        # Send Button
        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")

        # Voice Toggle Button
        self.voice_enabled = True
        self.voice_button = tk.Button(root, text="🔊 Voice: ON", command=self.toggle_voice)
        self.voice_button.grid(row=2, column=0, columnspan=2, pady=5)

        # Typing Indicator
        self.typing_label = tk.Label(root, text="", fg="gray")
        self.typing_label.grid(row=3, column=0, columnspan=2)

        # Initialize Conversation Context
        self.context = "The bot's name is Tessa."
        self.engine = pyttsx3.init()  # Initialize text-to-speech engine
        self.engine.setProperty('rate', 160)  # Reduce voice speed (default ~200)

        # Display welcome message
        self.display_welcome_message()

    def update_chat(self, message, color):
        """Update chat display with new messages."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, message, (color,))
        self.chat_display.tag_config(color, foreground=color)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.yview(tk.END)

    def display_welcome_message(self):
        """Display initial welcome message when the chatbot starts."""
        welcome_message = "Hi, my name is Tessa."
        self.update_chat(f"Tessa: {welcome_message}\n", "deeppink")
        self.speak(welcome_message)

    def send_message(self):
        user_text = self.user_input.get().strip().lower()  # Convert to lowercase for better matching
        if user_text:
            self.update_chat(f"You: {user_text}\n", "black")
            self.user_input.delete(0, tk.END)

            # Handle small talk and emotional queries before invoking AI
            response = self.handle_small_talk(user_text) or self.handle_emotional_queries(user_text)
            if response:
                self.update_chat(f"Tessa: {response}\n", "deeppink")
                self.speak(response)
            elif self.is_math_expression(user_text):
                response = self.solve_math_expression(user_text)
                self.update_chat(f"Tessa: {response}\n", "deeppink")
                self.speak(response)
            else:
                threading.Thread(target=self.get_response, args=(user_text,), daemon=True).start()

    def handle_small_talk(self, user_text):
        """Check if the user input matches a small talk response."""
        for key in SMALL_TALK_RESPONSES:
            if key in user_text:
                return SMALL_TALK_RESPONSES[key]
        return None  # No small talk detected

    def handle_emotional_queries(self, user_text):
        """Check if the user input matches an emotional query."""
        for key in EMOTIONAL_RESPONSES:
            if key in user_text:
                return EMOTIONAL_RESPONSES[key]
        return None  # No emotional response detected

    def is_math_expression(self, user_text):
        """Check if the input is a mathematical expression."""
        return bool(re.search(r'[\d\+\-\*/\^=]', user_text))  # Looks for numbers and math symbols

    def solve_math_expression(self, expression):
        """Solve mathematical expressions using SymPy."""
        try:
            result = sp.sympify(expression)
            return f"The answer is: {result}"
        except (sp.SympifyError, ValueError):
            return "Sorry, I couldn't understand the mathematical expression."

    def get_response(self, user_text):
        """Get AI-generated response from the model."""
        self.typing_label.config(text="Tessa is typing...")
        result = chain.invoke({"context": self.context, "question": user_text})
        self.context += f"\nUser: {user_text}\nTessa: {result}"
        self.typing_label.config(text="")
        self.update_chat(f"Tessa: {result}\n", "deeppink")
        self.speak(result)

    def toggle_voice(self):
        """Toggle voice output on or off."""
        self.voice_enabled = not self.voice_enabled
        self.voice_button.config(text="🔊 Voice: ON" if self.voice_enabled else "🔇 Voice: OFF")

    def speak(self, text):
        """Speak out the chatbot's response if voice is enabled."""
        if self.voice_enabled:
            self.engine.say(text)
            self.engine.runAndWait()

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

  