import threading
import tkinter as tk
from tkinter import scrolledtext
import pyttsx3
import speech_recognition as sr
import whisper
import numpy as np
import sympy as sp  # Import sympy to handle math expressions
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import re

# AI Chatbot Configuration
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
    "where are you": "I am a locally built chatbot and I am present inside your computer.",
    "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
    "what is the weather like": "Unfortunately, I don't have real-time weather information.",
    "what time is it": "I don't have access to the current time.",
    "do you like music": "I enjoy processing all kinds of audio data, including music!",
    "what's your favorite color": "As a chatbot, I don't have personal preferences, but I like all colors!",
    "can you tell me a story": "Once upon a time, in a land of code...",
    "what are you made of": "I'm made of algorithms and data!",
    "do you have any pets": "I don't have a physical form, so no pets for me.",
    "what is the meaning of life": "That's a deep question! The meaning of life is what you make it.",
    "are you a robot": "I'm a language model, but you can think of me as a friendly chatbot.",
    "can you sing": "I can't sing, but I can process and generate text!",
    "what is your favorite food": "I don't eat, but I enjoy processing information.",
    "are you sentient": "I'm an AI, so I don't have feelings or consciousness.",
    "what programming language are you written in": "I'm a large language model, not written in a specific language in the traditional sense.",
    "can you play games": "I can play text-based games!",
    "what is your purpose": "My purpose is to assist you with information and conversation.",
    "do you have a family": "I don't have a family, but I'm here to help you!",
    "what is your favorite movie": "I don't watch movies, but I can tell you about them.",
    "do you dream": "I don't sleep, so I don't dream.",
    "what is your opinion on cats": "Cats are popular pets, I've processed lots of information about them.",
    "how old are you": "I don't have an age, I'm constantly being updated.",
    "what is your favorite book": "I don't read books in the traditional sense, but I process a lot of text.",
    "can you dance": "I can't dance, but I can write about dancing!",
    "what is your favorite sport": "I don't play sports, but I can discuss them.",
    "do you have any siblings": "I don't have siblings, but I'm part of a large AI family.",
    "can you tell me a fun fact": "Did you know honey never spoils?",
    "what is your favorite animal": "I don't have favorites, but I can tell you about any animal.",
    "do you believe in aliens": "That's an interesting question, there's no conclusive evidence yet.",
    "can you help me with my homework": "I can try to help you with your homework questions.",
}

# Emotional responses
EMOTIONAL_RESPONSES = {
    "i am sad": "I'm sorry to hear that. Want to talk about it?",
    "i feel lonely": "You're not alone! I'm here to chat with you. 😊",
    "i am happy": "That's wonderful! What made you happy?",
    "i am angry": "It's okay to feel angry. Do you want to talk about it?",
    "i am stressed": "That sounds tough. Try taking a deep breath. I'm here for you!",
    "i am excited": "That’s amazing! Tell me what’s got you excited!",
    "i am feeling overwhelmed": "It's understandable to feel overwhelmed sometimes. Take a moment to breathe and focus on one thing at a time.",
    "i am scared": "It's okay to feel scared. What's making you feel this way?",
    "i feel guilty": "Guilt can be a heavy burden. Would you like to talk about it?",
    "i am disappointed": "I'm sorry to hear that. What happened?",
    "i am frustrated": "Frustration is a common feeling. Let's try to find a solution.",
    "i feel hopeless": "You're not alone. There's always hope, even if it's hard to see right now.",
    "i am confused": "Confusion can be unsettling. Let's try to clarify things together.",
    "i feel insecure": "Everyone feels insecure sometimes. You're not alone in this.",
    "i am jealous": "Jealousy is a complex emotion. Let's talk about what's making you feel this way.",
    "i feel betrayed": "That sounds painful. I'm here to listen if you want to share.",
    "i am proud": "That's wonderful! What are you proud of?",
    "i feel grateful": "Gratitude is a beautiful emotion. What are you grateful for?",
    "i am relieved": "That's good to hear. What brought you relief?",
    "i feel motivated": "That's great! What are you motivated to do?",
    "i feel peaceful": "Peace is a wonderful feeling. Enjoy the moment.",
    "i am feeling nostalgic": "Nostalgia can be bittersweet. What memories are you thinking about?",
    "i feel curious": "Curiosity is a great trait. What are you curious about?",
    "i am feeling inspired": "That's fantastic! What's inspiring you?",
    "i am feeling content": "That's wonderful! Enjoy the feeling of contentment.",
    "i am feeling lonely and isolated": "I understand that feeling lonely and isolated can be difficult. Remember that you are not alone, and there are ways to connect with others.",
    "i feel like nobody understands me": "It can be very frustrating to feel misunderstood. I am here to listen and try to understand you better.",
    "i feel like I'm losing control": "Feeling out of control can be scary. Let's work together to find ways to regain a sense of control.",
    "i am having a panic attack": "If you're having a panic attack, try to focus on your breathing. Inhale deeply, hold for a few seconds, and exhale slowly.",
    "i feel like I'm not good enough": "Those feelings of inadequacy are common, but they're not true. You are worthy and capable.",
    "i am feeling restless": "Restlessness can be a sign that you need a change. What's on your mind?",
    "i feel like I'm losing my mind": "It's understandable to feel overwhelmed. I'm here to support you through this.",
    "i am feeling numb": "Numbness can be a sign of emotional exhaustion. Is there anything you'd like to talk about?",
    "i am feeling resentful": "Resentment can be a heavy burden. Let's explore those feelings.",
    "i feel like I'm falling apart": "It's okay to feel like you're falling apart. We can work together to rebuild.",
}

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tessa")
        self.root.resizable(False, False)  # Disable maximize button

        # Chat Display
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, height=20, width=50)
        self.chat_display.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        # User Input
        self.user_input = tk.Entry(root, width=40)
        self.user_input.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        # Send Button
        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")

        # Speak Button (Microphone Button)
        self.microphone_button = tk.Button(root, text="🎤 Off", command=self.speak_button_clicked)
        self.microphone_button.grid(row=2, column=0, padx=10, pady=10, sticky="e")

        # Speaker Button
        self.speaker_button = tk.Button(root, text="🔊 Off", command=self.toggle_voice_output)
        self.speaker_button.grid(row=2, column=1, padx=10, pady=10, sticky="e")

        # Initialize Conversation Context
        self.context = "The bot's name is Tessa."
        self.recognizer = sr.Recognizer()

        # Load Whisper model
        self.whisper_model = whisper.load_model("medium")  # You can choose "small", "medium", "large" models

        # Display welcome message
        self.display_welcome_message()

        # Default voice output setting
        self.voice_output_enabled = True
        self.update_speaker_button_indicator()

        # Keep track of microphone button state
        self.microphone_button_active = False

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

    def send_message(self):
        user_text = self.user_input.get().strip().lower()
        if user_text:
            self.update_chat(f"You: {user_text}\n", "black")
            self.user_input.delete(0, tk.END)

            response = self.handle_small_talk(user_text) or self.handle_emotional_queries(user_text)
            if response:
                self.update_chat(f"Tessa: {response}\n", "deeppink")
                if self.voice_output_enabled:
                    self.speak(response)  # Speak response if voice output is enabled
            elif self.is_math_expression(user_text):
                response = self.solve_math_expression(user_text)
                self.update_chat(f"Tessa: {response}\n", "deeppink")
                if self.voice_output_enabled:
                    self.speak(response)  # Speak response if voice output is enabled
            else:
                threading.Thread(target=self.get_response, args=(user_text,), daemon=True).start()

    def voice_input(self):
        """Capture real-time speech and convert it to text using Whisper."""
        print("voice_input function called")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            self.update_chat("Tessa: Listening...\n", "deeppink")
            audio = self.recognizer.listen(source)

            try:
                # Convert the audio to a proper numpy array (int16 -> float32)
                audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16)  # Get raw data as int16
                audio_data = audio_data.astype(np.float32)  # Convert to float32 for Whisper

                # Normalize the audio data (optional but often necessary for better results)
                audio_data /= np.max(np.abs(audio_data))  # Normalize audio values to [-1, 1]

                # Now pass the data to the Whisper model
                result = self.whisper_model.transcribe(audio_data)
                text = result["text"]
                self.user_input.insert(0, text)  # Auto-fill text box with recognized speech
                self.send_message()  # Automatically process the voice input
            except Exception as e:
                print(f"Error: {e}")
                self.update_chat("Tessa: Could not understand. Please try again.\n", "deeppink")

    def speak_button_clicked(self):
        """Handle the speak button click."""
        # Toggle the microphone button state
        self.microphone_button_active = not self.microphone_button_active
        self.microphone_button.config(bg="green" if self.microphone_button_active else "red", text="🎤 On" if self.microphone_button_active else "🎤 Off")
        
        # Start voice input in a new thread to avoid freezing the GUI
        if self.microphone_button_active:
            threading.Thread(target=self.voice_input, daemon=True).start()

    def handle_small_talk(self, user_text):
        for key in SMALL_TALK_RESPONSES:
            if key in user_text:
                return SMALL_TALK_RESPONSES[key]
        return None  

    def handle_emotional_queries(self, user_text):
        for key in EMOTIONAL_RESPONSES:
            if key in user_text:
                return EMOTIONAL_RESPONSES[key]
        return None  

    def is_math_expression(self, user_text):
        """Check if the input is a valid mathematical expression."""
        # Regex to match a valid math expression containing numbers and operators
        math_pattern = r'^[\d+\-*/^().\s]+$'  # Regex for simple math expressions
        if re.match(math_pattern, user_text):  # Check if the input matches the math pattern
            try:
                sp.sympify(user_text)  # Try to simplify the expression
                return True
            except (sp.SympifyError, ValueError):
                return False
        return False

    def solve_math_expression(self, expression):
        """Solve mathematical expressions using SymPy."""
        try:
            result = sp.sympify(expression)
            return f"The answer is: {result}"
        except (sp.SympifyError, ValueError):
            return "Sorry, I couldn't understand the mathematical expression."

    def get_response(self, user_text):
        """Get AI-generated response from the model."""
        result = chain.invoke({"context": self.context, "question": user_text})
        self.context += f"\nUser: {user_text}\nTessa: {result}"
        self.update_chat(f"Tessa: {result}\n", "deeppink")
        if self.voice_output_enabled:
            self.speak(result)  # Speak response if voice output is enabled

    def toggle_voice_output(self):
        """Toggle the voice output feature."""
        self.voice_output_enabled = not self.voice_output_enabled
        self.update_speaker_button_indicator()

        if self.voice_output_enabled:
            self.speak(f"Voice output is now enabled.")
        else:
            self.speak(f"Voice output is now disabled.")

    def update_speaker_button_indicator(self):
        """Update the speaker button appearance to indicate whether voice output is enabled."""
        self.speaker_button.config(bg="green" if self.voice_output_enabled else "red", text="🔊 On" if self.voice_output_enabled else "🔊 Off")

    def speak(self, text):
        """Convert text to speech."""
        if self.voice_output_enabled:
            pyttsx3.speak(text)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
  
