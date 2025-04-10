import tkinter as tk
from datetime import datetime, timedelta
import time

class DisplaySignLanguage:
    def __init__(self, root, words, handleCompletion, gready_timer = 3, action_timer = 5):
        self.root = root
        self.root.configure(bg='black')

        # Tracks if user has completed all sign gestures prior to exit
        self.hasCompleted = False

        # Timers
        self.gready_timer = gready_timer
        self.action_timer = action_timer

        # Function for Handling Word Signing Completion
        self.handleCompletion = handleCompletion

        # List of words to display
        self.words = words
        self.current_word_index = 0

        # Dictionary to store word data
        self.word_data = {word: {"name": word, "timestamps": [0,0], "valid": True} for word in self.words}

        # Top left label for validity
        self.valid_info_label = tk.Label(root, text="Press Space Bar To Toggle Valid", fg="white", bg="black", font=("Arial", 21))
        self.valid_info_label.place(relx=0.001, rely=0.0, anchor='nw')

        # Top left label for validity
        self.valid_label = tk.Label(root, text="Valid: <>", fg="white", bg="black", font=("Arial", 43))
        self.valid_label.place(relx=0.0, rely=0.1, anchor='nw')

        # Top right label for countdown
        self.countdown_label = tk.Label(root, text="", fg="white", bg="black", font=("Arial", 43))
        self.countdown_label.place(relx=1.0, rely=0.0, anchor='ne')

        # Center label for word display
        self.word_label = tk.Label(root, text="", fg="white", bg="black", font=("Arial", 82))
        self.word_label.place(relx=0.5, rely=0.5, anchor='center')

        # Start/Stop button
        self.button = tk.Button(root, text="Start", command=self.toggle_start_stop, width=6, height=1, font=("Arial", 40))
        self.button.place(relx=0.5, rely=0.7, anchor='center')

        # Variable to track if the countdown is running
        self.running = False

    def toggle_start_stop(self):
        if self.running:
            self.running = False
            self.button.config(text="Start")
            self.root.after_cancel(self.countdown_id)
            self.countdown_label.config(text="")
        else:
            self.running = True
            self.button.config(text="Stop")
            self.start_sequence()

    def start_sequence(self):
        if self.current_word_index >= len(self.words):
            self.word_label.config(text="End")
            self.countdown_label.config(text="")
            return

        word = self.words[self.current_word_index]
        self.word_label.config(text=word)

        self.countdown(self.gready_timer, self.start_countdown, "Get Ready: ", word)

    def countdown(self, time_left, next_step, tag, word):
        if time_left >= 0:
            self.countdown_label.config(text=tag + str(time_left))
            self.valid_label.config(text="Valid: "+str(self.word_data[word]["valid"]))
            self.countdown_id = self.root.after(1000, self.countdown, time_left - 1, next_step, tag, word)
        else:
            next_step()

    def start_countdown(self):
        word = self.words[self.current_word_index]

        custom_epoch = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        epoch_ms = int(custom_epoch.timestamp() * 1000)
        timestamp = int(time.time() * 1000) - epoch_ms

        self.word_data[word]["valid"] = True
        self.word_data[word]["timestamps"][0] = timestamp

        self.countdown(self.action_timer, self.finish_countdown, "Sign:", word)

    def finish_countdown(self):
        word = self.words[self.current_word_index]

        custom_epoch = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        epoch_ms = int(custom_epoch.timestamp() * 1000)
        timestamp = int(time.time() * 1000) - epoch_ms

        self.word_data[word]["timestamps"][1] = timestamp

        # Move to the next word
        self.current_word_index += 1

        if self.current_word_index < len(self.words):
            self.start_sequence()
        else:
            self.appEnd()
            self.word_label.config(text="End")
            self.countdown_label.config(text="")

    def appEnd(self):
        self.hasCompleted = True
        self.handleCompletion(data=self.word_data)

    def toggle_valid(self, event=None):
        word = self.words[self.current_word_index]
        self.word_data[word]["valid"] = not self.word_data[word]["valid"]

    def get_data(self):
        return self.word_data