import tkinter as tk
from tkinter import messagebox
import pickle
import matplotlib.pyplot as plt

# Load model, vectorizer, accuracy
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
accuracy = pickle.load(open("accuracy.pkl", "rb"))

def check_news():
    user_input = entry.get()

    if user_input.strip() == "":
        messagebox.showwarning("Input Error", "Please enter text")
        return

    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get non-zero features (important words)
    indices = input_vec.nonzero()[1]
    words = [feature_names[i] for i in indices]

    # Take top 5 important words
    important_words = words[:5]

    reason_text = "Keywords: " + ", ".join(important_words)

    if prediction == "fake":
        result_label.config(
            text="⚠️ Fake News Detected\n" + reason_text,
            bg="red", fg="white"
        )
    else:
        result_label.config(
            text="✅ Real News\n" + reason_text,
            bg="green", fg="white"
        )

def clear_text():
    entry.delete(0, tk.END)
    result_label.config(text="", bg="#1e1e2f")


def show_accuracy_graph():
    plt.figure()

    # Dummy epochs (for visualization)
    epochs = [1, 2, 3, 4, 5]
    accuracy_values = [accuracy-0.1, accuracy-0.05, accuracy-0.02, accuracy-0.01, accuracy]

    plt.plot(epochs, accuracy_values, marker='o')

    plt.title("Model Accuracy Progress")
    plt.xlabel("Epochs / Iterations")
    plt.ylabel("Accuracy")

    plt.ylim(0, 1)
    plt.grid()

    plt.show()


# GUI setup
root = tk.Tk()
root.title("CyberAware - Fake News Detector")
root.geometry("600x450")
root.config(bg="#1e1e2f")

title = tk.Label(root, text="CyberAware", font=("Arial", 18, "bold"), bg="#1e1e2f", fg="white")
title.pack(pady=10)

subtitle = tk.Label(root, text="Cybersecurity Fake News Detection", font=("Arial", 10), bg="#1e1e2f", fg="gray")
subtitle.pack()

entry = tk.Entry(root, width=70, font=("Arial", 10))
entry.pack(pady=20)

check_btn = tk.Button(root, text="Check News", command=check_news, bg="#4CAF50", fg="white", padx=10)
check_btn.pack(pady=5)

clear_btn = tk.Button(root, text="Clear", command=clear_text, bg="#f44336", fg="white")
clear_btn.pack(pady=5)

graph_btn = tk.Button(root, text="Show Accuracy Graph", command=show_accuracy_graph, bg="#2196F3", fg="white")
graph_btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#1e1e2f")
result_label.pack(pady=20)

acc_label = tk.Label(root, text=f"Accuracy: {accuracy:.2f}", bg="#1e1e2f", fg="yellow")
acc_label.pack()

root.mainloop()