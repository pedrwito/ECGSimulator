import tkinter as tk
from functools import partial









def button_function():
    print(entry.get())


window = tk.Tk()

greeting = tk.Label(text="Hello, Tkinter")
greeting.pack()



button = tk.Button(
    text="Click me!",
    width=25,
    height=5,
    bg="blue",
    fg="yellow",
    command = button_function
)
button.pack()
label = tk.Label(text="Name")
entry = tk.Entry()
label.pack()
entry.pack()

text_box = tk.Text()
text_box.pack()

window.mainloop()



