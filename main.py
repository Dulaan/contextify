# main.py

import os
from tkinter import Tk, Entry, Button, Label, Toplevel, Text, Scrollbar, END
from tkinter.filedialog import askopenfile
from funcs import (
    extract_citations,
    download_and_retrieve,
    n_generate_summaries,
    add_annotations,
    extract_citation_context,
    extract_citation_locations,
)
import threading

def main(pdf_path, bing_key, groq_key, output_callback, error_callback):
    try:
        os.environ["BING_API_KEY"] = bing_key
        os.environ["GROQ_API_KEY"] = groq_key

        cites = extract_citations(pdf_path)

        if not cites:
            raise ValueError("No citations found in the PDF.")

        embeds, texts = download_and_retrieve(cites)

        ctx = extract_citation_context(pdf_path, cites)

        locs = extract_citation_locations(pdf_path, cites)

        sums = n_generate_summaries(ctx, embeds, texts)
        print(sums)
        final = add_annotations(pdf_path, cites, locs, sums)

        if final:
            output_callback(f'Annotated PDF saved as "{final}"')
        else:
            raise ValueError("Failed to create annotated PDF.")
    except Exception as e:
        error_callback(str(e))

def open_file(root, output_text, error_text):
    file = askopenfile(mode="rb", filetypes=[("PDF Files", "*.pdf")])
    if file is not None:
        pdf_path = file.name

        bing_key = serp.get()
        groq_key = groq.get()

        if not bing_key or not groq_key:
            error_text.insert(END, "API keys are required.\n")
            return

        # Clear previous outputs
        output_text.delete(1.0, END)
        error_text.delete(1.0, END)

        # Run the main process in a separate thread to keep the GUI responsive
        thread = threading.Thread(target=main, args=(pdf_path, bing_key, groq_key, 
                                                    lambda msg: output_text.insert(END, msg + "\n"),
                                                    lambda err: error_text.insert(END, err + "\n")))
        thread.start()

def create_gui():
    root = Tk()
    root.geometry("500x400")
    root.title("Contextify MVP")

    # Labels
    Label(root, text="BING API Key").grid(row=0, column=0, padx=10, pady=10, sticky='e')
    Label(root, text="Groq API Key").grid(row=1, column=0, padx=10, pady=10, sticky='e')

    # Entry fields
    global serp, groq
    serp = Entry(root, width=50, show="*")  # Hide API key input
    groq = Entry(root, width=50, show="*")

    serp.grid(row=0, column=1, padx=10, pady=10)
    groq.grid(row=1, column=1, padx=10, pady=10)

    # Buttons
    btn = Button(root, text="Select PDF File", command=lambda: open_file(root, output_text, error_text))
    btn.grid(row=2, column=0, columnspan=2, pady=10)

    # Output Text
    Label(root, text="Output:").grid(row=3, column=0, padx=10, pady=10, sticky='nw')
    output_text = Text(root, height=5, width=60)
    output_text.grid(row=4, column=0, columnspan=2, padx=10)
    output_scroll = Scrollbar(root, command=output_text.yview)
    output_scroll.grid(row=4, column=2, sticky='nsew')
    output_text.config(yscrollcommand=output_scroll.set)

    # Error Text
    Label(root, text="Errors:").grid(row=5, column=0, padx=10, pady=10, sticky='nw')
    error_text = Text(root, height=5, width=60, fg="red")
    error_text.grid(row=6, column=0, columnspan=2, padx=10)
    error_scroll = Scrollbar(root, command=error_text.yview)
    error_scroll.grid(row=6, column=2, sticky='nsew')
    error_text.config(yscrollcommand=error_scroll.set)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
