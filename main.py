import os
from tkinter import Tk, Entry, Button, Label, Toplevel, mainloop
from tkinter.filedialog import askopenfile
from funcs import (
    extract_citations,
    download_and_retrieve,
    n_generate_summaries,
    add_annotations,
    extract_citation_context,
    extract_citation_locations,
)

    
def main(pdf_path):

    cites = extract_citations(pdf_path)

    # test with one
    cite, title = list(cites.items())[0]
    cites = {cite: title}

    embeds, texts = download_and_retrieve(cites)

    ctx = extract_citation_context(pdf_path, cites)

    locs = extract_citation_locations(pdf_path, cites)

    sums = n_generate_summaries(ctx, embeds, texts)

    final = add_annotations(pdf_path, cites, locs, sums)

    print(f'Annotated pdf can be found at "{final}"')


root = Tk()
root.geometry("200x100")
root.title("Contextify")


# This function will be used to open
# file in read mode and only Python files
# will be opened
def open_file():
    file = askopenfile(mode="r", filetypes=[("PDF Files", "*.pdf")])
    if file is not None:
        try:
            os.environ["BING_API_KEY"] = serp.get()
            os.environ["GROQ_API_KEY"] = groq.get()
            main(file.name)
        except Exception as e:
            top = Toplevel(root)
            top.geometry("750x250")
            top.title("Error")
            Label(top, text=str(e), font=("Mistral 18 bold")).place(x=150, y=80)


Label(root, text="BING API key").grid(row=0)
Label(root, text="Groq API key").grid(row=1)

serp = Entry(root)
groq = Entry(root)

serp.grid(row=0, column=1)
groq.grid(row=1, column=1)

btn = Button(root, text="Select File", command=lambda: open_file())

btn.grid(row=2, column=0)

mainloop()
