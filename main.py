from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
from funcs import *


def main(pdf_path):
    folder_path = create_folder(pdf_path)

    cites = extract_citations(pdf_path)

    #test with one
    cite, title = list(cites.items())[0]
    cites = {cite:title}

    download_all_documents(folder_path, cites)

    locs = extract_citation_locations(pdf_path, cites)

    ctx = extract_citation_context(pdf_path, cites)

    sums = generate_summaries(folder_path, ctx)

    final = add_annotations(pdf_path, cites, locs, sums)

    print(f"Annotated pdf can be found at \"{final}\"")






root = Tk()
root.geometry('200x100')
 
# This function will be used to open
# file in read mode and only Python files
# will be opened
def open_file():
    file = askopenfile(mode ='r', filetypes =[('PDF Files', '*.pdf')])
    if file is not None:
        try:
            main(file)
        except Exception as e:
            print(e)
 
btn = Button(root, text ='Open', command = lambda:open_file())
btn.pack(side = TOP, pady = 10)
 
mainloop()
