import os


def folder_pass(mappe):
    """ Gets the path to the folder with files as input, then goes through each folder with documents and
    extracts texts from txt files. Returns a dictionary with filenames and texts from them """
    mappe = mappe.replace('/', '//')
    names = os.listdir(mappe)
    names.sort()
    texts = {}

    for name in names:
        if not name.endswith('.DS_Store'):
            p = mappe + '/' + name
            name_files = os.listdir(p)
            for file in name_files:
                if file.endswith('.txt'):
                    path = p + '/' + file
                    with open(path, encoding='utf-8') as f:
                        text = f.read()
                    texts[file] = text
    return texts
