import os
from pathlib import Path
folder = Path("../Pictures/")

def Pictures():
    myList = os.listdir(folder)
    return myList

