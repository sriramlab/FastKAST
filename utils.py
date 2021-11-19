import os.path
from joblib import dump, load
import glob
def dumpfile(data, path, filename, overwrite=False):
    isdir = os.path.isdir(path)
    if not isdir:
        os.makedirs(path)
    filepath = f'{path}{filename}'
    isfile = os.path.isfile(filepath)
    if isfile:
        if overwrite:
            print(f'overwrite the existing file')
            dump(data, filepath)
        else:
            print(f'file exists, please enforce overwrite to overwriet it')
    else:
        dump(data, filepath)
        
def readfiles(rootPath, patternFile):
    traits = []
    for name in glob.glob(f'{rootPath}{patternFile}'):
        traits.append(name)
    return traits
    
def fileExist(path, filename):
    isdir = os.path.isdir(path)
    if not isdir:
        return False
    filepath = f'{path}{filename}'
    isfile = os.path.isfile(filepath)
    if not isfile:
        return False
    return True



if __name__ == "__main__":
    path = './test/'
    data = ['a']
    filename = 'testfile.pkl'
    dumpfile(data,path,filename,overwrite=True)
