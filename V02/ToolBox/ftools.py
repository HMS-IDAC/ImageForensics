from os.path import *
from os import listdir, makedirs, remove
import pickle
import shutil

def fileparts(path): # path = file path
    [p,f] = split(path)
    [n,e] = splitext(f)
    return [p,n,e]

def listfiles(path,token): # path = folder path
    l = []
    for f in listdir(path):
        fullPath = join(path,f)
        if isfile(fullPath) and token in f:
            l.append(fullPath)
    l.sort()
    return l

def listsubdirs(path): # path = folder path
    l = []
    for f in listdir(path):
        fullPath = join(path,f)
        if isdir(fullPath):
            l.append(fullPath)
    l.sort()
    return l

def pathjoin(p,ne): # '/path/to/folder', 'name.extension' (or a subfolder)
    return join(p,ne)

def saveData(data,path,verbose=False):
    if verbose:
        print('saving data')
    dataFile = open(path, 'wb')
    pickle.dump(data, dataFile)

def loadData(path,verbose=False):
    if verbose:
        print('loading data')
    dataFile = open(path, 'rb')
    return pickle.load(dataFile)

def createFolderIfNonExistent(path):
    if not exists(path): # from os.path
        makedirs(path)

def removeFolderIfExistent(path):
    if exists(path):
        shutil.rmtree(path)

def moveFile(fullPathSource,folderPathDestination):
    [p,n,e] = fileparts(fullPathSource)
    shutil.move(fullPathSource,pathjoin(folderPathDestination,n+e))

def copyFile(fullPathSource,folderPathDestination):
    [p,n,e] = fileparts(fullPathSource)
    shutil.copy(fullPathSource,pathjoin(folderPathDestination,n+e))

def removeFile(path):
    remove(path)