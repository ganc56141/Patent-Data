import os, sys

filepath = os.path.dirname(os.path.realpath(__file__))          # NOT os.getcwd() <——> this incantation is faulty

def request_decorator(interface):
    def inner(*args, **kwargs):     # must have inner function to take and transfer the proper arguments
        print("\n----------------------------------")
        print("         Start of Request\n")
        
        interface(*args, **kwargs)

        print("\n         End Of Request")
        print("----------------------------------\n")
    return inner


# ====================================
#           Utility Methods
# ====================================

def creat_dir(folder_name: str, path: str = None) -> str:
    """helper function that creates directory if not yet exists

        Args:
            folder_name (str): name of new subdirectory

        Returns:
            str: complete path to subdirectory
            
    """    
    
    dir_path = filepath if path==None else path
    path = "{directory}/{subdirectory}".format(directory = dir_path, subdirectory=folder_name)
    mode = 0o755
    try:  
        os.makedirs(path, mode)
    except OSError:
        pass
    return path


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def next_available_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b