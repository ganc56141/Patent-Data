import os, sys, time
import threading

class Dots:     # for user entertainment
        def __init__(self, num_dots=5):
                self.num_dots = num_dots
                self.done_flag = 0
                
        # __call__ is not working with multithreading, needs fixing
        def __call__(self, status):
                if status == 0: self.start()
                elif status == 1: self.stop()
                else: print("Error: Invalid Dot Animation State", flush=True)
        
        def __enter__(self):
            self.start()
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stop()
        
        def start(self):
                def begin_loading(num_dots):
                        self.done_flag = 0   # begins the animation
                        while True:
                                for i in range(num_dots+1):
                                        if self.done_flag:
                                                sys.stdout.write('\r' + "." * num_dots + "Done\n")       # draw all dots + "Done"
                                                sys.stdout.flush()
                                                return
                                        x = i % (num_dots+1)
                                        sys.stdout.write('\r' + "." * x )
                                        sys.stdout.write(" " * (num_dots - x))
                                        sys.stdout.flush()
                                        time.sleep(0.15)
                t1 = threading.Thread(target=begin_loading, args=[self.num_dots])
                self.t1 = t1
                t1.start()
                
        def stop(self):
                self.done_flag = 1
                self.t1.join()


def request_decorator(interface):
    def inner(*args, **kwargs):     # must have inner function to take and transfer the proper arguments
        print("\n----------------------------------")
        print("         Start of Request\n")
        
        return_value = interface(*args, **kwargs)

        print("\n         End Of Request")
        print("----------------------------------\n")
        
        return return_value
    return inner


# ====================================
#           Utility Methods
# ====================================


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        


filepath = os.path.dirname(os.path.realpath(__file__))          # NOT os.getcwd() <——> this incantation is faulty

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


def touch(path):
    if path == None: return None
    
    # create subdirectories is not exist
    basedir = os.path.dirname(path)
    if basedir == None: return None
    
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    
    # make path (or update time if exists)
    with open(path, 'a'):
        os.utime(path, None)
        
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


def format_path_for_enumeration(path_pattern):
    """Formats path for enuermated storage
    
    e.g. path_pattern = 'img/pie.jpg' -> 'img/pie-%s.jpg'

    """
    path = list(path_pattern.rpartition('.'))
    path[0] += '-%s'
    path = ''.join(map(str, path))
    return path

