import os
from PIL import Image

base_dir = '/home/student/05/b05902016/pythonFL/data/'
parts = ['train/', 'test/']
categories = ['hentai/', 'porn/', 'neutral/', 'drawings/', 'sexy/']

for part in parts:
    for cate in categories:
        DIR = base_dir + part + cate
        print("Checking: ", DIR);
        for filename in os.listdir(DIR):
            if filename.endswith(".jpg"):
                filepath = os.path.join(DIR, filename)
                try:
                    img = Image.open(filepath)
                    # Possible EXIF error
                    img._getexif()
                    # Possible decompression bomb error
                    if img.size[0] * img.size[1] > 89478485:
                        print(str(filepath) + " is too big")
                        os.remove(filepath)
                    # Possible truncated file
                    img.load()
                except OSError:
                    print(str(filepath) + " is broken or truncated")
                    os.remove(filepath)
                except IndexError:
                    print(str(filepath) + " EXIF data is broken")
                    os.remove(filepath)
                except AttributeError:
                    print(str(filepath) + " has no EXIF data")
                    os.remove(filepath)
                except IOError:
                    print(str(filepath) + " has corrupted EXIF data")
                    os.remove(filepath)
                except SyntaxError:
                    print(str(filepath) + " not a TIFF file")
                    os.remove(filepath)
                img.close()
