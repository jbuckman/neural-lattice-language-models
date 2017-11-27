import argparse, os, sys, tarfile
from urllib2 import urlopen

parser = argparse.ArgumentParser(description='Downloads and pre-processes data for Neural Lattice Language Models.')
parser.add_argument('--language', choices={"english", "chinese"},
                    help='Choose what language to download')
parser.add_argument('--max_vocab_size', default=10000, type=int,
                    help='Number of words in vocabulary')
parser.add_argument('--download', action="store_true",
                    help='Re-download data')

args = parser.parse_args()

## set up directories
if args.language == "english":
    if not os.path.exists("en"): os.makedirs("en")
    PATH = "en/"
    DATA_URL = "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
else:
    if not os.path.exists("zh"): os.makedirs("zh")
    PATH = "zh/"
    DATA_URL = None

if not os.path.exists(PATH+"raw"): os.makedirs(PATH+"raw")
if not os.path.exists(PATH+"train"): os.makedirs(PATH+"train")
if not os.path.exists(PATH+"valid"): os.makedirs(PATH+"valid")
if not os.path.exists(PATH+"test"): os.makedirs(PATH+"test")
if not os.path.exists(PATH+"vocabularies"): os.makedirs(PATH+"vocabularies")

## download raw
if args.download:
    if args.language == "chinese": raise Exception("""Sorry - this data is distributed by LDC.
                                                    Please follow the instructions in GET_CHINESE.txt instead!""")

    ### helper functions to download files (https://stackoverflow.com/questions/2028517/python-urllib2-progress-hook)
    def chunk_report(bytes_so_far, chunk_size, total_size):
        percent = float(bytes_so_far) / total_size
        percent = round(percent*100, 2)
        sys.stdout.write("Downloaded %d of %d bytes (%0.2f%%)\r" %
                         (bytes_so_far, total_size, percent))

        if bytes_so_far >= total_size:
            sys.stdout.write('\n')
    def chunk_read(response, target_file, chunk_size=8192, report_hook=None):
        total_size = response.info().getheader('Content-Length').strip()
        total_size = int(total_size)
        bytes_so_far = 0

        while 1:
            chunk = response.read(chunk_size)
            bytes_so_far += len(chunk)

            if not chunk:
                break

            target_file.write(chunk)

            if report_hook:
                report_hook(bytes_so_far, chunk_size, total_size)

        return bytes_so_far
    ###

    RAW_DL_PATH = PATH+"raw/raw_download.tar.gz"
    print "Downloading from %s to %s..." % (DATA_URL, RAW_DL_PATH)
    response = urlopen(DATA_URL)
    with open(RAW_DL_PATH, 'wb') as f:
        chunk_read(response, f, report_hook=chunk_report)

    print "Extracting...",
    tar = tarfile.open(RAW_DL_PATH, "r:gz")
    tar.extractall(path=PATH+'raw')
    tar.close()
    print "done."