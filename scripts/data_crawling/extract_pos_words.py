import sys
import os
from collections import defaultdict

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2:
        sys.exit('Pass infile and output folder')
    infile, outdir = args
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(infile, "r", encoding="utf8") as fin:
        fin.readline()
        words_by_pos = defaultdict(list)
        for line in fin:
            line = line.strip()
            splitted = line.split()
            if len(splitted) != 6:
                continue
            word, pos, count = splitted[:3]
            count = float(count)
            words_by_pos[pos].append((word, count))
        for pos, pos_words in words_by_pos.items():
            outfile = os.path.join(outdir, "rnc_{}.out".format(pos))
            with open(outfile, "w", encoding="utf8") as fout:
                for word, count in sorted(pos_words, key=(lambda x:x[1]),
                                          reverse=True):
                    fout.write("{}\t{:.1f}\n".format(word, count))




