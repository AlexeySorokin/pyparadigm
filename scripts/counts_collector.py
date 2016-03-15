import sys
import numpy as np


class LanguageModel:

    def __init__(self, order=3, append_first=True, append_last=True):
        self.order = order
        self.root = 0
        # вершина: предок, уровень, сыновья, счётчик, вероятность, backoff
        self.trie = [[None, 0, dict(), 0, 0.0, 0.0]]
        self.append_first = append_first
        self.append_last = append_last

    def train(self, sents):
        for i, sent in enumerate(sents):
            if self.append_first:
                sent = ["<s>"] + sent
            if self.append_last:
                sent += ["</s>"]
            nodes = [self.root] + [None] * (self.order - 1)
            for word in sent:
                new_nodes = [None] * (self.order)
                prev, prev_index = self.trie[self.root], self.root
                prev[3] += 1
                for i, prev_index in enumerate(nodes):
                    child_index = prev[2].get(word)
                    if child_index is None:
                        prev[2][word] = child_index = len(self.trie)
                        self.trie.append([prev_index, i + 1, dict(), 0, 0.0, 0.0])
                    self.trie[child_index][3] += 1
                    new_nodes[i] = child_index
                    if prev_index is None:
                        break
                    prev = self.trie[prev_index]
                nodes = new_nodes
        return self

    def __str__(self):
        return self._str_from_node(self.root)

    def _str_from_node(self, index, path=[]):
        node = self.trie[index]
        answer = "{}\t{}\t{:.2f}\t{:.2f}".format(" ".join(path), node[3], node[4], node[5])
        if len(node[2]) > 0:
            answer += "\n"
            answer += "\n".join(self._str_from_node(child_index, path + [word])
                                for word, child_index in sorted(node[2].items()))
        return answer

    def make_WittenBell_smoothing(self):
        for node in self.trie:
            continuations_count = len(node[2])
            if node[1] == self.order or continuations_count == 0:
                continue
            continuations_sum = sum(self.trie[child][3] for child in node[2].values())
            denominator_log = np.log10(continuations_count + continuations_sum)
            for child in node[2].values():
                child_node = self.trie[child]
                child_node[4] = np.log10(child_node[3]) - denominator_log
            node[5] = np.log10(continuations_count) - denominator_log
        return self

    def make_arpa_file(self, outfile):
        with open(outfile, "w", encoding="utf8") as fout:
            counts = [0] * self.order
            for node in self.trie[1:]:
                counts[node[1]-1] += 1
            counts[0] += 1
            # счётчики n-грамм
            fout.write("\\data\\\n")
            fout.write("\n".join("ngram {}={}".format(i+1, count)
                                 for i, count in enumerate(counts)))
            fout.write("\n\n")
            fout.write("\\1-grams:\n{:.2f}\t<unk>\t0\n".format(self.trie[0][5]))
            stack = [(0, [])]
            node_data = [[] for _ in range(self.order)]
            while(len(stack) > 0):
                index, key = stack.pop()
                node = self.trie[index]
                for word, child_index in node[2].items():
                    stack.append((child_index, key+[word]))
                if node[1] > 0:
                    if key != ['<s>']:
                        node_data[node[1] - 1].append((key, node[4], node[5]))
                    else:
                        node_data[0] = [(key, node[4], node[5])] + node_data[0]
            for key, prob, backoff in node_data[0]:
                fout.write("{:.6f}\t{}\t{:.6f}\n".format(prob, key[0], backoff))
            for i, curr_node_data in enumerate(node_data[1:-1], 1):
                fout.write("\n\n")
                fout.write("\\{}-grams:\n".format(i+1))
                for key, prob, backoff in curr_node_data:
                    fout.write("{:.6f}\t{}\t{:.6f}\n".format(prob, " ".join(key), backoff))
            fout.write("\\{}-grams:\n".format(self.order))
            for key, prob, backoff in node_data[-1]:
                fout.write("{:.6f}\t{}\n".format(prob, " ".join(key)))
            fout.write("\n\\end\\\n")


def read_data(infile):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for i, line in enumerate(fin):
            if i % 500000 == 0:
                print(i)
            line = line.strip()
            if line == "":
                continue
            answer.append(line.split())
    return answer


if __name__ == "__main__":
    args = sys.argv[1:]
    infile, order, outfile = args
    order = int(order)
    language_model = LanguageModel(order)
    data = read_data(infile)
    language_model.train(data)
    language_model.make_WittenBell_smoothing()
    # with open(outfile, "w", encoding="utf8") as fout:
    #     fout.write(str(language_model))
    language_model.make_arpa_file(outfile)


