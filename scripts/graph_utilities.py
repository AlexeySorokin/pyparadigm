class TopologicalSorter:
    """
    Класс для выполнения топологической сортировки вершин графа
    """

    def __init__(self, graph):
        self.vertexes = graph.vertexes
        self.transitions = graph.transitions

    def component_topological_sort(self, source):
        """
        Топологическая сортировка связной компоненты графа self.graph,
        начиная с вершины source

        Возвращает:
        -----------
        order: list or None
            список вершин в порядке топологической сортировки
            или None, если такая сортировка невозможна
        """
        if source not in self.vertexes:
            return None
        color = {x: 'white' for x in self.vertexes}
        stack, process_flags_stack = [source], [False]
        order = []
        while len(stack) > 0:
            current, current_flag = stack[-1], process_flags_stack[-1]
            if current_flag:  # выход из рекурсии для вершины
                stack.pop()
                process_flags_stack.pop()
                color[current] = 'black'
                order = [current] + order
            else:
                if color[current] == 'grey':
                    # граф имеет циклы, сортировка невозможна
                    return None
                elif color[current] == 'white':
                    color[current] = 'grey'
                    process_flags_stack[-1] = True
                    for other in self.transitions[current]:
                        stack.append(other)
                        process_flags_stack.append(False)
                elif color[current] == 'black':
                    stack.pop()
                    process_flags_stack.pop()
        return order


class Graph:
    """
    Представление графа в виде списка вершин
    """

    def __init__(self, transitions=None):
        """
        Атрибуты:
        ---------
        transitions: list or None(default=None) --- список рёбер графа
        """
        if transitions is None:
            transitions = []
        try:
            transitions = list(transitions)
        except:
            raise TypeError("Transitions must be a list or None")
        self.vertexes = set()
        self.transitions = dict()
        for first, second in transitions:
            if first in self.vertexes:
                self.transitions[first].add(second)
            else:
                self.vertexes.add(first)
                self.transitions[first] = {second}
            if second not in self.vertexes:
                self.vertexes.add(second)
                self.transitions[second] = set()

    def find_longest_paths(self, source):
        """
        Находит самые длинные пути в ациклическом графе,
        начинающиеся в вершине source

        Аргументы:
        ----------
        source, object --- стартовая вершина

        Возвращает:
        -----------
        answer, list of lists
            список путей, представленных как список состояний
        """
        if not hasattr(self, "topological_order_"):
            self.topological_order_ = self.get_topological_sort(source)
        if self.topological_order_ is None:
            return []
        longest_path_lengths = {v: -1 for v in self.vertexes}
        longest_path_lengths[source] = 0
        max_length, furthest_vertexes = 0, {0}
        predecessors_in_longest_paths = {v: set() for v in self.vertexes}
        for first in self.topological_order_:
            length_to_source = longest_path_lengths[first]
            for second in self.transitions[first]:
                length_to_second = length_to_source + 1
                # обновляем максимальное расстояние
                if length_to_second > longest_path_lengths[second]:
                    longest_path_lengths[second] = length_to_second
                    predecessors_in_longest_paths[second] = {first}
                    # обновляем текущий список наиболее удалённых вершин
                    if length_to_second > max_length:
                        max_length = length_to_second
                        furthest_vertexes = {second}
                    elif length_to_second == max_length:
                        furthest_vertexes.add(second)
                elif length_to_second == longest_path_lengths[second]:
                    predecessors_in_longest_paths[second].add(first)
        return _backtrace_longest_paths(furthest_vertexes, source, max_length,
                                        predecessors_in_longest_paths)

    def get_topological_sort(self, source):
        """
        Находит топологическую сортировку вершин, начиная с заданной
        """
        topological_sorter = TopologicalSorter(self)
        return topological_sorter.component_topological_sort(source)

    def find_maximal_paths(self, source):
        """
        Находит такие пути в ациклическом графе,
        начинающиеся в вершине source, что их нельзя продолжить дальше

        Аргументы:
        ----------
        source, object --- стартовая вершина

        Возвращает:
        -----------
        answer, list of lists
            список путей, представленных как список состояний,
            в порядке убывания их длины
            в порядке убывания их длины
        """
        if not hasattr(self, "_reverse_transitions_"):
            self._make_reverse_transitions()
        stock_vertexes = [v for v in self.vertexes
                          if (v not in self.transitions
                              or len(self.transitions[v]) == 0)]
        paths = [[v] for v in stock_vertexes]
        answer = []
        for path in paths:
            v = path[-1]
            if v == source:
                answer.append(path[::-1])
                continue
            for other in self._reverse_transitions_[v]:
                paths.append(path + [other])
        return sorted(answer, key=len, reverse=True)

    def _make_reverse_transitions(self):
        """
        Строит граф с обратными рёбрами
        """
        self._reverse_transitions_ = dict()
        for v in self.vertexes:
            self._reverse_transitions_[v] = set()
        for first, first_transitions in self.transitions.items():
            for second in first_transitions:
                self._reverse_transitions_[second].add(first)
        return


def _backtrace_longest_paths(finals, source, length, predecessors):
    """
    Восстанавливает самые длинные пути с помощью обратных ссылок
    """
    longest_paths_stack = []
    answer = []
    for vertex in finals:
        longest_paths_stack.append((vertex, length, [vertex]))
    for curr_vertex, curr_length, curr_path in longest_paths_stack:
        if curr_length == 0:
            if curr_vertex == source:
                answer.append(curr_path[::-1])
        else:
            for next_vertex in predecessors[curr_vertex]:
                longest_paths_stack.append((next_vertex, curr_length - 1,
                                            curr_path + [next_vertex]))
    return answer


def test():
    """
    Тесты
    """
    transitions = [(0, 1), (0, 3), (1, 2), (3, 2), (2, 4), (3, 4)]
    graph = Graph(transitions)
    print(graph.find_longest_paths(0))


if __name__ == "__main__":
    test()
