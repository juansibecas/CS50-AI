import sys
import copy

from crossword import *


def sort_by_n(word):
    return word["n"]


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        words_to_remove = set()
        for var in self.domains:
            words = self.domains[var]
            for word in words:
                if len(word) != var.length:
                    words_to_remove.add(word)
            for word in words_to_remove:
                words.remove(word)
            words_to_remove.clear()

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        words_to_remove = set()
        (x_overlap, y_overlap) = self.crossword.overlaps[x, y]  # overlap positions
        if x_overlap is None and y_overlap is None:
            return False
        x_domain = self.domains[x]
        y_domain = self.domains[y]
        for x_word in x_domain:
            remove = True
            x_overlap_char = x_word[x_overlap]
            for y_word in y_domain:                 # doesnt remove word in x if theres at least 1 word in y
                y_overlap_char = y_word[y_overlap]  # which has the same char at overlap position
                if x_overlap_char == y_overlap_char:
                    remove = False
                    break
            if remove:
                words_to_remove.add(x_word)
        revised = False
        for word in words_to_remove:
            self.domains[x].remove(word)
            revised = True

        if revised:
            return True
        else:
            return False

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            queue = [(i, j) for i in self.crossword.variables for j in self.crossword.neighbors(i)]
        else:
            queue = arcs

        while queue:
            arc = queue.pop(0)
            x = arc[0]
            y = arc[1]
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                y2 = set()
                y2.add(y)
                for z in self.crossword.neighbors(x) - y2:
                    queue.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for var in self.crossword.variables:
            if var not in assignment.keys():
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        values = set(assignment.values())           # set should delete duplicates
        if len(values) < len(assignment.keys()):    # returns if theres less values than vars (repeated value)
            return False
        for (var1, word1) in assignment.items():
            if len(word1) != var1.length:           # returns if a var is assigned a word not matching length
                return False
            for var2 in self.crossword.neighbors(var1):
                if var2 in assignment.keys():
                    word2 = assignment[var2]
                    (var1_overlap, var2_overlap) = self.crossword.overlaps[var1, var2]
                    if word1[var1_overlap] != word2[var2_overlap]:
                        return False                # returns if the overlap char doesnt match
        return True  # all good

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        domain = self.domains[var]
        neighbors = [i for i in self.crossword.neighbors(var) if i not in assignment.keys()]  # neighbors that havent
        words_deleted = []                                                                    # been assigned yet
        for word1 in domain:
            n = 0       # counts words ruled out because of word1
            for var2 in neighbors:
                (var1_overlap, var2_overlap) = self.crossword.overlaps[var, var2]
                for word2 in self.domains[var2]:
                    if word1[var1_overlap] != word2[var2_overlap]:
                        n += 1
            words_deleted.append({"word": word1, "n": n})
        words_deleted.sort(reverse=False, key=sort_by_n)

        ordered_domain = []
        for word in words_deleted:
            ordered_domain.append(word["word"])

        return ordered_domain

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned_vars = set()
        for var in self.crossword.variables:
            if var not in assignment.keys():
                unassigned_vars.add(var)
        domain_len = 1e10
        for var in unassigned_vars:
            if domain_len > len(self.domains[var]):
                var_to_return = var
            elif domain_len == len(self.domains[var]):
                if len(self.crossword.neighbors(var_to_return)) < len(self.crossword.neighbors(var)):
                    var_to_return = var

        return var_to_return

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            assignment[var] = value
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result is not None:
                    return result
                else:
                    assignment.pop(var)
            else:
                assignment.pop(var)
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
