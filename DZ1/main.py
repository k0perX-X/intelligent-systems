import tkinter as tk
from tkinter import Canvas, PhotoImage
import tkinter.ttk as ttk
import barley_break
import numpy as np
from typing import Tuple


class App(tk.Tk):
    def __init__(self, values: Tuple[barley_break.Branch, int, barley_break.Branch]):
        super().__init__()
        self.title("Steps treeview")
        root = values[0]
        self.branch = values[2]
        self.nodes = {}
        self.branches = {}
        self._img = PhotoImage(file="galochka_n33co3rs23b1_16.png")
        self.tree = ttk.Treeview(self, show="tree")
        ysb = ttk.Scrollbar(self, orient=tk.VERTICAL,
                            command=self.tree.yview)
        xsb = ttk.Scrollbar(self, orient=tk.HORIZONTAL,
                            command=self.tree.xview)
        self.tree.configure(yscroll=ysb.set, xscroll=xsb.set)

        self.tree.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        ysb.grid(row=0, column=1, sticky=tk.N + tk.S)
        xsb.grid(row=1, column=0, sticky=tk.E + tk.W)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.tree.bind("<<TreeviewOpen>>", self.open_node)
        node = self.tree.insert("", tk.END, text='start', open=True)
        self.nodes[node] = root
        self.branches[node] = root
        self.populate_node(node, root)
        self.tree.bind("<Double-1>", self.OnDoubleClick)

    def OnDoubleClick(self, event):
        item = self.tree.selection()[0]
        root = self.branches[item]
        if root:
            view_matrix = ViewMatrix(root)
            view_matrix.mainloop()

    def populate_node(self, parent, root):
        for k, v in root.tree.items():
            if all(map(lambda x: x[0] == x[1],
                       [(v.directions[i], self.branch.directions[i]) for i in range(1, len(v.directions))])):
                node = self.tree.insert(parent, tk.END, text=k, open=False, image=self._img)
            else:
                node = self.tree.insert(parent, tk.END, text=k, open=False)
            self.branches[node] = v
            if len(v.tree) > 0:
                self.nodes[node] = v
                self.tree.insert(node, tk.END)

    def open_node(self, event):
        item = self.tree.focus()
        root = self.nodes.pop(item, False)
        if root:
            children = self.tree.get_children(item)
            self.tree.delete(children)
            self.populate_node(item, root)


class ViewMatrix(tk.Tk):
    def __init__(self, root: barley_break.Branch):
        super().__init__()
        self.title(">".join(map(str, ["start"] + root.directions[1:])))
        h, w = root.matrix.shape
        SQUARE_SIZE = 80
        self.geometry(f"{w * SQUARE_SIZE}x{h * SQUARE_SIZE}")
        self.resizable(False, False)
        c = Canvas(self, width=w * SQUARE_SIZE,
                   height=h * SQUARE_SIZE,
                   bg='#808080')
        c.pack()
        c.delete('all')
        for i in range(h):
            for j in range(w):
                index = str(root.matrix[i, j])
                if index != str(0):
                    c.create_rectangle(j * SQUARE_SIZE, i * SQUARE_SIZE,
                                       j * SQUARE_SIZE + SQUARE_SIZE,
                                       i * SQUARE_SIZE + SQUARE_SIZE,
                                       fill='#43ABC9',
                                       outline='#FFFFFF')
                    c.create_text(j * SQUARE_SIZE + SQUARE_SIZE / 2,
                                  i * SQUARE_SIZE + SQUARE_SIZE / 2,
                                  text=index,
                                  font="Arial {} italic".format(int(SQUARE_SIZE / 4)),
                                  fill='#FFFFFF')


if __name__ == "__main__":
    # h, w = int(input('height: ')), int(input('weight: '))
    # first = np.matrix(np.matrix(list(range(1, h * w)) + [0]).reshape(h, w))
    # barley_break.shuffle(first)
    first = np.matrix("2, 4, 3; 1, 8, 5; 7, 0, 6")
    values = barley_break.get_tree(first)
    r, steps, b = values
    print(b.directions[1:])
    print(len(b.directions) - 1)
    print(steps)
    app = App(values)
    app.mainloop()
