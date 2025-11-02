from graphviz import Digraph

class value:
    def __init__(self, data, children = (), op='', label =""):
        self.data = data
        self.children = set(children)
        self.op = op
        self.label = label

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((v,child))
                build(child)
    build(root)
    return nodes,edges


def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    dot.node( name  = uid, label = "{%s | data %4f}" %(n.label, n.data))
    if n.op:
      dot.node(name = uid + n.op, label = n.op)
      dot.edge(uid + n.op, uid)

  for n1, n2 in edges:
    dot.edge(str(id(n1)), str(id(n2)) + n2.op)

  return dot

# create simple graph
if __name__ == '__main__':
    a = value(3.0, label = "a")
    b = value(4.0, label = "b")
    c = value(a.data + b.data, children=(a,b), op='+', label='c')
    d = draw_dot(c)
    # try to render to svg bytes
    svg = d.pipe(format='svg')
    print('Rendered svg length:', len(svg))
