import tkinter as tk
from tkinter import messagebox, ttk
import os
import re
import graphviz

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ============================================================
#                     CONSTANTES
# ============================================================
TOKEN_CARACTER = "CARACTER"
TOKEN_PUNTO = "PUNTO - CARACTER_GLOBAL"
TOKEN_ASTERISCO = "ASTERISCO - OPERADOR_REPETICION_CERO_MUCHOS"
TOKEN_MAS = "MAS - OPERADOR_REPETICION_UNO_MUCHOS"
TOKEN_OPCIONAL = "OPERADOR_CONDICION_OPCIONAL"
TOKEN_O = "O - SIMBOLO_OR"
TOKEN_PARIZQ = "SIMBOLO_APERTURA_AGRUPACION"
TOKEN_PARDER = "SIMBOLO_CIERRE_AGRUPACION"
TOKEN_CORCHIZQ = "SIMBOLO_APERTURA_CONJUNTO"
TOKEN_CORCHDER = "SIMBOLO_CIERRE_CONJUNTO"
TOKEN_CIRCUNFLEJO = "OPERADOR_NEGACION"
TOKEN_PESOS = "SIMBOLO_FIN_CADENA"
TOKEN_ESCAPAR = "SIMBOLO_ESCAPE_LITERAL"
TOKEN_RANGO = "OPERADOR_RANGO"
TOKEN_COMA = "COMA"
TOKEN_DOSPUNTOS = "DOSPUNTOS"
TOKEN_FIN = "FIN"
TOKEN_CUANT_FIJO = "CUANT_FIJO"
TOKEN_CUANT_MIN = "CUANT_MIN"
TOKEN_CUANT_RANGO = "CUANT_RANGO"
EPSILON = "ε"

# ============================================================
#                     LEXER - ANALIZADOR LEXICO
# ============================================================
class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos] if self.text else None
    def advance(self):
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None
    def peek(self):
        if self.pos + 1 < len(self.text):
            return self.text[self.pos + 1]
        return None
    def generate_tokens(self):
        tokens = []
        while self.current_char is not None:
            c = self.current_char
            if c.isspace():
                self.advance()
                continue
            if c == '\\':
                next_char = self.peek()
                if next_char is None:
                    raise Exception("Escape incompleto.")
                self.advance()
                escaped = self.current_char
                self.advance()
                tokens.append((TOKEN_ESCAPAR, f"\\{escaped}"))
                continue
            if c == '[':
                tokens.append((TOKEN_CORCHIZQ, c))
                self.advance()
                continue
            if c == ']':
                tokens.append((TOKEN_CORCHDER, c))
                self.advance()
                continue
            if c == '{':
                brace_content = self._read_brace_content()
                pattern_range = r'^\s*(\d+)\s*,\s*(\d+)\s*$'
                pattern_min = r'^\s*(\d+)\s*,\s*\}?$'
                pattern_fixed = r'^\s*(\d+)\s*$'
                if re.match(pattern_range, brace_content):
                    tokens.append((TOKEN_CUANT_RANGO, brace_content))
                elif re.match(pattern_min, brace_content):
                    tokens.append((TOKEN_CUANT_MIN, brace_content))
                elif re.match(pattern_fixed, brace_content):
                    tokens.append((TOKEN_CUANT_FIJO, brace_content))
                else:
                    raise Exception(f"Cuantificador inválido: {{ {brace_content} }}")
                continue
            if c == '.':
                tokens.append((TOKEN_PUNTO, c))
                self.advance()
                continue
            if c == '*':
                tokens.append((TOKEN_ASTERISCO, c))
                self.advance()
                continue
            if c == '+':
                tokens.append((TOKEN_MAS, c))
                self.advance()
                continue
            if c == '?':
                tokens.append((TOKEN_OPCIONAL, c))
                self.advance()
                continue
            if c == '^':
                tokens.append((TOKEN_CIRCUNFLEJO, c))
                self.advance()
                continue
            if c == '$':
                tokens.append((TOKEN_PESOS, c))
                self.advance()
                continue
            if c == '(':
                tokens.append((TOKEN_PARIZQ, c))
                self.advance()
                continue
            if c == ')':
                tokens.append((TOKEN_PARDER, c))
                self.advance()
                continue
            if c == '|':
                tokens.append((TOKEN_O, c))
                self.advance()
                continue
            if c == ',':
                tokens.append((TOKEN_COMA, c))
                self.advance()
                continue
            if c == ':':
                tokens.append((TOKEN_DOSPUNTOS, c))
                self.advance()
                continue
            if c == '-':
                tokens.append((TOKEN_RANGO, c))
                self.advance()
                continue
            tokens.append((TOKEN_CARACTER, c))
            self.advance()
        tokens.append((TOKEN_FIN, None))
        return tokens
    def _read_brace_content(self):
        self.advance()
        start_pos = self.pos
        while self.current_char is not None and self.current_char != '}':
            self.advance()
        if self.current_char is None:
            raise Exception("Falta '}'.")
        end_pos = self.pos
        content = self.text[start_pos:end_pos]
        self.advance()
        return content

# ============================================================
#                     PARSER - ANALIZADOR SINTACTICO
# ============================================================
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]
    def error(self, msg="Error de sintaxis"):
        raise Exception(msg)
    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = (TOKEN_FIN, None)
    def parse(self):
        node = self.expr()
        if self.current_token[0] != TOKEN_FIN:
            self.error("Quedan tokens sobrantes.")
        return node
    def expr(self):
        left_node = self.term()
        while self.current_token[0] == TOKEN_O:
            self.advance()
            right_node = self.term()
            left_node = ("OR", left_node, right_node)
        return left_node
    def term(self):
        node = self.factor()
        while self.current_token[0] not in (TOKEN_O, TOKEN_PARDER, TOKEN_FIN):
            node2 = self.factor()
            node = ("CONCAT", node, node2)
        return node
    def factor(self):
        node = self.base()
        quant_tokens = (
            TOKEN_ASTERISCO, TOKEN_MAS, TOKEN_OPCIONAL,
            TOKEN_CUANT_FIJO, TOKEN_CUANT_MIN, TOKEN_CUANT_RANGO
        )
        while self.current_token[0] in quant_tokens:
            tk = self.current_token
            self.advance()
            if tk[0] == TOKEN_ASTERISCO:
                node = ("STAR", node)
            elif tk[0] == TOKEN_MAS:
                node = ("PLUS", node)
            elif tk[0] == TOKEN_OPCIONAL:
                node = ("QUESTION", node)
            elif tk[0] == TOKEN_CUANT_FIJO:
                node = ("QUANT_FIXED", node, tk[1])
            elif tk[0] == TOKEN_CUANT_MIN:
                node = ("QUANT_MIN", node, tk[1])
            elif tk[0] == TOKEN_CUANT_RANGO:
                node = ("QUANT_RANGE", node, tk[1])
        return node
    def base(self):
        ttype, tval = self.current_token
        if ttype == TOKEN_PARIZQ:
            self.advance()
            if self.current_token[0] == TOKEN_OPCIONAL:
                self.advance()
                if self.current_token[0] == TOKEN_DOSPUNTOS:
                    self.advance()
                    inner = self.expr()
                    if self.current_token[0] != TOKEN_PARDER:
                        self.error("Falta ')' en (?: ... ).")
                    self.advance()
                    return ("GROUP_NONCAPTURING", inner)
                else:
                    self.error("Se esperaba ':' tras '(?'.")
            else:
                inner = self.expr()
                if self.current_token[0] != TOKEN_PARDER:
                    self.error("Falta ')' en ( ... ).")
                self.advance()
                return ("GROUP", inner)
        elif ttype == TOKEN_CORCHIZQ:
            return self._parse_char_class()
        elif ttype == TOKEN_CARACTER:
            self.advance()
            return ("CHAR", tval)
        elif ttype == TOKEN_ESCAPAR:
            self.advance()
            return ("ESCAPE", tval)
        elif ttype == TOKEN_PUNTO:
            self.advance()
            return ("DOT", ".")
        elif ttype == TOKEN_CIRCUNFLEJO:
            self.advance()
            return ("ANCHOR_START", "^")
        elif ttype == TOKEN_PESOS:
            self.advance()
            return ("ANCHOR_END", "$")
        else:
            self.error(f"Token inesperado: {ttype}, {tval}")
    def _parse_char_class(self):
        self.advance()
        negated = False
        if self.current_token[0] == TOKEN_CIRCUNFLEJO:
            negated = True
            self.advance()
        contenido = []
        while self.current_token[0] not in (TOKEN_CORCHDER, TOKEN_FIN):
            ttype, tval = self.current_token
            contenido.append((ttype, tval))
            self.advance()
        if self.current_token[0] != TOKEN_CORCHDER:
            self.error("Falta ']' en la clase.")
        self.advance()
        final_tokens = self._interpret_char_class_content(contenido)
        return ("CHAR_CLASS", negated, final_tokens)
    def _interpret_char_class_content(self, tokens_list):
        result = []
        i = 0
        while i < len(tokens_list):
            ttype, tval = tokens_list[i]
            if ttype == TOKEN_CARACTER:
                if (i + 2 < len(tokens_list) and
                    tokens_list[i+1][0] == TOKEN_RANGO and
                    tokens_list[i+2][0] == TOKEN_CARACTER):
                    start_char = tval
                    end_char = tokens_list[i+2][1]
                    result.append(("RANGE", start_char, end_char))
                    i += 3
                else:
                    result.append(("CHAR", tval))
                    i += 1
            elif ttype == TOKEN_ESCAPAR:
                result.append(("ESCAPE", tval))
                i += 1
            elif ttype == TOKEN_RANGO:
                result.append(("CHAR", tval))
                i += 1
            else:
                result.append((ttype, tval))
                i += 1
        return result

# ============================================================
#            EXPANSION DE CUANTIFICADORES
# ============================================================
def expand_ast(ast):
    nodetype = ast[0]
    if nodetype in ("CHAR", "ESCAPE", "DOT", "ANCHOR_START", "ANCHOR_END"):
        return ast
    elif nodetype == "CHAR_CLASS":
        return ast
    elif nodetype in ("GROUP", "GROUP_NONCAPTURING"):
        inside = expand_ast(ast[1])
        return (nodetype, inside)
    elif nodetype == "CONCAT":
        left_expanded = expand_ast(ast[1])
        right_expanded = expand_ast(ast[2])
        return ("CONCAT", left_expanded, right_expanded)
    elif nodetype == "OR":
        left_expanded = expand_ast(ast[1])
        right_expanded = expand_ast(ast[2])
        return ("OR", left_expanded, right_expanded)
    elif nodetype == "STAR":
        sub_expanded = expand_ast(ast[1])
        return ("STAR", sub_expanded)
    elif nodetype == "PLUS":
        sub_expanded = expand_ast(ast[1])
        return ("PLUS", sub_expanded)
    elif nodetype == "QUESTION":
        sub_expanded = expand_ast(ast[1])
        return ("QUESTION", sub_expanded)
    elif nodetype == "QUANT_FIXED":
        subnodo = expand_ast(ast[1])
        n = int(ast[2].strip())
        if n <= 0:
            return ("EPSILON_NODE",)
        tmp = subnodo
        for _ in range(n - 1):
            tmp = ("CONCAT", tmp, subnodo)
        return expand_ast(tmp)
    elif nodetype == "QUANT_MIN":
        subnodo = expand_ast(ast[1])
        n = int(ast[2].replace(",", "").strip())
        part_fixed = ("QUANT_FIXED", subnodo, str(n))
        part_star = ("STAR", subnodo)
        concat_ast = ("CONCAT", part_fixed, part_star)
        return expand_ast(concat_ast)
    elif nodetype == "QUANT_RANGE":
        subnodo = expand_ast(ast[1])
        content = ast[2].split(",")
        n = int(content[0].strip())
        m = int(content[1].strip())
        if n > m:
            raise Exception(f"Rango inválido: {n},{m}")
        base_ast = ("QUANT_FIXED", subnodo, str(n))
        current = base_ast
        for _ in range(m - n):
            current = ("CONCAT", current, ("QUESTION", subnodo))
        return expand_ast(current)
    elif nodetype == "EPSILON_NODE":
        return ("EPSILON_NODE",)
    else:
        raise Exception(f"Nodo AST no reconocido: {nodetype}")

# ============================================================
#               NFA DE THOMPSON
# ============================================================
class NFA_Thompson:
    def __init__(self):
        self.states = set()
        self.transitions = {}
        self.start = None
        self.accept = None
def new_state_id():
    new_state_id.counter += 1
    return f"S{new_state_id.counter}"
new_state_id.counter = -1
def add_transition_nfa(nfa, src, symbol, dst):
    if (src, symbol) not in nfa.transitions:
        nfa.transitions[(src, symbol)] = set()
    nfa.transitions[(src, symbol)].add(dst)
def reconstruir_char_class(negated, sublist):
    contenido = ""
    for item in sublist:
        if item[0] == "RANGE":
            contenido += f"{item[1]}-{item[2]}"
        elif item[0] == "CHAR":
            contenido += item[1]
        elif item[0] == "ESCAPE":
            contenido += item[1]
        else:
            contenido += str(item[1])
    if negated:
        return f"[^{contenido}]"
    else:
        return f"[{contenido}]"
def construir_nfa_thompson(ast):
    nodetype = ast[0]
    if nodetype == "EPSILON_NODE":
        nfa = NFA_Thompson()
        s0 = new_state_id()
        nfa.states.add(s0)
        nfa.start = s0
        nfa.accept = s0
        return nfa
    elif nodetype == "CHAR":
        nfa = NFA_Thompson()
        s0 = new_state_id()
        s1 = new_state_id()
        nfa.states.update([s0, s1])
        nfa.start = s0
        nfa.accept = s1
        add_transition_nfa(nfa, s0, ast[1], s1)
        return nfa
    elif nodetype == "ESCAPE":
        nfa = NFA_Thompson()
        s0 = new_state_id()
        s1 = new_state_id()
        nfa.states.update([s0, s1])
        nfa.start = s0
        nfa.accept = s1
        add_transition_nfa(nfa, s0, ast[1], s1)
        return nfa
    elif nodetype == "DOT":
        nfa = NFA_Thompson()
        s0 = new_state_id()
        s1 = new_state_id()
        nfa.states.update([s0, s1])
        nfa.start = s0
        nfa.accept = s1
        add_transition_nfa(nfa, s0, ".", s1)
        return nfa
    elif nodetype in ("ANCHOR_START", "ANCHOR_END"):
        nfa = NFA_Thompson()
        s0 = new_state_id()
        s1 = new_state_id()
        nfa.states.update([s0, s1])
        nfa.start = s0
        nfa.accept = s1
        add_transition_nfa(nfa, s0, ast[1], s1)
        return nfa
    elif nodetype == "CHAR_CLASS":
        nfa = NFA_Thompson()
        s0 = new_state_id()
        s1 = new_state_id()
        nfa.states.update([s0, s1])
        nfa.start = s0
        nfa.accept = s1
        label = reconstruir_char_class(ast[1], ast[2])
        add_transition_nfa(nfa, s0, label, s1)
        return nfa
    elif nodetype in ("GROUP", "GROUP_NONCAPTURING"):
        return construir_nfa_thompson(ast[1])
    elif nodetype == "CONCAT":
        nfa1 = construir_nfa_thompson(ast[1])
        nfa2 = construir_nfa_thompson(ast[2])
        nfa = NFA_Thompson()
        nfa.states = nfa1.states.union(nfa2.states)
        nfa.transitions = {}
        for k,v in nfa1.transitions.items():
            nfa.transitions[k] = v.copy()
        for k,v in nfa2.transitions.items():
            if k not in nfa.transitions:
                nfa.transitions[k] = set()
            nfa.transitions[k].update(v)
        add_transition_nfa(nfa, nfa1.accept, EPSILON, nfa2.start)
        nfa.start = nfa1.start
        nfa.accept = nfa2.accept
        return nfa
    elif nodetype == "OR":
        nfa1 = construir_nfa_thompson(ast[1])
        nfa2 = construir_nfa_thompson(ast[2])
        nfa = NFA_Thompson()
        s0 = new_state_id()
        s3 = new_state_id()
        nfa.states = nfa1.states.union(nfa2.states)
        nfa.states.update([s0, s3])
        nfa.transitions = {}
        for k,v in nfa1.transitions.items():
            nfa.transitions[k] = v.copy()
        for k,v in nfa2.transitions.items():
            if k not in nfa.transitions:
                nfa.transitions[k] = set()
            nfa.transitions[k].update(v)
        add_transition_nfa(nfa, s0, EPSILON, nfa1.start)
        add_transition_nfa(nfa, s0, EPSILON, nfa2.start)
        add_transition_nfa(nfa, nfa1.accept, EPSILON, s3)
        add_transition_nfa(nfa, nfa2.accept, EPSILON, s3)
        nfa.start = s0
        nfa.accept = s3
        return nfa
    elif nodetype == "STAR":
        subnfa = construir_nfa_thompson(ast[1])
        nfa = NFA_Thompson()
        s0 = new_state_id()
        s1 = new_state_id()
        nfa.states = subnfa.states.copy()
        nfa.states.update([s0, s1])
        nfa.transitions = {}
        for k,v in subnfa.transitions.items():
            nfa.transitions[k] = v.copy()
        add_transition_nfa(nfa, s0, EPSILON, subnfa.start)
        add_transition_nfa(nfa, subnfa.accept, EPSILON, subnfa.start)
        add_transition_nfa(nfa, s0, EPSILON, s1)
        add_transition_nfa(nfa, subnfa.accept, EPSILON, s1)
        nfa.start = s0
        nfa.accept = s1
        return nfa
    elif nodetype == "PLUS":
        subnfa = construir_nfa_thompson(ast[1])
        star_ast = ("STAR", ast[1])
        subStar = construir_nfa_thompson(star_ast)
        nfa = NFA_Thompson()
        nfa.states = subnfa.states.union(subStar.states)
        nfa.transitions = {}
        for k,v in subnfa.transitions.items():
            nfa.transitions[k] = v.copy()
        for k,v in subStar.transitions.items():
            if k not in nfa.transitions:
                nfa.transitions[k] = set()
            nfa.transitions[k].update(v)
        add_transition_nfa(nfa, subnfa.accept, EPSILON, subStar.start)
        nfa.start = subnfa.start
        nfa.accept = subStar.accept
        return nfa
    elif nodetype == "QUESTION":
        subnfa = construir_nfa_thompson(ast[1])
        eps_ast = ("EPSILON_NODE",)
        return construir_nfa_thompson(("OR", subnfa, eps_ast))
    else:
        raise Exception(f"Nodo AST no reconocido en Thompson: {nodetype}")

# ============================================================
#           NFA -> DFA (Subconjunto)
# ============================================================
class DFA:
    def __init__(self):
        self.states = []
        self.transitions = {}
        self.start_state = None
        self.accept_states = set()
def epsilon_closure(nfa, states):
    stack = list(states)
    closure = set(states)
    while stack:
        top = stack.pop()
        if (top, EPSILON) in nfa.transitions:
            for nxt in nfa.transitions[(top, EPSILON)]:
                if nxt not in closure:
                    closure.add(nxt)
                    stack.append(nxt)
    return closure
def move_nfa(nfa, states, symbol):
    reachable = set()
    for st in states:
        if (st, symbol) in nfa.transitions:
            reachable |= nfa.transitions[(st, symbol)]
    return reachable
def obtener_alfabeto(nfa):
    symbols = set()
    for (origin, sym) in nfa.transitions.keys():
        if sym != EPSILON:
            symbols.add(sym)
    return symbols
def convert_nfa_to_dfa(nfa):
    dfa = DFA()
    start_closure = frozenset(epsilon_closure(nfa, {nfa.start}))
    dfa.start_state = start_closure
    dfa.states.append(start_closure)
    if nfa.accept in start_closure:
        dfa.accept_states.add(start_closure)
    unmarked = [start_closure]
    alphabet = obtener_alfabeto(nfa)
    while unmarked:
        s = unmarked.pop()
        for sym in alphabet:
            nxt = set()
            mov = move_nfa(nfa, s, sym)
            for st in mov:
                nxt |= epsilon_closure(nfa, {st})
            nxt_fs = frozenset(nxt)
            if not nxt_fs:
                continue
            if nxt_fs not in dfa.states:
                dfa.states.append(nxt_fs)
                unmarked.append(nxt_fs)
                if nfa.accept in nxt_fs:
                    dfa.accept_states.add(nxt_fs)
            dfa.transitions[(s, sym)] = nxt_fs
    return dfa

# ============================================================
#                 GRAFICAR DFA
# ============================================================
def graficar_dfa(dfa: DFA, filename="dfa_result"):
    dot = graphviz.Digraph(comment="AFD Resultante (Subset Construction)")
    dot.attr(rankdir='LR')
    dot.graph_attr["dpi"] = "200"
    state_map = {}
    for i, st in enumerate(dfa.states):
        state_map[st] = f"q{i}"
    for st in dfa.states:
        name = state_map[st]
        if st == dfa.start_state and st in dfa.accept_states:
            dot.node(name, label=f"{name}\n(Start, Accept)",
                     shape="doublecircle", style="filled", fillcolor="green")
        elif st == dfa.start_state:
            dot.node(name, label=f"{name}\n(Start)",
                     shape="circle", style="filled", fillcolor="green")
        elif st in dfa.accept_states:
            dot.node(name, label=f"{name}\n(Accept)",
                     shape="doublecircle", style="filled", fillcolor="lightblue")
        else:
            dot.node(name, label=name, shape="circle")
    for (orig, sym), dest in dfa.transitions.items():
        orig_name = state_map[orig]
        dest_name = state_map[dest]
        dot.edge(orig_name, dest_name, label=str(sym))
    dot.render(filename, format="pdf", view=True, cleanup=True)
    dot.render(filename, format="png", view=False, cleanup=True)

# ============================================================
#                 INTERFAZ GRAFICA
# ============================================================
class RegexApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Compilador REGEX a AFD")
        self.master.geometry("1000x600")
        self.master.resizable(True, True)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f7f7f7")
        style.configure("TLabel", background="#f7f7f7", font=("Segoe UI", 12))
        style.configure("TEntry", font=("Segoe UI", 12))
        style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=8, relief="flat")
        style.map("TButton", background=[("active", "#0056b3")], foreground=[("active", "#ffffff")])
        style.configure("TLabelframe", background="#f7f7f7", font=("Segoe UI", 12, "bold"))
        style.configure("TLabelframe.Label", font=("Segoe UI", 12, "bold"))
        self.frame_main = ttk.Frame(master, padding=20)
        self.frame_main.pack(fill=tk.BOTH, expand=True)
        self.label = ttk.Label(self.frame_main, text="Ingrese la Expresión Regular:")
        self.label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.regex_entry = ttk.Entry(self.frame_main, width=60)
        self.regex_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.btn_frame = ttk.Frame(self.frame_main)
        self.btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        self.btn_build_dfa = ttk.Button(self.btn_frame, text="Generar AFD", command=self.build_dfa)
        self.btn_build_dfa.pack(side=tk.LEFT, padx=5)
        self.btn_clear = ttk.Button(self.btn_frame, text="Limpiar", command=self.clear_details)
        self.btn_clear.pack(side=tk.LEFT, padx=5)
        self.details_frame = ttk.Labelframe(self.frame_main, text="Detalles del Análisis", padding=10)
        self.details_frame.grid(row=2, column=0, columnspan=2, sticky=tk.NSEW, padx=5, pady=5)
        self.text_details = tk.Text(self.details_frame, height=15, wrap=tk.WORD, font=("Consolas", 11))
        self.text_details.pack(fill=tk.BOTH, expand=True)
        self.image_frame = ttk.Labelframe(self.frame_main, text="Visualización del AFD", padding=10)
        self.image_frame.grid(row=3, column=0, columnspan=2, sticky=tk.NSEW, padx=5, pady=5)
        self.canvas = tk.Canvas(self.image_frame, width=900, height=300,
                                bg="#f0f0f0", highlightthickness=1, highlightbackground="#d1d1d1")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.current_dfa = None
        self.frame_main.rowconfigure(3, weight=1)
        self.frame_main.columnconfigure(1, weight=1)
    def build_dfa(self):
        regex_text = self.regex_entry.get().strip()
        if not regex_text:
            messagebox.showerror("Error", "La expresión regular está vacía.")
            return
        try:
            lexer = Lexer(regex_text)
            tokens = lexer.generate_tokens()
            parser = Parser(tokens)
            raw_ast = parser.parse()
            expanded_ast = expand_ast(raw_ast)
            new_state_id.counter = -1
            nfa = construir_nfa_thompson(expanded_ast)
            dfa = convert_nfa_to_dfa(nfa)
            self.current_dfa = dfa
            graficar_dfa(dfa, filename="temp_dfa")
            self.text_details.delete("1.0", tk.END)
            self.text_details.insert(tk.END, "TOKENS:\n")
            for t in tokens:
                self.text_details.insert(tk.END, f"  {t}\n")
            self.text_details.insert(tk.END, "\nAST (Original):\n")
            self.text_details.insert(tk.END, str(raw_ast))
            self.text_details.insert(tk.END, "\n\nAST (Expandido):\n")
            self.text_details.insert(tk.END, str(expanded_ast))
            self.text_details.insert(tk.END, "\n\n=== AUTÓMATA FINITO NO DETERMINISTA ===\n")
            self.text_details.insert(tk.END, f"  Inicion: {nfa.start}\n")
            self.text_details.insert(tk.END, f"  Aceptacion: {nfa.accept}\n")
            self.text_details.insert(tk.END, "  Transiciones (AFN):\n")
            for (st, sym), dsts in nfa.transitions.items():
                for d in dsts:
                    self.text_details.insert(tk.END, f"    {st} --{sym}--> {d}\n")
            self.text_details.insert(tk.END, "\n=== AFD (Construccion) ===\n")
            self.text_details.insert(tk.END, f"  Inicio: {dfa.start_state}\n")
            self.text_details.insert(tk.END, f"  Aceptacion: {dfa.accept_states}\n")
            self.text_details.insert(tk.END, "  Transiciones (AFD):\n")
            for (origin, sym), destino in dfa.transitions.items():
                self.text_details.insert(tk.END, f"    {origin} --{sym}--> {destino}\n")
            self.display_image("temp_dfa.png")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    def display_image(self, path):
        if not os.path.exists(path):
            messagebox.showerror("Error", f"No se encontró {path}")
            return
        try:
            if PIL_AVAILABLE:
                img = Image.open(path)
                img.thumbnail((900, 300))
                self.image = ImageTk.PhotoImage(img)
                self.canvas.delete("all")
                self.canvas.create_image(450, 150, image=self.image)
            else:
                self.canvas.delete("all")
                self.canvas.create_text(450, 150,
                                        text="PIL no está disponible.",
                                        fill="black", font=("Arial", 14))
        except Exception as e:
            messagebox.showerror("Error", str(e))
    def clear_details(self):
        self.text_details.delete("1.0", tk.END)
        self.canvas.delete("all")

def main():
    root = tk.Tk()
    app = RegexApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
