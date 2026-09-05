#!/usr/bin/env python3
"""
CSC 461 - Programming Languages and Structures
Assignment: Implementation of Compiler Front-End

Implements:
1. Lexeme separation
2. Lexeme grouping
3. Identifier numbering and lookup table
4. Parsing, parse-tree generation, and attribute-based semantic analysis

The parser follows the supplied grammar and adds the constructs present in
the assignment's main sample: comma-separated declarations, for-loops, and
function calls.
"""

from dataclasses import dataclass, field
from pathlib import Path
import sys


@dataclass
class Token:
    kind: str
    value: str
    position: int


@dataclass
class Node:
    name: str
    children: list = field(default_factory=list)
    typ: str | None = None


class Lexer:
    KEYWORDS = {"int", "float", "char", "return", "for"}
    TWO_CHAR_OPERATORS = {"<=", ">=", "==", "!=", "++", "--", "&&", "||"}
    ONE_CHAR_OPERATORS = {"+", "-", "*", "/", "=", "<", ">"}
    DELIMITERS = {"(", ")", "{", "}", ",", ";"}

    def __init__(self, source: str):
        self.source = source

    def tokenize(self):
        tokens = []
        i = 0
        n = len(self.source)

        while i < n:
            c = self.source[i]

            if c.isspace():
                i += 1
                continue

            if self.source.startswith("//", i):
                end = self.source.find("\n", i)
                i = n if end == -1 else end + 1
                continue

            if self.source.startswith("/*", i):
                end = self.source.find("*/", i + 2)
                if end == -1:
                    raise ValueError("Unterminated block comment.")
                i = end + 2
                continue

            two = self.source[i:i + 2]
            if two in self.TWO_CHAR_OPERATORS:
                tokens.append(Token("OPERATOR", two, i))
                i += 2
                continue

            if c.isalpha() or c == "_":
                start = i
                i += 1
                while i < n and (self.source[i].isalnum() or self.source[i] == "_"):
                    i += 1
                value = self.source[start:i]
                kind = "KEYWORD" if value in self.KEYWORDS else "IDENTIFIER"
                tokens.append(Token(kind, value, start))
                continue

            if c.isdigit():
                start = i
                i += 1
                while i < n and self.source[i].isdigit():
                    i += 1
                kind = "INT_CONST"
                if i < n and self.source[i] == ".":
                    kind = "FLOAT_CONST"
                    i += 1
                    while i < n and self.source[i].isdigit():
                        i += 1
                tokens.append(Token(kind, self.source[start:i], start))
                continue

            if c == "'":
                start = i
                i += 1
                if i >= n or self.source[i] == "'":
                    raise ValueError(f"Invalid character literal at position {start}.")
                if self.source[i] == "\\":
                    i += 2
                else:
                    i += 1
                if i >= n or self.source[i] != "'":
                    raise ValueError(f"Unterminated character literal at position {start}.")
                i += 1
                tokens.append(Token("CHAR_CONST", self.source[start:i], start))
                continue

            if c in self.ONE_CHAR_OPERATORS:
                tokens.append(Token("OPERATOR", c, i))
                i += 1
                continue

            if c in self.DELIMITERS:
                tokens.append(Token("DELIMITER", c, i))
                i += 1
                continue

            raise ValueError(f"Illegal character {c!r} at position {i}.")

        tokens.append(Token("EOF", "", n))
        return tokens


class CompilerFrontEnd:
    TYPES = {"int", "float", "char"}
    ASSIGNMENT_OPS = {"=", "+=", "-=", "*=", "/="}
    RELATIONAL_OPS = {"<", "<=", ">", ">=", "==", "!="}
    ARITHMETIC_OPS = {"+", "-", "*", "/"}

    def __init__(self, source: str):
        self.source = source
        self.tokens = Lexer(source).tokenize()
        self.index = 0
        self.syntax_errors = []
        self.semantic_errors = []
        self.functions = {}
        self.symbol_tables = {}
        self.current_symbols = {}
        self.current_function = None
        self.current_return_type = None

    # ---------- Task 1 and Task 2 ----------
    def lexemes(self):
        return [t.value for t in self.tokens if t.kind != "EOF"]

    def group_lexemes(self):
        functions, variables = [], []
        keywords, operators, constants, others = [], [], [], []

        for i, token in enumerate(self.tokens[:-1]):
            v = token.value

            if token.kind == "KEYWORD":
                keywords.append(v)
            elif token.kind in {"INT_CONST", "FLOAT_CONST", "CHAR_CONST"}:
                constants.append(v)
            elif token.kind == "IDENTIFIER":
                if self.tokens[i + 1].value == "(":
                    functions.append(v)
                else:
                    variables.append(v)
            elif token.kind == "OPERATOR":
                operators.append(v)
            else:
                others.append(v)

        return {
            "Functions": unique(functions),
            "Variables": unique(variables),
            "Keywords": unique(keywords),
            "Operators": unique(operators),
            "Constants": unique(constants),
            "Others": unique(others),
        }

    # ---------- Task 3 ----------
    def identifier_lookup(self):
        groups = self.group_lexemes()
        function_names = set(groups["Functions"])
        mapping, ordered = {}, []

        # Task 3 explicitly says first appearance, so the first lexical
        # occurrence of each variable receives the next identifier number.
        for token in self.tokens:
            if token.kind != "IDENTIFIER" or token.value in function_names:
                continue
            if token.value not in mapping:
                mapping[token.value] = len(mapping) + 1
                ordered.append(token.value)

        lookup = []
        for name in ordered:
            lookup.append((mapping[name], name, self.find_variable_type(name)))

        return mapping, lookup

    def find_variable_type(self, name):
        for table in self.symbol_tables.values():
            if name in table:
                return table[name]
        return "unknown"

    # ---------- Parser helpers ----------
    def current(self):
        return self.tokens[self.index]

    def peek(self, amount=1):
        return self.tokens[min(self.index + amount, len(self.tokens) - 1)]

    def accept(self, value):
        if self.current().value == value:
            token = self.current()
            self.index += 1
            return token
        return None

    def expect(self, value):
        if self.accept(value):
            return True

        found = self.current().value if self.current().value else "EOF"
        self.syntax_errors.append(
            f"Syntax error at '{found}': expected '{value}'."
        )

        # Basic error recovery: consume one token and continue.
        if self.current().kind != "EOF":
            self.index += 1
        return False

    def type_spec(self):
        if self.current().value in self.TYPES:
            value = self.current().value
            self.index += 1
            return value

        found = self.current().value if self.current().value else "EOF"
        self.syntax_errors.append(
            f"Syntax error: expected type (int/float/char), found '{found}'."
        )
        if self.current().kind != "EOF":
            self.index += 1
        return "error"

    # ---------- Task 4: Parsing ----------
    def parse(self):
        root = Node("Program")

        # The supplied sample contains two functions, so the implementation
        # accepts one or more functions although the printed BNF says
        # <Program> ::= <Function>.
        while self.current().kind != "EOF":
            before = self.index
            root.children.append(self.parse_function())
            if self.index == before:
                self.index += 1

        return root

    def parse_function(self):
        return_type = self.type_spec()

        if self.current().kind == "IDENTIFIER":
            name = self.current().value
            self.index += 1
        else:
            name = "<missing_function>"
            self.syntax_errors.append("Syntax error: expected function name.")

        self.expect("(")
        params = []
        if self.current().value != ")":
            params = self.parse_parameter_list()
        self.expect(")")

        self.functions[name] = {
            "return_type": return_type,
            "parameters": [typ for _, typ in params],
        }

        self.expect("{")

        self.current_function = name
        self.current_return_type = return_type

        local_symbols = {}
        for param_name, param_type in params:
            if param_name in local_symbols:
                self.semantic_errors.append(
                    f"Semantic error: parameter '{param_name}' is redeclared."
                )
            local_symbols[param_name] = param_type

        old_symbols = self.current_symbols
        self.current_symbols = local_symbols

        statements = []
        while self.current().kind != "EOF" and self.current().value != "}":
            statements.append(self.parse_statement())

        self.expect("}")
        self.symbol_tables[name] = dict(self.current_symbols)
        self.current_symbols = old_symbols

        function_node = Node(f"Function {name}", typ=return_type)
        function_node.children.append(
            Node("Parameters", [Node(f"{p} : {t}") for p, t in params])
        )
        function_node.children.append(Node("Body", statements))
        return function_node

    def parse_parameter_list(self):
        params = []

        while True:
            param_type = self.type_spec()

            if self.current().kind == "IDENTIFIER":
                param_name = self.current().value
                self.index += 1
                params.append((param_name, param_type))
            else:
                self.syntax_errors.append(
                    "Syntax error: expected parameter identifier."
                )

            if not self.accept(","):
                break

        return params

    def parse_statement(self):
        if self.current().value in self.TYPES:
            return self.parse_declaration()

        if self.current().value == "return":
            return self.parse_return()

        if self.current().value == "for":
            return self.parse_for()

        if (
            self.current().kind == "IDENTIFIER"
            and self.peek().value in self.ASSIGNMENT_OPS
        ):
            return self.parse_assignment()

        self.syntax_errors.append(
            f"Syntax error near '{self.current().value or 'EOF'}'."
        )
        self.synchronize({";", "}"})
        self.accept(";")
        return Node("ErrorStatement")

    def parse_declaration(self, consume_semicolon=True):
        declared_type = self.type_spec()
        items = []

        # Extension required by the assignment's sample: int a, b=5;
        while True:
            if self.current().kind != "IDENTIFIER":
                self.syntax_errors.append(
                    "Syntax error: expected identifier in declaration."
                )
                break

            name = self.current().value
            self.index += 1

            if name in self.current_symbols:
                self.semantic_errors.append(
                    f"Semantic error: variable '{name}' redeclared."
                )
            else:
                self.current_symbols[name] = declared_type

            initializer = None
            if self.accept("="):
                initializer = self.parse_expression()
                source_type = self.infer_type(initializer)
                if not self.assignment_compatible(declared_type, source_type):
                    self.semantic_errors.append(
                        f"Type Compatibility Error: Can't assign "
                        f"{source_type} to {declared_type}."
                    )

            item_children = [initializer] if initializer is not None else []
            items.append(Node(f"{name} : {declared_type}", item_children))

            if not self.accept(","):
                break

        if consume_semicolon:
            self.expect(";")

        return Node(
            "Declaration",
            [Node(f"Type = {declared_type}"), Node("Items", items)],
        )

    def parse_assignment(self, consume_semicolon=True):
        name = self.current().value
        self.index += 1
        op = self.current().value
        self.index += 1

        expression = self.parse_expression()
        expr_type = self.infer_type(expression)

        if name not in self.current_symbols:
            self.semantic_errors.append(
                f"Semantic error: undeclared variable '{name}'."
            )
            target_type = "error"
        else:
            target_type = self.current_symbols[name]

        if not self.assignment_compatible(target_type, expr_type):
            self.semantic_errors.append(
                f"Type Compatibility Error: Can't assign "
                f"{expr_type} to {target_type}."
            )

        if consume_semicolon:
            self.expect(";")

        return Node(f"Assignment {name} {op}", [expression])

    def parse_return(self):
        self.index += 1
        expression = self.parse_expression()
        expression_type = self.infer_type(expression)

        if not self.assignment_compatible(
            self.current_return_type, expression_type
        ):
            self.semantic_errors.append(
                f"Type Compatibility Error: function '{self.current_function}' "
                f"returns {expression_type}, but declared return type is "
                f"{self.current_return_type}."
            )

        self.expect(";")
        return Node("Return", [expression])

    # Extension required by the assignment's main sample:
    # for(int i=0; i<=10; i++) { ... }
    def parse_for(self):
        self.index += 1
        self.expect("(")

        if self.current().value in self.TYPES:
            init = self.parse_declaration(consume_semicolon=True)
        elif (
            self.current().kind == "IDENTIFIER"
            and self.peek().value in self.ASSIGNMENT_OPS
        ):
            init = self.parse_assignment(consume_semicolon=True)
        else:
            self.syntax_errors.append(
                "Syntax error: invalid for-loop initialization."
            )
            self.synchronize({";", ")"})
            self.accept(";")
            init = Node("InvalidForInit")

        condition = self.parse_expression()
        self.infer_type(condition)
        self.expect(";")

        update = self.parse_for_update()
        self.expect(")")
        self.expect("{")

        body = []
        while self.current().kind != "EOF" and self.current().value != "}":
            body.append(self.parse_statement())

        self.expect("}")

        return Node(
            "For",
            [
                init,
                Node("Condition", [condition]),
                update,
                Node("Body", body),
            ],
        )

    def parse_for_update(self):
        if self.current().kind != "IDENTIFIER":
            self.syntax_errors.append(
                "Syntax error: expected identifier in for-loop update."
            )
            self.synchronize({")"})
            return Node("InvalidForUpdate")

        name = self.current().value
        self.index += 1

        if self.current().value in {"++", "--"}:
            op = self.current().value
            self.index += 1

            if name not in self.current_symbols:
                self.semantic_errors.append(
                    f"Semantic error: undeclared variable '{name}'."
                )

            return Node(f"{name}{op}")

        if self.current().value in self.ASSIGNMENT_OPS:
            op = self.current().value
            self.index += 1
            expression = self.parse_expression()
            expr_type = self.infer_type(expression)
            target_type = self.current_symbols.get(name, "error")

            if name not in self.current_symbols:
                self.semantic_errors.append(
                    f"Semantic error: undeclared variable '{name}'."
                )
            elif not self.assignment_compatible(target_type, expr_type):
                self.semantic_errors.append(
                    f"Type Compatibility Error: Can't assign "
                    f"{expr_type} to {target_type}."
                )

            return Node(f"ForUpdate {name} {op}", [expression])

        self.syntax_errors.append(
            f"Syntax error: expected ++, --, or assignment after '{name}'."
        )
        return Node("InvalidForUpdate")

    # ---------- Expression parsing ----------
    def parse_expression(self):
        node = self.parse_term()

        while self.current().value in {"+", "-"}:
            op = self.current().value
            self.index += 1
            right = self.parse_term()
            node = Node(op, [node, right])

        # Needed by the sample's i<=10 loop condition; not printed in the
        # supplied BNF.
        if self.current().value in self.RELATIONAL_OPS:
            op = self.current().value
            self.index += 1
            right = self.parse_expression()
            node = Node(op, [node, right])

        return node

    def parse_term(self):
        node = self.parse_factor()

        while self.current().value in {"*", "/"}:
            op = self.current().value
            self.index += 1
            right = self.parse_factor()
            node = Node(op, [node, right])

        return node

    def parse_factor(self):
        token = self.current()

        if token.kind == "IDENTIFIER":
            name = token.value
            self.index += 1

            # Function call: func(i)
            if self.accept("("):
                args = []

                if self.current().value != ")":
                    args.append(self.parse_expression())
                    while self.accept(","):
                        args.append(self.parse_expression())

                self.expect(")")

                function = self.functions.get(name)
                if function is None:
                    self.semantic_errors.append(
                        f"Semantic error: function '{name}' is not declared."
                    )
                    return Node(f"Call {name}", args, typ="error")

                expected = function["parameters"]

                if len(args) != len(expected):
                    self.semantic_errors.append(
                        f"Semantic error: function '{name}' expects "
                        f"{len(expected)} argument(s), got {len(args)}."
                    )

                for arg, expected_type in zip(args, expected):
                    actual_type = self.infer_type(arg)
                    if not self.assignment_compatible(
                        expected_type, actual_type
                    ):
                        self.semantic_errors.append(
                            f"Type Compatibility Error: argument type "
                            f"{actual_type} cannot be passed to {expected_type} "
                            f"in function '{name}'."
                        )

                return Node(
                    f"Call {name}",
                    args,
                    typ=function["return_type"],
                )

            if name not in self.current_symbols:
                self.semantic_errors.append(
                    f"Semantic error: undeclared variable '{name}'."
                )
                return Node(name, typ="error")

            return Node(name, typ=self.current_symbols[name])

        if token.kind == "INT_CONST":
            self.index += 1
            return Node(token.value, typ="int")

        if token.kind == "FLOAT_CONST":
            self.index += 1
            return Node(token.value, typ="float")

        if token.kind == "CHAR_CONST":
            self.index += 1
            return Node(token.value, typ="char")

        if self.accept("("):
            expression = self.parse_expression()
            self.expect(")")
            return expression

        found = token.value if token.value else "EOF"
        self.syntax_errors.append(
            f"Syntax error: expected expression, found '{found}'."
        )
        if token.kind != "EOF":
            self.index += 1
        return Node("error", typ="error")

    # ---------- Attribute grammar / semantic analysis ----------
    def infer_type(self, node):
        if node is None:
            return "error"

        if node.typ is not None:
            return node.typ

        if node.name in self.ARITHMETIC_OPS:
            left_type = self.infer_type(node.children[0])
            right_type = self.infer_type(node.children[1])

            if "error" in {left_type, right_type}:
                node.typ = "error"
                return "error"

            if left_type == right_type:
                node.typ = left_type
                return left_type

            if {left_type, right_type} == {"int", "float"}:
                node.typ = "float"
                return "float"

            self.semantic_errors.append(
                f"Type Compatibility Error: incompatible operands "
                f"{left_type} and {right_type}."
            )
            node.typ = "error"
            return "error"

        if node.name in self.RELATIONAL_OPS:
            left_type = self.infer_type(node.children[0])
            right_type = self.infer_type(node.children[1])

            if "error" in {left_type, right_type}:
                node.typ = "error"
                return "error"

            if left_type not in {"int", "float"} or right_type not in {"int", "float"}:
                self.semantic_errors.append(
                    "Type Compatibility Error: relational operands must be numeric."
                )
                node.typ = "error"
                return "error"

            node.typ = "int"
            return "int"

        node.typ = "error"
        return "error"

    @staticmethod
    def assignment_compatible(target, source):
        if target == "error" or source == "error":
            return True

        if target == "int":
            return source == "int"

        if target == "float":
            return source in {"int", "float"}

        if target == "char":
            return source == "char"

        return target == source

    def semantic_ok(self):
        return not self.syntax_errors and not self.semantic_errors

    # ---------- Parse-tree reporting ----------
    def parse_tree_lines(self, node, prefix="", is_last=True):
        connector = "└── " if is_last else "├── "
        label = node.name
        if node.typ is not None:
            label += f" : {node.typ}"

        lines = [prefix + connector + label]
        children = [c for c in node.children if c is not None]

        for i, child in enumerate(children):
            child_prefix = prefix + ("    " if is_last else "│   ")
            lines.extend(
                self.parse_tree_lines(
                    child,
                    child_prefix,
                    i == len(children) - 1,
                )
            )
        return lines

    def synchronize(self, stop_values):
        while (
            self.current().kind != "EOF"
            and self.current().value not in stop_values
        ):
            self.index += 1


def unique(items):
    result = []
    seen = set()

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def numbered_grouped_output(frontend):
    groups = frontend.group_lexemes()
    mapping, lookup = frontend.identifier_lookup()

    variables_with_ids = [
        f"<{mapping[name]} id>"
        for name in groups["Variables"]
        if name in mapping
    ]

    return groups, variables_with_ids, lookup


def run(source):
    frontend = CompilerFrontEnd(source)

    # Task 1
    lexemes = frontend.lexemes()

    # Parsing also constructs the symbol tables used by Task 3.
    tree = frontend.parse()

    # Tasks 2 and 3
    groups, variables_with_ids, lookup = numbered_grouped_output(frontend)

    print("=" * 70)
    print("TASK 1 - LEXEME SEPARATION")
    print("=" * 70)
    for lexeme in lexemes:
        print(lexeme)

    print("\n" + "=" * 70)
    print("TASK 2 - LEXEME GROUPING")
    print("=" * 70)
    for key in [
        "Functions", "Variables", "Keywords",
        "Operators", "Constants", "Others"
    ]:
        values = groups[key]
        print(f"{key}: {', '.join(values) if values else '-'}")

    print("\n" + "=" * 70)
    print("TASK 3 - IDENTIFIER NUMBERING AND LOOKUP TABLE")
    print("=" * 70)
    print("Functions:", ", ".join(groups["Functions"]) or "-")
    print("Variables:", ", ".join(variables_with_ids) or "-")
    print("\nLookup table:")
    print("ID\tVariable\tType")
    for ident, name, typ in lookup:
        print(f"{ident}\t{name}\t{typ}")

    print("\n" + "=" * 70)
    print("TASK 4 - PARSE TREE")
    print("=" * 70)
    for line in frontend.parse_tree_lines(tree):
        print(line)

    print("\n" + "=" * 70)
    print("SEMANTIC ANALYSIS")
    print("=" * 70)

    if frontend.syntax_errors:
        print("Syntax errors:")
        for error in unique(frontend.syntax_errors):
            print("-", error)

    if frontend.semantic_errors:
        print("Semantic errors:")
        for error in unique(frontend.semantic_errors):
            print("-", error)

    if frontend.semantic_ok():
        print("Semantic checking result: PASSED")
    else:
        print("Semantic checking result: FAILED")

    return frontend


def read_source():
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).read_text(encoding="utf-8")

    print(
        "Enter source code. Finish with EOF "
        "(Ctrl+D on Linux/macOS or Ctrl+Z on Windows):"
    )
    return sys.stdin.read()


if __name__ == "__main__":
    run(read_source())
