import ast
import os

def check_file(filepath):
    try:
        with open(filepath, "r") as f:
            tree = ast.parse(f.read(), filename=filepath)
        
        expensive_nodes = []
        for node in tree.body:
            # Check for loops, with statements at top level
            if isinstance(node, (ast.For, ast.While, ast.With)):
                expensive_nodes.append((node.lineno, type(node).__name__))
            
            # Check for Expr or Assign involving potentially expensive calls
            call_node = None
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call_node = node.value
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call_node = node.value
            
            if call_node:
                func = call_node.func
                func_name = ""
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name):
                        func_name = f"{func.value.id}.{func.attr}"
                    else:
                        func_name = func.attr
                
                if any(x in func_name for x in ["read_", "subprocess", "call", "run", "load", "open"]):
                    expensive_nodes.append((node.lineno, f"Call: {func_name}"))

        if expensive_nodes:
            print(f"FILE: {filepath}")
            for line, info in expensive_nodes:
                print(f"  Line {line}: {info}")
    except Exception:
        pass

target_dirs = ["matsim", "synthesis", "data", "calibration"]
for root, dirs, files in os.walk("."):
    if any(root.startswith(f"./{d}") or root == d for d in target_dirs) and "__pycache__" not in root:
        for file in files:
            if file.endswith(".py"):
                check_file(os.path.join(root, file))
