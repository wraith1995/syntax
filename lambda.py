from adt import ADT
from auxtypes import stamp
def is_valid_name(x):
    return type(x) is str

#We need to ensure simple conversion in the matching...

L = ADT("""
module LAM {

var = (name vname, name id)

expr = App (expr lhs, expr rhs)
     | Var (var arg)
     | Lam (var arg, expr body)
}
""", ext_checks={"name" : is_valid_name}, ext_types={'name' : str,
                                                     'stamp' : stamp}, defaults = {'stamp' : stamp()})


a = L.var("a", "b")
print(a)
App = L.App(L.Var(a), L.Var(a))
print(App)



def evp(a : L.expr, env : dict[str, L.expr]) -> L.expr:
    match a:
        case L.Var(name):
            if name in env:
                return env[name]
            else:
                return L.Var(name)
        case L.Lam(a, b):
            return L.Lam(a, b)
        case L.App(L.Lam(name, body), arg):
            envp = env.copy() #Use immutable dictionaries, frozendict to avoid this -> make an immutable dict with sharing.

            envp[name] = arg
            return evp(body, envp)
        case L.App(f, x):
            fp = evp(f, env)
            match fp:
                case L.Lam(_, _):
                    return evp(L.App(fp, x))
                case _:
                    return L.App(fp, x)

q1 = L.App(L.Lam("a", L.Var("a")), L.Var("b"))
q2 = L.App(L.Lam("a", L.Var("a")), L.Var("b"))
print(q1 is q2)
b = evp(q1, {})
print(b)
