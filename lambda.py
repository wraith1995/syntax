from adt import ADT

def is_valid_name(x):
    return type(x) is str

#We need to ensure simple conversion in the matching...

L = ADT("""
module LAM {

expr = App (expr lhs, expr rhs)
     | Var (name vname)
     | Lam (name arg, expr body)
}
""", ext_checks={"name" : is_valid_name}, types={'name' : str})


a = L.Var("a")
App = L.App(a, a)
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
            envp = env.copy()
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
