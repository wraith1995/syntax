from adt import ADT
from auxtypes import stamp
def is_valid_name(x):
    return type(x) is str

#We need to ensure simple conversion in the matching...

L = ADT("""
module LAM {

var = (name vname, stamp? id)

expr = App (expr lhs, expr rhs)
     | Var (var arg)
     | Lam (var arg, expr body)
}
""", ext_checks={"name" : is_valid_name}, ext_types={'name' : str,
                                                     'stamp' : stamp}, defaults = {stamp : lambda: stamp()})




q = L.Var("a")
print(q)
a = L.var("a")
b = L.var("a")
print(a,b)
App = L.Lam(a, L.Var(a))
Appp = L.Lam(a, L.Var(a))
print(App)
print(Appp)
print(App is Appp)
oapp = App.__copy__()
print(oapp is not App)
print(oapp)
print("Lambda eval test")
#Properties?
def freevars(e : L.expr) -> set[L.var]:
    match e:
        case L.Var(v):
            return set([v])
        case L.Lam(v, e):
            r = freevars(e)
            return r.difference(set([v]))
        case L.App(e1, e2):
            r1 = freevars(e1)
            r2 = freevars(e2)
            return r1.union(r2)
#Alpha conversion... change the names of all bound vars
def alpha(e : L.expr) -> L.expr:
    fvs = freevars(e)
    copies = {x : x for x in fvs}
    return e.__copy__(copies=copies) # copy everything except these.

#write alpha convert version
def subst(x : L.var, s : L.expr, e : L.expr) -> L.expr:
    match e:
        case L.Var(a):
            if a == x:
                return s
            else:
                return L.Var(a)
        case L.App(e1, e2):
            return L.App(subst(x, s, e1), subst(x, s, e2))
        case L.Lam(v, body):
            if v == x:
                return e
            else:
                fvs = freevars(s)
                if v in fvs:
                    vp = v.copy()
                    return L.Lam(vp, subst(x, s, subst(vp, L.Var(vp), body)))
                else:
                    return L.Lam(v, subst(x, s, body))


def evp(a : L.expr) -> L.expr:
    match a:
        case L.Var(var):
            return L.Var(var)
        case L.Lam(a, b):
            return L.Lam(a, b)
        case L.App(L.Lam(var, body), arg):
            return subst(var, arg, body)
        case L.App(f, x):
            fp = evp(f)
            match fp:
                case L.Lam(_, _):
                    return evp(L.App(fp, x))
                case _:
                    return L.App(fp, x)

vara = L.var("a")
varb = L.var("b")
term = L.App(L.Lam(vara, L.Var(vara)), L.Var(varb))
b = evp(term)
print(term)
print("->")
print(b)
