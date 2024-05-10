from ADT import ADT
from typing import Union
from collections.abc import Mapping


# #QUESTIONS:
#1. in beta reduction ,should i convert back to L format?


# # Lambda expression ex:  lambdax.x
# # Lambda app ex: lambdax.x y
# # Lambda Variable ex: x

L = ADT(
    """
module LAMBDACALC {


var = (name vname, stamp id)

expr = App (expr lhs, expr rhs)
     | Var (var arg)
     | Lam (var arg, expr body)
     

}

""",
)

print("L TEST,",L)

rename_to= {}

#Convert to BTerm in alpha and then back to L ADT in beta 

"""
Alpha Conversion using De Bruijn index
"""
def convert(t, rename_to={}):
  
    match t:
        
        case L.Variable(name):
            #base case : 
            return BTermVar(rename_to.get(name, len(rename_to)))

            # return L.Variable(rename_to.get(name, str(len(rename_to)))) 

        case L.Expression(arg, body):
            #access the dictionary, and set the value associated with body to the index
            # rename_to[arg]=str(len(rename_to))
            index = len(rename_to)
            new_rename_to = {**rename_to, arg: index}
            
            # Convert the body using the updated environment
            converted_body = convert(body, new_rename_to)
            
            # Return the expression with the new index for 'arg'
            return BTermLambda(converted_body)
            
    
        case L.Application(t1, t2):
            return BTermApplication(convert(t1, rename_to), convert(t2, rename_to))
            
        




from typing import Union, Dict, Any, List


class BTermVar:
    def __init__(self, index):
        self.index = index

class BTermLambda:
    def __init__(self, body):
        self.body = body

class BTermApplication:
    def __init__(self, term1, term2):
        self.term1 = term1
        self.term2 = term2



def beta_reduction(t, de_bruijn_level=0, subst=None):
    if subst is None:
        subst = {}

    if isinstance(t, BTermVar):
        return subst.get(t.index, t)
    elif isinstance(t, BTermLambda):
        new_subst = dict(subst)
        new_subst[de_bruijn_level] = BTermVar(de_bruijn_level)
        return BTermLambda(beta_reduction(t.body, de_bruijn_level + 1, new_subst))
    elif isinstance(t, BTermApplication):
        if isinstance(t.term1, BTermLambda):
            reduced_term = beta_reduction(t.term1.body, de_bruijn_level + 1, subst)
            subst[de_bruijn_level] = t.term2
            return beta_reduction(reduced_term, de_bruijn_level, subst)
        else:
            raise ValueError("Cannot do application on non-lambda term")


#TEST: should go to lambda0.lambda 1 0
app= L.Application(L.Expression("x",L.Variable("x")),L.Expression("y",L.Application(L.Variable("x"),L.Variable("y"))))
result=convert(app,rename_to)
print(result.term1.body.index)
print(result.term2.body.term1.index)
print(result.term2.body.term2.index)

print(rename_to)

print("BETA REDUCTION",beta_reduction(result))


test= L.Application(L.Expression("x",L.Application(L.Variable("x"),L.Variable("x"))),L.Expression("z",L.Variable("u")))
con_result= convert(test)
print("CON", con_result.term1.body)

con_result_beta= beta_reduction(convert(test))
print(con_result_beta)



#how to represent low-level