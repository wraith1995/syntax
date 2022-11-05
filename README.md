# TODO (in no particular order)
0. Resolve serious disagreements on the nature of validation beyond types. The possibility of internal type conversion, types of types and their handling, shittyness of python typing module.
1. Different types of iteration methods: depth vs. breadth, external or internal, dup vs no dup, dup notion.
2. The problem of matching over an IR that does not yet exist.
3. Binders unbound
4. Extension of existing ADTs or merging of ADTS
5. Basic documentation.
6. Vistior patterns
7. Caching dec for ad-hoc polymorphism.
8. Custom show or at least better show. 
9. Python front end instead of text. See (2).
10. Functorial IRs
11. Logging intergration
12. Github integrations
13. Variations on the lambda calc implementation for examples. Add these to tests.
14. Attrs vs. Dataclasses. Attrs might save some boiler plate, but it outside standard python.
15. Greater variation on internal errors and consistenty with python.
16. psf black
17. Maps

Random notes:
    """
    Collections.abc
    Ideal Itter paramters:
    Internal vs. SumLocal vs. External: do we loop over just our types or also include outside types
    Include Nones vs No Nones
    Names vs nonames: do we include what field we come from if any.
    Order: Post-order vs. pre-order vs. in order. (d vs b)
    Flatten vs non-flatten: do we flatten lists that we find or not.
    Order?

    Other features:
    Copying
    Disjointness, containedness (Set of terms -> support set interface)
    isomorphism (__itter__ is the same.)
    Mapping (what do we need for CG)
    Folding
    Visitor pattern/rewriter pattern.

    IR:
    Ref mutability
    Other mutability.
    Partial frozen.n
    Type Checkers/other adt validators
    Functions: Recursion
    Compatability with mtype/pyright.
    """
