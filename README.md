# TODO

-3. Create Prods with just one argument inline - check if they can be intersected and add custom __init__?
Yes: in __post__init - if a an element can be created with just one element, allow one cut instance - subtype based on that element if it is just initalize it - also allow one to pas in the inference

Rule: for products with no recursion to external type, do them first, try to call their constructor.
Maybe better: try to coerce by calling constructor if any check fails.

-2. Consistent error and exception rules.
-1. Computeable defaults - for the optionals, I should be able to compute. But where? New risks hitting bad defaults. 
1. Iteration
0. Objects with IDs: substitution (Partially there via stamp, but it feels rather... artificial)
2. Binders
3. Extension of existing ADTs
4. Update docs
6. Vistior patterns
5. Custom show or at least better show
7. Python front end instead of text
8. Property inference
9. Functorial IRs
10. Pattern matching over an IR that does not exist yet?
11. Consistent module definition.
12. Logging intergration
13. Tests


