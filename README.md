# Affine Invariant Function Sandbox
"A function *f* on a set *S* is said to be invariant under a transformation *T* of *S* into iteself if *f(T(x)) = f(x)* for all *x* in *S*". [[1]]

I define an affine invariant function to be a function where there exists an affine transformation *Ax + b* where *A &ne; **0*** invertible, *b &ne; **0*** such that
*f(Ax + b) = f(x)* for all *x* in the domain of the function with *Ax + b* also in the domain of the function.

This is like an extension of periodic functions *f(x + P) = f(x)* [[2]], which is the special 1d case with *A = 1*, *b = P*.

This repository contains a sandbox of python (and some matlab) scripts and tools I have used to explore and generate images and animations of affine invariant 2d surface functions. That is functions *f: **R***x***R** &rarr; **R***.
This includes affine invariant functions were the affine transformation is scale, and/or rotation, and/or translation. 
Like a log spiral snail shell where *f(e<sup>&theta;</sup>[cos(&theta;) -sin(&theta;); sin(&theta;) cos(&theta;)]**x** + [0; 0]) = f(**x**)*

## Gallery
<img src="https://raw.githubusercontent.com/nmillerns/affine_invariant_functions/master/figs/tran_rot.gif" height=300> <img src="https://raw.githubusercontent.com/nmillerns/affine_invariant_functions/master/figs/snailshell.png" height=300>

<img src="https://raw.githubusercontent.com/nmillerns/affine_invariant_functions/master/figs/scaled_tran.gif" height=400> <img src="https://raw.githubusercontent.com/nmillerns/affine_invariant_functions/master/figs/smooth_magic.gif" height=400>

[1]: https://encyclopedia2.thefreedictionary.com/invariant+function 
[2]: https://en.wikipedia.org/wiki/Periodic_function
