#Day 30: Strassen multiplication
<br>
https://en.wikipedia.org/wiki/Strassen_algorithm
<br>
In linear algebra, the Strassen algorithm, named after Volker Strassen, is an algorithm for matrix multiplication. It is faster than the standard matrix multiplication algorithm and is useful in practice for large matrices, but would be slower than the fastest known algorithms for extremely large matrices.
<br>
Strassen's algorithm works for any ring, such as plus/multiply, but not all semirings, such as min/plus or boolean algebra, where the naive algorithm still works, and so called combinatorial matrix multiplication.
<br>
array:

```
    let a = array![
        [8., 4., 2., 4., 0., 5., 6., 1.],
        [5., 6., 3., 1., 7., 6., 6., 3.],
        [6., 6., 8., 7., 3., 6., 2., 2.],
        [2., 1., 1., 5., 7., 9., 4., 6.],
        [8., 3., 3., 4., 3., 1., 3., 4.],
        [5., 6., 3., 3., 3., 9., 6., 8.],
        [5., 0., 3., 2., 7., 6., 9., 5.],
        [6., 2., 0., 7., 8., 1., 9., 4.]
    ];
    let b = array![
        [7., 9., 2., 3., 3., 1., 6., 1.],
        [1., 4., 0., 1., 0., 6., 8., 0.],
        [2., 7., 3., 6., 5., 7., 0., 4.],
        [6., 7., 4., 4., 5., 2., 2., 6.],
        [9., 5., 1., 6., 0., 0., 1., 8.],
        [0., 4., 0., 7., 3., 6., 9., 2.],
        [8., 5., 6., 9., 0., 3., 1., 7.],
        [9., 5., 1., 2., 8., 9., 5., 2.]
    ];
    let r = strassen(&a, &b);
```

result:

```
[[145, 185, 75, 147, 77, 111, 144, 94],
 [191, 201, 69, 187, 77, 145, 162, 139],
 [167, 242, 81, 182, 127, 172, 167, 134],
 [196, 185, 64, 186, 111, 145, 152, 150],
 [176, 187, 66, 121, 94, 106, 115, 99],
 [212, 232, 78, 202, 136, 212, 214, 135],
 [233, 209, 93, 216, 98, 138, 129, 170],
 [266, 220, 106, 192, 88, 101, 112, 185]]
```