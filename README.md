# Examples

3 up and 3 down parameters with univariate moves only:

    ./up-down-multiplier-move-test.py -u 3 -d 3 -a 0 -n 500000 -f 5 -b 1 -x

All is well.
Same as above with multivariate move turned on, and no addend to the power of
the hastings ratio:

    ./up-down-multiplier-move-test.py -u 3 -d 3 -a 0 -n 500000 -f 5 -b 1

Not getting the prior back.
Try subtracting 2 from the power of the hastings ratio:

    ./up-down-multiplier-move-test.py -u 3 -d 3 -a -2.0 -n 500000 -f 5 -b 1

Still not getting the prior back.
