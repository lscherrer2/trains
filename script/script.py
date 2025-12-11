def test_fn(i: int):
    if i > 0:
        test_fn(i - 1)


test_fn(3)
