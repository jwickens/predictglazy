from datasets import GlazeCompositionDataset, GlazeColor2CompositionDataset


def test_CompositionDataset():
    ds = GlazeCompositionDataset()
    assert len(ds) > 0
    item = ds[0]
    assert(item[1].sum() > 0)


def test_Color2CompositionDataset():
    ds = GlazeColor2CompositionDataset()
    assert len(ds) > 0
    item = ds[0]
    # shold be array of RGB, RGB values
    assert(len(item[0]) == 6)
    for _ in ds:
        pass
