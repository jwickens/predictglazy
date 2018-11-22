from datasets import GlazeCompositionDataset


def test_CompositionDataset():
    ds = GlazeCompositionDataset()
    assert len(ds) > 0
    item = ds[0]
    assert(item[1].sum() == 1)
