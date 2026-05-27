import pyccl


def test_pyccl_import():
    cosmo = pyccl.CosmologyVanillaLCDM()
    assert cosmo is not None
