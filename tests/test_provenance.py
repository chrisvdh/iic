from iic import __version__
from iic.provenance import runtime_identity, source_identity


def test_source_identity_uses_the_loaded_package_version():
    identity = source_identity()

    assert identity["package_version"] == __version__
    assert len(identity["fingerprint"]) == 64


def test_runtime_identity_has_portable_core_fields():
    identity = runtime_identity()

    assert identity["python"]
    assert identity["platform"]
    assert identity["torch"]
    assert isinstance(identity["cuda_devices"], list)
