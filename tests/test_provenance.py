from caypollard.provenance import manifest_digest


def test_manifest_digest_is_order_invariant_by_id():
    a = [{"id": "b", "value": 2}, {"id": "a", "value": 1}]
    b = list(reversed(a))
    assert manifest_digest(a) == manifest_digest(b)
