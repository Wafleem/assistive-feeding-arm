import pytest

# Spec tests for sections that are scaffolded but not implemented yet. They are
# reported as "xfail" while the code raises NotImplementedError. When you implement
# a section, delete the @todo mark on its tests: they must then pass.
todo = pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="TODO: section not implemented yet")
