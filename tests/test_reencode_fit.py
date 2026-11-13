import pytest

pytest.importorskip("pyarrow")
from thesis.datasets.mixture.reencode import target_shape


@pytest.mark.parametrize("shape, out", [
    ((480, 640, 3), (240, 320)),     # toto: halved
    ((240, 320, 3), (240, 320)),     # hydra: already there
    ((150, 200, 3), (150, 200)),     # taco static: never enlarged
    ((256, 256, 3), (240, 240)),     # robocook: height-bound, square kept
    ((84, 84, 3), (84, 84)),
    ((225, 301, 3), (224, 300)),     # odd sizes round down to even
])
def test_fit_keeps_aspect_and_never_enlarges(shape, out):
    assert target_shape(shape, fit=(320, 240)) == out


def test_size_squashes_to_a_square():
    assert target_shape((480, 640, 3), size=224) == (224, 224)
