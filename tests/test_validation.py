import numpy as np

from pitch_engine.validation import touches_all_edges


def test_full_frame_rectangle_touches_all_edges():
    polygon = np.array([[0, 0], [1279, 0], [1279, 719], [0, 719]])
    assert touches_all_edges(polygon, (720, 1280)) is True


def test_inset_quad_does_not_touch_any_edge():
    polygon = np.array([[100, 100], [1180, 100], [1230, 620], [50, 620]])
    assert touches_all_edges(polygon, (720, 1280)) is False


def test_touching_only_two_edges_is_not_enough():
    polygon = np.array([[0, 0], [1000, 0], [1000, 600], [0, 600]])
    assert touches_all_edges(polygon, (720, 1280)) is False
