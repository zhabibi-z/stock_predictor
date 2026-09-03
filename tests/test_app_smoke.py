"""
Headless smoke test for the Streamlit dashboard, using Streamlit's own
AppTest framework.

This exists because a real bug shipped invisibly for the app's entire
history: app.py called st.image(..., use_container_width=True), which
does not exist on streamlit==1.35.0 (the version actually pinned in
requirements.txt) — only on the newer version that had drifted into local
dev environments. Nothing ever launched the app against the real pinned
dependency, so nobody caught it until it broke in production. This test
launches the actual app and clicks the actual button, so a regression
like that fails locally and in CI instead of only in front of a user.
"""

import os

# Some TensorFlow builds deadlock under default multi-threaded op execution on
# this class of machine (the Eager executor waits on its own Eigen thread pool
# that never receives dispatched work) — reproduced with tensorflow==2.21.0,
# absent with the pinned tensorflow==2.18.0. This is the only test module that
# actually trains an MLP (via app.py's pipeline), so the workaround is scoped
# here rather than in conftest.py, which would force an eager TensorFlow
# import for every test file and undo Phase 4's lazy-import fix.
try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except ImportError:
    pass

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_loads_without_exceptions():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"app.py raised on initial load: {[e.value for e in at.exception]}"


def test_app_runs_pipeline_without_exceptions():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    run_button = next(b for b in at.button if b.label == "Run Quant Pipeline")
    run_button.set_value(True)
    at.run(timeout=120)

    assert not at.exception, f"app.py raised after running the pipeline: {[e.value for e in at.exception]}"
