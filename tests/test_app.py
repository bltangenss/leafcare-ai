"""Legacy Flask tests were superseded by test_api.py and test_predictor.py.

The earlier tests imported a real dataset and modified production history.db and
uploads. All active HTTP tests now inject a predictor and use temporary storage;
model-quality evaluation lives in the explicit evaluation scripts.
"""
