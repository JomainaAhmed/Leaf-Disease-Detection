"""
Streamlit App Entrypoint.
Delegates to app/app.py for seamless compatibility with both:
  streamlit run main.py
and:
  streamlit run app/app.py
"""
import os
import runpy

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(__file__), "app", "app.py")
    runpy.run_path(app_path, run_name="__main__")
