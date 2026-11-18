# Art of Japan

A scroll-driven cinematic gallery: river village, portrait gallery, fishing scene, painting gallery,
art sphere and footer, rendered with hand-written WebGL2 (no libraries) with a live water simulation.

## Run
Open `index.html` in a browser. It is one self-contained file (images are embedded).

## Edit
- `src/index.html`  page, styles, WebGL shaders and scroll timeline
- `src/physics.js`  water solver, fishing line, float and droplets
- `assets/`         images used by the page, and `meta.json`
- `build.py`        run `python3 build.py` to rebuild `index.html` after editing (needs Python 3 only)
