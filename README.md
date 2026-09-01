# Book Cover Fixer

Local review and crop tool for book scans.

## Manual review tool

Install the dependencies:

```sh
python3 -m pip install -r requirements.txt
```

Start the local editor:

```sh
python3 manual_tool.py /path/to/scanned --output manual_crops --port 8765
```

Open <http://127.0.0.1:8765>.

The editor supports rectangular crop handles, draggable crop edges, corner rotation, Option/Alt straighten guides, zoom, corner magnification, filename URL hashes, and the optional ColorChecker correction preset.

## Automatic suggestions

`unified_detect.py` and `run_unified_all.py` contain the current experimental computer-vision detector. It should propose starting values for the manual editor rather than silently replacing manual review.

`ai_review.py` is an optional fallback for difficult scans. API keys belong in `.env`, which is ignored by Git. AI results are advisory and should not be used for blind cropping.

## Project layout

- `manual_tool.py`: local HTTP server and full-resolution crop/export processing
- `manual_tool.html`: editor markup
- `manual_tool.css`: editor styles
- `manual_tool.js`: editor interaction logic
- `unified_detect.py`: current detector ensemble
- `cutout_all.py`: detector-based batch crop experiment
- `ai_review.py`: optional vision-model review
- `compare_results.py`: comparison report for AI review results
