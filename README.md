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

The **Settings** sidebar view stores the source scans and final-store
directories in `~/.config/bookcropper/config.json`. After saving settings, the
source argument is optional:

```sh
python3 manual_tool.py --port 8765
```

CLI values override persisted settings for the current process. Use
`--final-store /path/to/final` to override the final-store directory.

The editor supports rectangular crop handles, draggable crop edges, corner rotation, Option/Alt straighten guides, zoom, corner magnification, filename URL hashes, and the optional ColorChecker correction preset.

**Trim dark scanner edges** is enabled by default. During export it removes
only a thin near-black margin that transitions immediately into clean book
pixels. It is capped at 2.5%/64 pixels per edge and leaves dark covers alone
when no clean transition is found. Disable it for artwork with an intentional
thin black outer border.

### CMS upload

With the JC CMS API running on `http://localhost:3008` and the admin on
`http://jc.localhost:3009`, use the **CMS Upload** panel to:

1. Connect and approve the five-minute pairing code in the CMS admin window.
2. Search for a book by title, author, publisher, or ID.
3. Enter any positive image position.
4. Upload the current corrected crop directly to the selected book.

Filenames ending in `_<number>` select the matching image position
automatically. Occupied positions are locked against replacement until
**Unlock replacement** is explicitly enabled. The CMS token is short-lived and
kept only in browser session storage. **Disconnect** revokes it immediately.

After CMS confirms an upload, the exact 150 DPI JPEG sent to CMS is written to
the configured final-store directory and the original scan is deleted. The
file selector and thumbnails then refresh from the source directory. An
existing final-store filename stops local finalization without deleting the
source scan.

After changing `manual_tool.py`, restart the local server. HTML, CSS, and
JavaScript changes are served directly and only require a browser refresh.

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
