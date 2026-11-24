# Editable manuscript figures

The three original PowerPoint slides, SVGs and manuscript PDFs are in this directory. Text and graphical elements remain editable in the combined [PowerPoint deck](manuscript_figures.pptx). The [crop ledger](figure_crops.json) gives each figure's original print dimensions, position in the combined deck, and source-slide SHA-256. Coordinates are measured in points from the upper left.

| Figure | Recorded inputs | What to check |
|---|---|---|
| 1 | Official MLVU test item `AC_8` in `external_eval/items.json` and `external_eval/calls.jsonl` | Three available reports of 3, 3 and 4 determine a final count of 3. After two reports the possible final counts are 1 and 3. |
| 2 | `reproduce.py` output `figure_data.sampling_control` | On 200 matched development questions, the best single call and four-report estimate score 73.5% and 78.5% on the primary inputs and 64.0% and 67.0% on random subsets. |
| 3 | `held_out_errors/results.json` field `figure_counts` | The archived estimator corrects eight answers and introduces five errors on 103 held-out videos. Its remaining errors comprise 16 undercounts and five overcounts. |

Sky blue (`#478BCA`) marks inputs, single calls and final error direction. Amethyst (`#6E5BA3`) marks accumulated evidence, four-report estimates and corrections. Champagne (`#FED479`) marks the exact-match score in Figure 1. Red (`#EB2124`) traces the request for another report in Figure 1 and marks new errors in Figure 3.

Figure 1 combines the frozen frame selections, each uniform report's score for the five candidate counts, the accumulated scores and the worst final lead over each rival. Its drawing script reads `external_eval/selected_frames.jsonl` and `external_eval/planned_frame_inputs.jsonl` for the observed sampling pattern. The fourth input was actually acquired in the evaluation and is skipped in the stopping replay.

## Verify the data and rebuild the deck

Run from the extracted package root. The figure dependencies are listed in `figures/requirements.txt`. Python reads the supplied evaluation records locally. The source checksum in each crop-ledger record identifies the exact slide used for the distributed deck.

```sh
python -B figures/verify_case.py --package . --out redraw/mechanism_case.json
python -B figures/draw_overview.py . redraw
python -B figures/draw_charts.py --package . --out redraw
python -B figures/merge_figures.py figures/overview.pptx figures/sampling_control.pptx figures/count_errors.pptx --output redraw/manuscript_figures.pptx --crop-json redraw/figure_crops.json
```

Compare `redraw/mechanism_case.json`, `redraw/figure_values.json` and `redraw/chart_data.json` with the supplied JSON files. The regenerated editable deck and crop ledger can also be compared byte for byte when the three supplied source slides are used. The merger keeps the print scale of the three individual slides and sets the canvas to their greatest width and height. It changes only the origin of the top-level drawing objects when centering each figure.

## Export the editable slides

To regenerate individual vector PDFs from the combined deck, run the following command from the package root with LibreOffice Impress on the command path.

```sh
python -B figures/export_figures.py --out redraw/pdf
```

PowerPoint users can export the combined presentation as a PDF at its original slide size, with one slide per page. Pass the resulting file to the same cropper.

```sh
python -B figures/export_figures.py --export exported_slides.pdf --out redraw/pdf
```

The supplied PDFs preserve the fonts chosen by the original renderer. Other installations may substitute a metrically compatible typeface. The chart drawing script also emits SVGs and a direct vector PDF from the underlying data. Arimo Regular and Bold provide portable Arial-compatible glyph metrics for this route. Font licensing accompanies the files in `fonts/`.

## PowerPoint conversion

The local `figgenie/` helpers are a subset of [FigGenie](https://github.com/Sleepy-Avacado/FigGenie-paper-diagram-skill/tree/c136a6aeefc6f6fe11fc615a5199cca57be3322a/skills/figgenie-paper-diagram). This copy resolves font files from `figures/fonts/`. The [FigGenie MIT license](FIGGENIE_LICENSE.txt) covers these helpers, and the [Arimo Apache license](fonts/LICENSE-Arimo.txt) covers the two font files. The figure scripts use text, boxes, lines and vector paths supported by the bundled converter.

```sh
python -B figures/figgenie/embed_fonts.py redraw/sampling_control.svg --font arial --no-cjk --strict -o redraw/sampling_control.embedded.svg
python -B figures/figgenie/svg2pptx.py redraw/sampling_control.svg -o redraw/sampling_control.raw.pptx
python -B figures/finalize_native.py redraw/sampling_control.raw.pptx redraw/sampling_control.pptx
```
